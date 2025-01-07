from collections import deque

import numpy as np
import torch
import torch.distributed
from torch.utils.tensorboard import SummaryWriter
from omegaconf import DictConfig

from qinco.utils import save_model
from qinco.metrics import MetricsManager, TimersManager


####################################################################
# Logging utilities
####################################################################

def get_metric_logger(cfg):
    logger = MetricLogger(cfg)
    if cfg._melog is not None: # Load state dict
        logger.load_state_dict(cfg._melog)
    cfg._melog = logger
    return logger

def conf_to_tensorboard_table(cfg):
    def conf_to_matrix(cfg):
        if isinstance(cfg, DictConfig):
            items = []
            for key, val in cfg.items():
                sub_items = conf_to_matrix(val)
                sub_items = [[key + '.' + k if k else key, v] for k, v in sub_items]
                items.extend(sub_items)
            return items
        else:
            return [['', str(cfg)]]
    items = conf_to_matrix(cfg)
    lines = [
        '| *Key* | *Value* |'
        '|---------|---------|'
    ] + [f'| {k} | {v} |' for k, v in items]
    lines = '\n'.join(lines)
    return lines

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value
        )

class SmoothedMetrics:
    def __init__(self):
        self.metrics = {}

    def create(self, metric_name, **kwargs):
        self.metrics[metric_name] = SmoothedValue(**kwargs)

    def update(self, n=1, **metrics):
        for name, value in metrics.items():
            if name not in self.metrics:
                self.metrics[name] = SmoothedValue(fmt="{global_avg:g}")
            self.metrics[name].update(value, n=n)
        return {name: self.get(name) for name in self.metrics.keys()}

    def get(self, name):
        return self.metrics[name]



####################################################################
# Logging manager
####################################################################

class MetricLogger:
    def __init__(self, cfg):
        self.cfg = cfg
        self.accelerator = cfg._accelerator
        self.eps = 1e-6

        self.stop_patience = cfg.scheduler.stop_patience
        self.checkpoint_path = cfg.output

        self.best_mse = float('inf')
        self.best_mse_epoch = cfg._cur_epoch - 1
        self.cur_epoch = cfg._cur_epoch - 1
        self.cur_step = 0
        self.last_lr = 0
        self.last_lr_change = cfg._cur_epoch - 1
        self.timers = TimersManager('train', 'epoch', 'eval', 'total')
        self.metrics = MetricsManager(cfg, self.accelerator.device, ['train', 'val'])
        self.smooth = SmoothedMetrics()

        self.timers.total.start()

        if self.accelerator.is_main_process and cfg.task == 'train' and cfg.tensorboard:
            self.writer = SummaryWriter(log_dir=cfg.tensorboard)
        else:
            self.writer = None
        self.write_initial_logs()

    def write_initial_logs(self):
        cfg = self.cfg
        accelerator = self.accelerator
        effective_bs = accelerator.num_processes * cfg.batch * cfg.grad_accumulate
        if 'eval' in cfg.task:
            accelerator.print(f'Starting evaluation')
        else:
            accelerator.print(f'Starting training')
            accelerator.print(f'Batch size of {cfg.batch} with {cfg.grad_accumulate} acumulation step(s) and {accelerator.num_processes} process(es)')
            accelerator.print(f'Effective batch size: {effective_bs}')
            accelerator.print(f'Will start with a learning rate of {cfg.lr} controlled by the scheduler {cfg.scheduler.name}')

            if self.writer:
                self.writer.add_custom_scalars({
                    "training": {
                        "loss": ["Multiline", ["Loss/train"]],
                        "accuracy": ["MSE", ["MSE/val", "MSE/RQ"]],
                        "hparams": ["LR", ["hparam/lr"]],
                    },
                })

    def state_dict(self):
        return {
            'timers': self.timers.state_dict(),
            'cur_step': self.cur_step,
            'best_mse': self.best_mse,
            'best_mse_epoch': self.best_mse_epoch,
            'last_lr': self.last_lr,
            'last_lr_change': self.last_lr_change,
        }

    def load_state_dict(self, state):
        self.timers.load_state_dict(state['timers'])
        self.cur_step = state['cur_step']
        self.best_mse = state['best_mse']
        self.best_mse_epoch = state['best_mse_epoch']
        self.last_lr = state['last_lr']
        self.last_lr_change = state['last_lr_change']

    # ===== train logging =====

    def _metric_tag(self, name, subtag):
        if '_' in name:
            parts = name.split('_')
            name = parts[-1]
            subtag = subtag + '_' + '_'.join(parts[:-1])

        return f'{name}/{subtag}'

    @torch.no_grad
    def end_epoch(self, model, val_mse):
        mean_loss = self.sum_loss / self.epoch_n_samples
        mean_all_losses = {k: v / self.epoch_n_samples for k, v in self.sum_all_losses.items()}
        losses_str = ' ; '.join([f"{k}={v:g}" for k, v in mean_all_losses.items()])

        if val_mse < self.best_mse - self.eps:
            self.best_mse = val_mse
            self.best_mse_epoch = self.cur_epoch

        m_vals = self.metrics.last_m_vals
        self.accelerator.print(f"[T_total={self.timers.total} | T_train={self.timers.train} | T_epoch={self.timers.epoch}] End of epoch {self.cur_epoch} ({self.cur_step} steps) train loss {mean_loss:g}")
        self.accelerator.print(f"All losses: [[{losses_str}]]")
        self.accelerator.print("Validation metrics:", self.metrics.metrics_as_str())
        self.accelerator.print("Best metrics:", self.metrics.bests_as_str())


        if self.best_mse_epoch == self.cur_epoch:
            self.accelerator.print(f"Best validation MSE so far, storing model to {self.checkpoint_path}")
            save_model(self.cfg, self.accelerator, model)

        if self.writer is not None:
            self.writer.add_scalar('Loss/mean', mean_loss, self.cur_step)
            self.writer.add_scalar('Step/elapsed_epochs', self.cur_epoch+1, self.cur_step)
            for name, v in m_vals.items():
                self.writer.add_scalar(self._metric_tag(name, 'val'), v, self.cur_step)
            for name, v in self.metrics.bests.items():
                self.writer.add_scalar(self._metric_tag(name, 'best'), v, self.cur_step)
            if self.cfg._rq_mse:
                self.writer.add_scalar('MSE/RQ', self.cfg._rq_mse, self.cur_step)
            self.writer.flush()
        
        # Show bits usage and entropy
        entropy = self.metrics.compute_codes_entropy()
        train_e, train_e_min = np.mean(entropy['train']), min(entropy['train'])
        val_e, val_e_min = np.mean(entropy['val']), min(entropy['val'])

        self.accelerator.print(
            f'train_codeword_entropy={train_e:g} (min={train_e_min:g})'
            f'  |  val_codeword_entropy={val_e:g} (min={val_e_min:g})'
            f'  |  step_entropies=[' + ', '.join([f'{e:.2f}' for e in entropy['train']]) + ']'
        )

        if self.writer is not None:
            self.writer.add_scalar('entropy/train_entropy', train_e, self.cur_step)
            self.writer.add_scalar('entropy/val_entropy', val_e, self.cur_step)

        uwn_model = self.accelerator.unwrap_model(model)
        uwn_model.reset_unused_codebooks(self.metrics.compute_codes_usage())

    def should_stop(self):
        if self.cur_epoch - self.best_mse_epoch > self.stop_patience:
            self.accelerator.print(f"Val loss did not improve for {self.stop_patience} steps, stopping")
            return True
        if self.cfg.scheduler.name == 'cosine' and self.cur_epoch >= self.cfg.epochs + self.cfg.scheduler.stop_patience:
            self.accelerator.print(f"Reached maximum epochs for cosine scheduler, stopping")
            return True
        return False

    def mark_end_training(self):
        self.accelerator.print(f"[T_total={self.timers.total} | T_train={self.timers.train}] Training done")
        if self.writer is not None:
            self.writer.close()

    # ===== epoch logging =====

    def start_epoch(self, train_set, lr):
        self.cur_epoch = self.cfg._cur_epoch
        self.accelerator.print(f" - ")
        self.accelerator.print(f"[T_total={self.timers.total} | T_train={self.timers.train}] Start epoch {self.cur_epoch} with {lr=:g}")
        self.metrics.reset_code_usage()

        self.epoch_n_samples = 0
        self.sum_loss = 0
        self.sum_all_losses = {}
        self.train_set_size = len(train_set)
        self.timers.epoch.start(reset=True)
        self.timers.train.start()

        if abs(lr - self.last_lr) > 1e-9:
            self.last_lr = lr
            self.last_lr_change = self.cur_step

    @torch.no_grad
    def step_epoch_batch(self, i_batch, batch, encoded_data, total_loss, losses, lr):
        self.metrics.register_codeword_usage("train", encoded_data)

        # Aggregate losses
        total_loss = self.accelerator.gather(total_loss).mean()
        losses = {k: self.accelerator.gather(l).mean() for k, l in sorted(losses.items())}

        # Convert losses to float
        if isinstance(total_loss, torch.Tensor):
            total_loss = float(total_loss.item())
        losses = {k: float(l.item()) for k, l in losses.items()}
        smooth_losses = self.smooth.update(**losses)
        smooth_total_loss = self.smooth.update(total_loss=total_loss)['total_loss']

        self.sum_loss += total_loss * len(batch)
        self.epoch_n_samples += len(batch)
        for k, v in losses.items():
            self.sum_all_losses[k] = self.sum_all_losses.get(k, 0) + v * len(batch)

        if self.writer is not None:
            self.writer.add_scalar('Loss/train', total_loss, self.cur_step)
            self.writer.add_scalar('hparam/lr', lr, self.cur_step)
            for k, v in losses.items():
                self.writer.add_scalar(f'Loss/{k}', v, self.cur_step)

        if self.cfg.verbose:
            losses_str = ' ; '.join([f"{k}={v}" for k, v in smooth_losses.items()])
            self.accelerator.print_nolog(
                "\033[K"
                f"[T_total={self.timers.total} | T_train={self.timers.train} | T_epoch={self.timers.epoch}] "
                f"train {i_batch+1} / {self.train_set_size} "
                f"(step {self.cur_step}) "
                f"{lr=:g} "
                f"loss={smooth_total_loss} (avg={self.sum_loss / self.epoch_n_samples:g}) "
                f"[[all losses: {losses_str}]]",
                end="\r",
                flush=True,
            )

        self.cur_step += 1

    def end_training_part_epoch(self):
        self.timers.train.stop()

    # ===== eval logging =====

    def start_eval(self, data_loader):
        self.n_total_batches = len(data_loader)
        self.timers.eval.start(reset=True)

        self.metrics.reset()

    def step_eval(self, i_batch, batch, xhat, encoded_data):
        if batch is not None:
            self.metrics.register_codeword_usage("val", encoded_data)
        
        self.accelerator.wait_for_everyone() # We will need to compute metrics for every processs anyway
        if batch is not None: # Update metric
            self.metrics.update(batch, xhat)

        # Compute & synchronise metrics
        self.metrics.compute()

        if self.cfg.verbose:
            self.accelerator.print_nolog(
                "\033[K"
                f"[T_total={self.timers.total} | T_train={self.timers.train} | T_inference={self.timers.eval}]",
                f"inference on validation split {i_batch + 1} / {self.n_total_batches}",
                self.metrics.metrics_as_str(),
                end="\r",
                flush=True,
            )

    def end_eval(self):
        self.accelerator.wait_for_everyone()
        if self.cfg.verbose:
            self.accelerator.print_nolog() # End line with '\r'
        self.metrics.compute(store_best=True) # Ensure we compute the metrics for the step, and store best ones

        if self.writer:
            self.writer.flush()
        self.accelerator.wait_for_everyone()

    def end_standalone_eval(self):
        self.accelerator.print("Validation metrics:", self.metrics.metrics_as_str())

        if self.writer is not None:
            m_vals = self.metrics.last_m_vals
            for name, v in m_vals.items():
                self.writer.add_scalar(self._metric_tag(name, 'val'), v, self.cur_step)
            for name, v in self.metrics.bests.items():
                self.writer.add_scalar(self._metric_tag(name, 'best'), v, self.cur_step)
            if self.cfg._rq_mse:
                self.writer.add_scalar('MSE/RQ', self.cfg._rq_mse, self.cur_step)
            self.writer.flush()
        
        # Show bits usage and entropy
        entropy = self.metrics.compute_codes_entropy()
        val_e, val_e_min = np.mean(entropy['val']), min(entropy['val'])
        self.accelerator.print(
            f'val_codeword_entropy={val_e:g} (min={val_e_min:g})'
        )


class TestMetricLogger:
    def __init__(self, cfg):
        self.cfg = cfg
        self.accelerator = cfg._accelerator
        self.eps = 1e-6

        self.timers = TimersManager('train', 'epoch', 'eval', 'total')
        self.metrics = MetricsManager(cfg, self.accelerator.device, ['test'])
        self.timers.total.start()

    def start_eval(self, data_loader):
        self.n_total_batches = len(data_loader)
        self.timers.eval.start(reset=True)

        self.metrics.reset()

    def step_eval(self, i_batch, batch, xhat, encoded_data):
        if batch is not None:
            self.metrics.register_codeword_usage("test", encoded_data)
        
        self.accelerator.wait_for_everyone() # We will need to compute metrics for every processs anyway
        if batch is not None: # Update metric
            self.metrics.update(batch, xhat)

        # Compute & synchronise metrics
        self.metrics.compute()

        if self.cfg.verbose:
            self.accelerator.print_nolog(
                "\033[K"
                f"[T_total={self.timers.total} | T_train={self.timers.train} | T_inference={self.timers.eval}]",
                f"inference on test split {i_batch + 1} / {self.n_total_batches}",
                self.metrics.metrics_as_str(),
                end="\r",
                flush=True,
            )

    def end_eval(self):
        self.accelerator.wait_for_everyone()
        if self.cfg.verbose:
            self.accelerator.print_nolog() # End line with '\r'
        self.metrics.compute(store_best=True) # Ensure we compute the metrics for the step, and store best ones

        self.accelerator.print("Test metrics:", self.metrics.metrics_as_str())

        # Show bits usage and entropy
        entropy = self.metrics.compute_codes_entropy()
        val_e, val_e_min = np.mean(entropy['test']), min(entropy['test'])
        self.accelerator.print(
            f'test_codeword_entropy={val_e:g} (min={val_e_min:g})'
        )