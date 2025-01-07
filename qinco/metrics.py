import torch
import torcheval.metrics as t_metrics
import scipy
import time


####################################################################
# Utilities
####################################################################

def compute_parallel(cfg, metric):
    if hasattr(metric, 'compute_sync'):
        val = metric.compute_sync(cfg._accelerator)
    else:
        val = metric.compute()
    return val


####################################################################
# Metrics
####################################################################

class AnyVectMSE(t_metrics.Metric):
    def __init__(self, device=None, scale=1.0, reduction='sum'):
        super().__init__(device=device)

        self.scale = scale
        self.loss_sum = None
        self.num_samples = None
        self.reduction = reduction
        self._add_state("loss_sum", torch.tensor(0., device=self.device))
        self._add_state("num_samples", torch.tensor(0, device=self.device))

    @torch.inference_mode()
    def update(self, batch, xhat):
        if self.reduction == 'sum':
            self.loss_sum += ((batch - xhat)**2).sum().cpu()
        elif self.reduction == 'mean':
            self.loss_sum += ((batch - xhat)**2).mean().cpu() * len(batch)
        else:
            raise ValueError(f'{self.reduction}')
        self.num_samples += len(batch)

    @torch.inference_mode()
    def compute(self):
        return (self.loss_sum * self.scale / self.num_samples)

    @torch.inference_mode()
    def compute_sync(self, accelerator):
        l_sum = accelerator.gather(self.loss_sum * self.scale).sum()
        N = accelerator.gather(self.num_samples).sum()
        return l_sum / N

    @torch.inference_mode()
    def merge_state(self, metrics):
        for m in metrics:
            self.loss_sum += m.loss_sum
            self.num_samples += m.num_samples
        return self


class CodebookEntropy(t_metrics.Metric):
    def __init__(self, rq_steps, codebook_sizes, device=None):
        super().__init__(device=device)

        self.usage = None
        self.enable = False
        self.codebook_sizes = codebook_sizes
        self.cumul_sizes = [int(sum(codebook_sizes[:i])) for i in range(len(codebook_sizes))]
        self.cumul_sizes.append(int(sum(codebook_sizes)))
        if sum(self.codebook_sizes) <= 8388608:
            self._add_state("usage", torch.zeros(sum(self.codebook_sizes), dtype=int, device=device))
            self.enable = True
        else:
            self._add_state("usage", torch.zeros((1,), dtype=int, device=device))

    @torch.inference_mode()
    def update(self, codes):
        assert len(codes) == len(self.codebook_sizes)
        if not self.enable:
            return
        for ibook, codewords in enumerate(codes):
            code_ids, counts = codewords.detach().to(self.device).unique(return_counts=True)
            code_ids, counts = code_ids.to(self.usage.device), counts.to(self.usage.device)
            self.usage[code_ids+self.cumul_sizes[ibook]] += counts

    @torch.inference_mode()
    def compute(self):
        if not self.enable:
            usage_2d = [0 for _ in self.codebook_sizes]
            return usage_2d, usage_2d
        usage_2d = [self.usage[self.cumul_sizes[i]:self.cumul_sizes[i+1]] for i in range(len(self.codebook_sizes))]
        return [scipy.stats.entropy(u.cpu(), base=2) if float(u.sum()) > 1e-6 else 0.0 for u in usage_2d], usage_2d

    @torch.inference_mode()
    def merge_state(self, metrics):
        if not self.enable:
            return
        for m in metrics:
            self.usage += m.usage.to(self.device)
        return self

    def gather(self, cfg):
        return compute_parallel(cfg, self)


####################################################################
# Managing sets of metrics
####################################################################


class MetricsManager:
    def __init__(self, cfg, device, splits):
        self.cfg = cfg
        self.metrics = {}

        self.metrics['MSE'] = AnyVectMSE(device=device, scale=cfg.mse_scale)

        self.bests = {}
        self.last_m_vals = {}
        self.code_usage = { s: CodebookEntropy(self.cfg.M, self.cfg._K_vals, device=device) for s in splits }

    def reset(self):
        for m in self.metrics.values():
            m.reset()

    def reset_code_usage(self):
        for m in self.code_usage.values():
            m.reset()

    def update(self, batch, xhat):
        for name, m in self.metrics.items():
            m.update(batch, xhat)

    def to(self, device):
        for m in self.metrics.values():
            m.to(device)

    def compute(self, store_best=False):
        m_vals = {}
        for m_name, m in self.metrics.items():
            name = m_name
            v = compute_parallel(self.cfg, m)
            v = float(v.item())
            m_vals[name] = v
            self.last_m_vals[name] = v
            if store_best:
                if name not in self.bests:
                    self.bests[name] = v
                else:
                    self.bests[name] = min(self.bests[name], v)
        return m_vals

    def metrics_as_str(self, m_vals=None):
        if not m_vals:
            m_vals = self.last_m_vals
        return '[[' + ' | '.join(f'{name}={v:g}' for name, v in m_vals.items()) + ']]'

    def bests_as_str(self):
        return '[[' + ' | '.join(f'min_{name}={v:g}' for name, v in self.bests.items()) + ']]'

    def register_codeword_usage(self, split, codes):
        self.code_usage[split].update(codes)

    def compute_codes_entropy(self):
        return { split: compute_parallel(self.cfg, usage)[0] for split, usage in self.code_usage.items() }

    def compute_codes_usage(self, split='train'):
        return compute_parallel(self.cfg, self.code_usage[split])[1]


####################################################################
# Timing metrics
####################################################################

class Timer:
    def __init__(self, ms=False, elapsed=0):
        self.elapsed = elapsed
        self.start_at = None
        self.show_ms = ms # Show milli-seconds

    def start(self, reset=False):
        """Start the timer. It should always be called when the timer is not currently running, except if used with reset=True

        Args:
            reset (bool, optional): Reset the total elapsed time to 0.
                If the timer was running, is will start again from the curent time.
                Defaults to False.
        """
        if reset:
            self.reset()
        else:
            assert self.start_at is None, "Timer is already in use"
        self.start_at = time.time()

    def reset(self):
        self.elapsed = 0
        if self.start_at:
            self.start_at = time.time()

    def stop(self):
        self.elapsed += time.time() - self.start_at
        self.start_at = None
        return self.get()

    def running(self):
        return self.start_at is not None

    def get(self):
        if self.running():
            return self.elapsed + time.time() - self.start_at
        return self.elapsed

    @staticmethod
    def format_time(t, ms=False):
        hours, t = divmod(t, 3600)
        minutes, seconds = divmod(t, 60)
        millis_str = ''
        if ms:
            millis = (seconds - int(seconds))*1000
            millis_str = f'.{int(millis):03}'
        return f'{int(hours):02}:{int(minutes):02}:{int(seconds):02}{millis_str}'

    def s(self, digits=3):
        fmt = '{0:.' + str(digits) + 'f}s'
        return fmt.format(self.get())

    def ms(self):
        return self.format_time(self.get(), True)

    def __str__(self) -> str:
        return self.format_time(self.get(), self.show_ms)

    def __repr__(self) -> str:
        return self.__str__()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def __call__(self, reset=False):
        if reset:
            self.reset()
        return self

class TimersManager:
    def __init__(self, *timer_list):
        self.add(*timer_list)

    def add(self, *names):
        for name in names:
            if not hasattr(self, name):
                self.__setattr__(name, Timer())

    def get_timer_list(self):
        return [attr for attr in self.__dir__() if isinstance(self[attr], Timer)]

    def __getitem__(self, attr):
        return self.__getattribute__(attr)

    def __str__(self) -> str:
        return ' '.join([f'T_{name}={self[name]}' for name in self.get_timer_list()])

    def __repr__(self) -> str:
        return self.__str__()

    def state_dict(self):
        return {
            name: self[name].get() for name in self.get_timer_list()
        }

    def sum(self):
        total_s = sum([self[name].get() for name in self.get_timer_list()])
        return Timer(elapsed=total_s)

    def load_state_dict(self, state):
        for name, val in state.items():
            self[name].elapsed = val