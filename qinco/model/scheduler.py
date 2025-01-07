import math
import torch
from torch.optim.lr_scheduler import LambdaLR


class RampCosineLRSchedule(LambdaLR):
    def __init__(
        self,
        optimizer,
        num_ramp_epochs: float,
        num_max_epochs: float,
        min_val: float = 1e-12,
    ):
        self._num_ramp_epochs = num_ramp_epochs
        self._max_epochs = num_max_epochs
        self._min_val = min_val

        def step_fn(step):
            if step < num_ramp_epochs:
                return max(min(step / num_ramp_epochs, 1.0), min_val)
            else:
                p = (step - num_ramp_epochs) / num_max_epochs
                angle = min(p, 1.0) * math.pi / 2
                return max(math.cos(angle), min_val)

        super().__init__(optimizer=optimizer, lr_lambda=step_fn)

    def train():
        pass


class NoLRScheduler(LambdaLR):
    def __init__(self, optimizer):
        super().__init__(optimizer=optimizer, lr_lambda=lambda step : 1)