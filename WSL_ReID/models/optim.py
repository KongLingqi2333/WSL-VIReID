import torch
from bisect import bisect_right
class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, milestones, gamma=0.1, warmup_factor=1.0 / 3, warmup_iters=500,
                 warmup_method='linear', last_epoch=-1, mode = 'undifined'):
        if mode == 'undifined':
            raise('scheduler mode should be defined!')
        self.mode = mode
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of " " increasing integers. Got {}", milestones)

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup method accepted got {}".format(warmup_method))
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / float(self.warmup_iters)
                if self.mode == 'model':
                    warmup_factor = (self.warmup_factor * (1 - alpha) + alpha)/5
                elif self.mode == 'cls':
                    warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        elif self.warmup_iters <= self.last_epoch < self.milestones[0]:
            if self.mode == 'model':
                warmup_factor = 1/5
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]