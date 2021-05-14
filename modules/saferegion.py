import torch
import torch.nn as nn
from torch.nn import Module
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch import Tensor


# TODO:
# engineering todo
# 1. quickly add DDP
# 2. swap in bigger models

# module todo
# 1. small precision err when cmp with nn.BatchNorm1d
# 2. can we use maybe affine parameters?
# 3. lazy initializer
# 4. support DDP
# 5. tests
# 6. module 1d, 2d, 3d
# plot number of in / outs per sample

class SafeRegion(Module):
    """
    Module for recording safe regions of neural units
    """
    _version = 1
    num_features: int
    eps: float
    momentum: float

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = None
    ) -> None:
        super(SafeRegion, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # training stats
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        self.register_buffer('num_samples_tracked', torch.tensor(0, dtype=torch.long))
        self.register_buffer('running_min', None)
        self.register_buffer('running_max', None)

        # test stats
        self.in_out = None
        self.distance = None

    def reset_stats(self) -> None:
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.num_batches_tracked.zero_()
        self.num_samples_tracked.zero_()
        self.running_min = None
        self.running_max = None

        # test time stats
        self.in_out = None
        self.distance = None

    def _check_input_dim(self, input):
        if input.dim() != 2:
            raise ValueError(f'expected 2D input (got {input.dim()}D input)')

        if input.shape[1] != self.num_features:
            raise ValueError(f'expected input of size (N, {self.num_features}) (got (N, {input.shape[1]}) input)')

    def forward(self, input: Tensor) -> Tensor:
        self._check_input_dim(input)
        batch_size = input.shape[0]

        if self.training:   # at training time we are record parameters
            if self.num_batches_tracked == 0:
                self.running_min = torch.min(input, dim=0)[0]
                self.running_max = torch.max(input, dim=0)[0]
            else:
                self.running_min = torch.min(self.running_min, torch.min(input, dim=0)[0])
                self.running_max = torch.max(self.running_max, torch.max(input, dim=0)[0])

            self.num_batches_tracked.add_(1)
            self.num_samples_tracked.add_(batch_size)

            if self.momentum:
                exponential_average_factor = self.momentum
            else:
                exponential_average_factor = 1.0 / float(self.num_batches_tracked)

            self.running_mean = (1 - exponential_average_factor) * self.running_mean + exponential_average_factor * torch.mean(input, dim=0)
            self.running_var = (1 - exponential_average_factor) * self.running_var + exponential_average_factor * torch.var(input, dim=0)
        else:
            # at test time we are saving in / out matrix and how far away last input is away from safe region
            pass

        return input
