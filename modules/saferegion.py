import torch
import torch.nn as nn
from torch.nn import Module
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch import Tensor


# TODO:
# 1. precision err when cmp with nn.BatchNorm1d
# 2. do we need affine parameters?
# 3. lazy initializer
# 4. support DDP
# 5. tests

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

        # test time stats
        self.in_out = None
        self.distance = None

    def reset_stats(self) -> None:
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.num_batches_tracked.zero_()
        self.num_samples_tracked.zero_()
        self.running_min = None
        self.running_max = None

    def _check_input_dim(self, input):
        if input.dim() != 2:
            raise ValueError("expected 2D input (got {}D input)".format(input.dim()))

        if input.shape[1] != self.num_features:
            raise ValueError("expected input of size (N, {}) (got (N, {}) input)".format(self.num_features,
                                                                                         input.shape[1]))

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
            # at test time we are saving in / out matrix and how far away input is away from safe region
            # for the last input
            pass

        return input


def main():
    # model
    # model = nn.Sequential(
    #     nn.Identity(5),
    #     SafeRegion(5)
    #     # nn.BatchNorm1d(5, eps=0, momentum=None, affine=False, track_running_stats=True)
    # )
    # model.train()
    #
    # # input
    # x = torch.tensor([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]], dtype=torch.float)
    # y = model(x)
    # x = torch.tensor([[3, 4, 5, 6, 7], [0, 0, 0, 0, 0]], dtype=torch.float)
    # y = model(x)
    # print(y)
    sr = SafeRegion(5)
    bn = nn.BatchNorm1d(5, eps=0, momentum=None, affine=False, track_running_stats=True)
    torch.set_printoptions(precision=10)
    for i in range(10):
        r1 = -100
        r2 = 100
        x = (r1 - r2) * torch.rand((13, 5), dtype=torch.float) + r2
        sr(x)
        bn(x)
        print(sr.running_mean)
        print(bn.running_mean)
        # assert torch.eq(sr.running_mean, bn.running_mean)
        # assert torch.eq(sr.running_var, bn.running_var)


if __name__ == "__main__":
    main()
