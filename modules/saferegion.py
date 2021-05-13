import torch
import torch.nn as nn
from torch.nn import Module
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch import Tensor


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
        eps: float = 1e-5
    ) -> None:
        super(SafeRegion, self).__init__()
        self.num_features = num_features
        self.eps = eps

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
                # concat running_min with input and min of that
                temp = self.running_min

                pass
            self.num_batches_tracked.add_(1)
            self.num_samples_tracked.add_(batch_size)
            exponential_average_factor = 1.0 / float(self.num_samples_tracked)
            self.running_mean.add_(exponential_average_factor * torch.mean(input, dim=0))

            self.running_var.add_(exponential_average_factor * torch.var(input, dim=0))
        else:
            # at test time we are saving in / out matrix and how far away input is away from safe region
            pass

        return input


def main():
    # model
    model = nn.Sequential(
        nn.Identity(5),
        SafeRegion(5)
    )
    model.train()

    # input
    x = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.float)
    y = model(x)
    print(y)


if __name__ == "__main__":
    main()
