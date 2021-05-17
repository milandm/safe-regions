import torch
import torch.nn as nn
from modules.saferegion import SafeRegion

import unittest


class TestSafeRegion(unittest.TestCase):

    def test_sample(self):
        self.assertEqual(1, 1)

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
        x = torch.rand((2, 28, 28))
        lin = nn.Linear(28, 17)
        y = lin(x)

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


if __name__ == '__main__':
    unittest.main()
