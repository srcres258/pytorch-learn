import torch
from torch import nn


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv_unit = nn.Sequential(
            # x: [b, 3, 32, 32] => [b, 6, ]
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            #
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            #
        )
        # flatten
        # fc unit
        self.fc_unit = nn.Sequential(
            nn.Linear(2, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

        # [b, 3, 32, 32]
        tmp = torch.randn(2, 3, 32, 32)
        out = self.conv_unit(tmp)
        print('conv out:', out.shape)


def main():
    net = LeNet5()


if __name__ == '__main__':
    main()