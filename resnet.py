import torch
from torch import nn
from torch.nn import functional as F


class ResBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        self.extra = nn.Sequential()
        if ch_out != ch_in:
            # [b, ch_in, h, w] => [b, ch_out, h, w]
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # short cut
        # extra module: [b, ch_in, h, w] with [b, ch_out, h, w]
        # element-wise add:
        out = self.extra(x) + out

        return out


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )
        # followed by 4 blocks
        # [b, 64, h, w] => [b, 128, h, w]
        self.block1 = ResBlock(64, 64)
        # [b, 128, h, w] => [b, 256, h, w]
        self.block2 = ResBlock(64, 128)
        # [b, 256, h, w] => [b, 512, h, w]
        self.block3 = ResBlock(128, 256)
        # [b, 512, h, w] => [b, 1024, h, w]
        self.block4 = ResBlock(256, 512)

        self.out_layer = nn.Linear(512 * 32 * 32, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))

        # [b, 64, h, w] => [b, 1024, h, w]
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = x.view(x.size(0), -1)
        x = self.out_layer(x)

        return x


def main():
    block = ResBlock(64, 128)
    tmp = torch.randn(2, 64, 32, 32)
    out = block(tmp)
    print('block', out.shape)

    model = ResNet18()
    tmp = torch.randn(2, 3, 32, 32)
    out = model(tmp)
    print('resnet18', out.shape)


if __name__ == '__main__':
    main()
