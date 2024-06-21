import torch
from torch import nn, optim, autograd
import numpy as np
import visdom
import random
from matplotlib import pyplot as plt


h_dim = 400
batch_size = 512
viz = visdom.Visdom()


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.net = nn.Sequential(
            # z: [b, 2] => [b, 2]
            nn.Linear(2, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, 2)
        )

    def forward(self, z):
        output = self.net(z)
        return output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.net(x)
        return output.view(-1)


def data_generator():
    """
    8-gaussian mixture models
    :return:
    """
    scale = 2.
    centers = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1. / np.sqrt(2), 1. / np.sqrt(2)),
        (1. / np.sqrt(2), -1. / np.sqrt(2)),
        (-1 / np.sqrt(2), 1. / np.sqrt(2)),
        (-1 / np.sqrt(2), -1. / np.sqrt(2))
    ]
    centers = [(scale * x, scale * y) for x, y in centers]

    while True:
        dataset = []

        for i in range(batch_size):
            point = np.random.randn(2) * 0.02
            center = random.choice(centers)
            # N(0, 1) + center x1/x2
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)

        dataset = np.array(dataset).astype(np.float32)
        dataset /= np.sqrt(2)
        yield dataset


def generate_image(D, G, xr, epoch):
    # TODO
    pass


def main():
    torch.manual_seed(114514)
    np.random.seed(114514)

    data_iter = data_generator()
    x = next(data_iter)
    # [b, 2]
    # print(x.shape)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G = Generator().to(device)
    D = Discriminator().to(device)
    # print(G)
    # print(D)
    optim_G = optim.Adam(G.parameters(), lr=5e-4, betas=(0.5, 0.9))
    optim_D = optim.Adam(D.parameters(), lr=5e-4, betas=(0.5, 0.9))

    viz.line([[0, 0]], [0], win='loss', opts=dict(title='loss', legend=['D', 'G']))

    for epoch in range(50000):
        # 1. train Discriminator at first
        for _ in range(5):
            # 1.1. train on real data
            x = next(data_iter)
            x = torch.from_numpy(x).to(device)
            # [b, 2] => [b, 1]
            predr = D(x)
            # max predr
            lossr = -predr.mean()

            # 1.2. train on fake data
            # [b, 2]
            z = torch.randn(batch_size, 2).to(device)
            xf = G(z).detach()
            predf = D(xf)
            lossf = predf.mean()

            # aggregate all
            loss_D = lossr + lossf

            # optimize
            optim_D.zero_grad()
            loss_D.backward()
            optim_D.step()

        # 2. train Generator
        z = torch.randn(batch_size, 2).to(device)
        xf = G(z)
        predf = D(xf)
        # max predf.mean()
        loss_G = -predf.mean()

        # optimize
        optim_G.zero_grad()
        loss_G.backward()
        optim_G.step()

        if epoch % 100 == 0:
            viz.line([[loss_D.item(), loss_G.item()]], [epoch], win='loss', update='append')

            print(loss_D.item(), loss_G.item())


if __name__ == '__main__':
    main()
