import numpy as np
import torch
from torch import nn, optim
from matplotlib import pyplot as plt


NUM_TIME_STEPS = 50
INPUT_SIZE = 1
HIDDEN_SIZE = 16
OUTPUT_SIZE = 1
LR = 0.01


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.rnn = nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=1,
            batch_first=True,
        )
        for p in self.rnn.parameters():
            nn.init.normal_(p, mean=0, std=0.001)

        self.linear = nn.Linear(in_features=HIDDEN_SIZE, out_features=OUTPUT_SIZE)

    def forward(self, x, hidden_prev):
        out, hidden_prev = self.rnn(x, hidden_prev)
        # [b, seq, h]
        out = out.view(-1, HIDDEN_SIZE)
        out = self.linear(out)
        out = out.unsqueeze(dim=0)
        return out, hidden_prev


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    hidden_prev = torch.zeros(1, 1, HIDDEN_SIZE).to(device)

    for iter in range(6000):
        start = np.random.randint(3, size=1)[0]
        time_steps = np.linspace(start, start + 10, NUM_TIME_STEPS)
        data = np.sin(time_steps)
        data = data.reshape(NUM_TIME_STEPS, 1)
        x = torch.tensor(data[:-1]).float().view(1, NUM_TIME_STEPS - 1, 1).to(device)
        y = torch.tensor(data[1:]).float().view(1, NUM_TIME_STEPS - 1, 1).to(device)

        output, hidden_prev = model(x, hidden_prev)
        hidden_prev = hidden_prev.detach()

        loss = criterion(output, y)
        model.zero_grad()
        loss.backward()
        optimizer.step()

        if iter % 100 == 0:
            print("iteration: {} loss {}".format(iter, loss.item()))

    start = np.random.randint(3, size=1)[0]
    time_steps = np.linspace(start, start + 10, NUM_TIME_STEPS)
    data = np.sin(time_steps)
    data = data.reshape(NUM_TIME_STEPS, 1)
    x = torch.tensor(data[:-1]).float().view(1, NUM_TIME_STEPS - 1, 1).to(device)
    y = torch.tensor(data[1:]).float().view(1, NUM_TIME_STEPS - 1, 1).to(device)

    predictions = []
    input = x[:, 0, :]
    for _ in range(x.shape[1]):
        input = input.view(1, 1, 1)
        (pred, hidden_prev) = model(input, hidden_prev)
        input = pred
        predictions.append(pred.detach().cpu().numpy().ravel()[0])

    x = x.data.cpu().numpy().ravel()
    y = y.data.cpu().numpy()
    plt.scatter(time_steps[:-1], x.ravel(), s=90)
    plt.plot(time_steps[:-1], x.ravel())

    plt.scatter(time_steps[1:], predictions)
    plt.show()


if __name__ == '__main__':
    main()
