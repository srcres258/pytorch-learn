import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from lenet5 import LeNet5
from resnet import ResNet18


BATCH_SIZE = 32


def main():
    cifar_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ]))
    cifar_train = DataLoader(cifar_train, batch_size=BATCH_SIZE, shuffle=True)

    cifar_test = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ]))
    cifar_test = DataLoader(cifar_test, batch_size=BATCH_SIZE, shuffle=True)

    x, label = next(iter(cifar_train))
    print('x:', x.shape, 'label:', label.shape)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = LeNet5().to(device)
    model = ResNet18().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print(model)

    for epoch in range(1000):
        model.train()
        for batch_idx, (x, label) in enumerate(cifar_train):
            # [b, 3, 32, 32]
            # [b]
            x, label = x.to(device), label.to(device)

            logits = model(x)
            # logits: [b, 10]
            # label:  [b]
            # loss: tensor scalar
            loss = criterion(logits, label)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #
        print(epoch, loss.item())

        model.eval()
        with torch.no_grad():
            # test
            total_correct = 0
            total_num = 0
            for x, label in cifar_test:
                # [b, 3, 32, 32]
                # [b]
                x, label = x.to(device), label.to(device)

                # [b, 10]
                logits = model(x)
                # [b]
                pred = logits.argmax(dim=1)
                # [b] vs [b] =>
                total_correct += torch.eq(pred, label).float().sum().item()
                total_num += x.size(0)

            acc = total_correct / total_num
            print(epoch, acc)


if __name__ == '__main__':
    main()
