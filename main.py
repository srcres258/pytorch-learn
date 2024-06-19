from torchvision import datasets, transforms
from torch.utils.data import DataLoader


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


if __name__ == '__main__':
    main()
