
from torchvision import transforms
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.distributions import Normal
from torch.utils.data import random_split



def dataloaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    cifar_dataset = datasets.CIFAR10(
        root='data',
        train=True,
        transform=transform,
        download=True
    )
    train_len = int(0.6 * len(cifar_dataset))
    val_len = int(0.2 * len(cifar_dataset))
    test_len = len(cifar_dataset) - train_len - val_len
    train_dataset, test_dataset, val_dataset= random_split(cifar_dataset, [train_len, test_len, val_len])

    train_dataloader = DataLoader(train_dataset, batch_size = 32)
    test_dataloader = DataLoader(test_dataset, batch_size = 32)
    val_dataloader = DataLoader(val_dataset, batch_size = 32 )
    return train_dataloader, val_dataloader, test_dataloader



