from utils.dataset_pt import FER2013TensorDataset
from torchvision import transforms
import os
import torch
from utils.transforms import get_transforms
from torchvision import datasets

def get_dataloaders_pt(root_dir, batch_size, num_workers, input_size=48):
    from utils.transforms import get_transforms

    train_dataset = FER2013TensorDataset(
        os.path.join(root_dir, 'train'),
        transform=get_transforms(input_size, augment=True)
    )
    test_dataset = FER2013TensorDataset(
        os.path.join(root_dir, 'test'),
        transform=get_transforms(input_size, augment=False)
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader

def get_dataloaders_original(data_dir, batch_size, num_workers, input_size=48):
    """
    Loads raw FER2013 images using torchvision.datasets.ImageFolder.
    Applies live augmentation to training data only.
    """
    train_transform = get_transforms(input_size, augment=True)
    test_transform = get_transforms(input_size, augment=False)

    train_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "train"),
        transform=train_transform
    )

    test_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "test"),
        transform=test_transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, test_loader