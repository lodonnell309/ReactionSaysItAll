import torch
import os, glob
from torch.utils.data import Dataset
from torchvision import transforms

class FER2013TensorDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.file_paths = sorted(glob.glob(os.path.join(root_dir, '*.pt')))
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        sample = torch.load(self.file_paths[idx], weights_only=True)
        image, label = sample['image'], sample['label']

        if self.transform:
            # Convert to PIL and re-augment
            image = transforms.ToPILImage()(image)
            image = self.transform(image)

        return image, label
