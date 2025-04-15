# scripts/preprocess_to_tensors.py

import os
import torch
from torchvision import datasets, transforms
from tqdm import tqdm

# Config
RAW_DATA_DIR = 'data/fer2013'
OUTPUT_DIR = 'data/fer2013_pt'
IMAGE_SIZE = 48

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def preprocess_split(split):
    dataset = datasets.ImageFolder(root=os.path.join(RAW_DATA_DIR, split), transform=transform)
    os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)

    for idx, (img, label) in enumerate(tqdm(dataset, desc=f'Preprocessing {split}')):
        save_path = os.path.join(OUTPUT_DIR, split, f'{idx:05d}.pt')
        torch.save({'image': img, 'label': label}, save_path)

    print(f"Saved {len(dataset)} {split} samples.")

if __name__ == '__main__':
    preprocess_split('train')
    preprocess_split('test')
