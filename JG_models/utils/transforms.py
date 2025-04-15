from torchvision import transforms

def get_transforms(input_size, augment=False):
    if augment:
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # 👈 THIS LINE
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop(input_size, scale=(0.9, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # 👈 THIS LINE
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])