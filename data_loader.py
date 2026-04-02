import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


def get_transforms():
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    return train_transforms, val_test_transforms


def get_dataloaders(data_dir, batch_size=32):
    train_transforms, val_test_transforms = get_transforms()
    dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transforms)
    total_size = len(dataset)
    train_size = int(0.7 * total_size)      # 70% for training
    val_size   = int(0.15 * total_size)     # 15% for validation
    test_size  = total_size - train_size - val_size
    
    # val_dataset   = datasets.ImageFolder(os.path.join(data_dir, 'valid'), transform=val_test_transforms)
    # test_dataset  = datasets.ImageFolder(os.path.join(data_dir, 'test'),  transform=val_test_transforms)

    # 3. Split the dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # for reproducibility
    )

    # 4. Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    print(f"Classes: {dataset.classes}")
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    return train_loader, val_loader, test_loader, dataset.classes

if __name__ == "__main__":
    train_loader, val_loader, test_loader, classes = get_dataloaders("data/raw")