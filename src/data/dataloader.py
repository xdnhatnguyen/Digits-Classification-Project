import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


# 1. Dinh nghia cac phep bien doi (Transforms)

def get_transform():
    """Dinh nghia cac phep bien doi anh (ToTensor va Chuan hoa)."""
    return transforms.Compose([
        # Chuyen anh sang Tensor PyTorch va chuan hoa ve [0, 1]
        transforms.ToTensor(), 
        
        # Chuan hoa (Normalization) cho tap MNIST: (x - mean) / std
        transforms.Normalize((0.1307,), (0.3081,))
    ])



# 2. Ham xay dung DataLoader chinh

def build_dataloaders(root_dir="./data", batch_size=64, train_ratio=0.8):
    """
    Tai du lieu MNIST, chia tap Train/Validation/Test, va tao DataLoader
    """
    transform = get_transform()
    
    # Tai tap du lieu Train (60,000 mau) va Test (10,000 mau)
    # Lan dau chay se tai du lieu ve thu muc root_dir (./data)
    full_train_dataset = datasets.MNIST(root=root_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=root_dir, train=False, download=True, transform=transform)

    # Chia tap du lieu Train thanh Train (80%) va Validation (20%)
    train_size = int(train_ratio * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    
    # Su dung random_split de chia ngau nhien dataset
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    
    print(f"Tong mau Train: {len(train_dataset)}, Validation: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Tao DataLoader
    # shuffle=True cho tap train de ngau nhien hoa du lieu moi epoch
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    # shuffle=False cho tap validation va test
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader


# Kiem tra nhanh 

if __name__ == '__main__':
    train_dl, val_dl, test_dl = build_dataloaders(batch_size=128)
    
    # Kiem tra kich thuoc cua mot batch
    images, labels = next(iter(train_dl))
    print(f"\nImage tensor shape: {images.shape}")
    print(f"Label tensor shape: {labels.shape}")
