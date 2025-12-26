import torch
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.utils.data import random_split, DataLoader
import src.utils

def Transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.1307, 0.3081) #mean, std
    ])

def getLoaders():

    yaml_data = src.utils.get_data_from_config()

    # datasets.MNIST.mirrors = ["https://ossci-datasets.s3.amazonaws.com/mnist/"]
    full_train_data = datasets.MNIST(  # tham số khởi tạo root, train, transform, download.
        root='data',            # chỉ định folder nơi dataset sẽ được tải, lưu
        train=True,             # chỉ định train data
        transform=ToTensor(),   # đổi dữ liệu thành tensor
        download=True
    )

    test_data=datasets.MNIST(
        root='data',
        train=False,
        transform=ToTensor(),
        download=True
    )

    # Tổng số mẫu full_train_data là 60000, tiếp tục chia:
    total_size = len(full_train_data)
    train_size = int(0.8 * total_size)  # 48000 mẫu
    valid_size = total_size - train_size  # 12000 mẫu

    train_data, valid_data = random_split(full_train_data, [train_size, valid_size])

    image, label = train_data[0]
    
    loaders = {
        'train': DataLoader(train_data, batch_size=int(yaml_data["BATCH_SIZE"]), shuffle=True, num_workers=2),
        'valid': DataLoader(valid_data, batch_size=int(yaml_data["BATCH_SIZE"]), shuffle=True, num_workers=2),
        'test' : DataLoader(test_data, batch_size=int(yaml_data["BATCH_SIZE"]), shuffle=True, num_workers=2)
    }

    return loaders

if __name__ == "__main__":
    print("Testing data loaders...")
    loaders = getLoaders()
    for category, loader in loaders.items():
        print(f"{category} loader size: {len(loader.dataset)}")
    images, labels = next(iter(loaders['train']))
    print(f"Batch shape: {images.shape}")