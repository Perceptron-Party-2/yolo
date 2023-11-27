#Create train and test data loaders for "The Street View House Numbers (SVHN)" Dataset

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import SVHN


# Define the transformation
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Download SVHN dataset
# 32 x 32 RGB
svhn_train = SVHN(root="./data", split="train", download=True, transform=transform)
svhn_test = SVHN(root="./data", split="test", download=True, transform=transform)
svhn_train_dl = DataLoader(svhn_train, batch_size=512)
svhn_test_dl = DataLoader(svhn_test, batch_size=512)

