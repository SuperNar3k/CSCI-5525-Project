# CSCI 5525 | Group 11
import matplotlib.pyplot as plt
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class CROHMEDatset(Dataset):
    def __init__(self, image_folder="src\data\crohme\images", label_file="src\data\crohme\CROHME_math.txt", transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])):
        self.image_folder = image_folder
        self.label_file = label_file
        self.transform = transform
        
        with open(label_file, 'r') as f:
            self.labels = [line.strip() for line in f.readlines() if line != ""]

        self.image_files = sorted(os.listdir(image_folder))

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.image_files[idx])
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        label = self.labels[idx]
        return img, label
    
def get_CROHME_loaders(batch_size=32):
    dataset = CROHMEDatset()

    # Split the dataset into training, testing, and validation sets
    train_ratio, test_ratio, val_ratio = 0.7, 0.2, 0.1
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    test_size = int(test_ratio * dataset_size)
    val_size = dataset_size - train_size - test_size

    train_set, test_set, val_set = torch.utils.data.random_split(dataset, [train_size, test_size, val_size])

    # Create DataLoaders for each set
    batch_size = 32
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader, val_loader

def print_sample(img, std, mean):
    img = img.numpy().transpose((1, 2, 0))
    img = (img * std) + mean  # Unnormalize
    img = img.clip(0, 1)      # Clip values to [0, 1] range
    plt.imshow(img)
    plt.show()

if __name__ == "__main__":
    # Test if this thing works
    cwd = os.getcwd()
    image_folder = "src\data\crohme\images"
    label_file = "src\data\crohme\CROHME_math.txt"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = CROHMEDatset(image_folder, label_file, transform=transform)

    # Split the dataset into training, testing, and validation sets
    train_ratio, test_ratio, val_ratio = 0.7, 0.2, 0.1
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    test_size = int(test_ratio * dataset_size)
    val_size = dataset_size - train_size - test_size

    train_set, test_set, val_set = torch.utils.data.random_split(dataset, [train_size, test_size, val_size])

    # Create DataLoaders for each set
    batch_size = 32
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)

    # Get a sample from the train_loader
    data_iter = iter(train_loader)
    sample_images, sample_labels = next(data_iter)

    # Display the first image and its label
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    print_sample(sample_images[0], mean, std)
    print("Label:", sample_labels[0])