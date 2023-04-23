# CSCI 5525 | Group 11
# Written by Alex
import matplotlib.pyplot as plt
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data.crohme import CROHMEDataset as crohme
from data.hasy import hasy_tools as hasy

################################
#        Load HASYv2(s)        #
################################

# Be sure to unzip the .tar before running
# help(hasy) # prints some helpful tools, somewhat deprecated?
def load_hasy(csv_filepath=None):
    """
    Load HASYv2 data
        - Note for the file path there are 10 folders (fold-1 through fold-10)
          each which have train.csv and test.csv
    """
    if csv_filepath is None:
        raise ValueError("Filepath to .csv is not defined")
    
    print(f"Loading HASYv2 data from: '{csv_filepath}'")
    symbol_index = hasy.generate_index(csv_filepath)            # returns dict of labels
    images,labels = hasy.load_images(csv_filepath,symbol_index) # images has size (index,y,x,depth)
    print(f"Loaded {len(images)} images...")

    return images, labels

def plot_hasy_img(image):
    # plot a sample image has size (32,32), b&w img (1s and 0s)
    plt.imshow(hasy.thresholdize(image), cmap='gray')       # plot b&w image
    plt.show()

################################
#         Load CROHME          #
################################

# Load CROHME LaTeX data
## default batch sizes are 32
## see details within CROHMEDataset.py
def get_CROHME_loaders(batch_size=32):
    dataset = crohme.CROHMEDataset()

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

def plot_crohme_img(img, std=[0.485, 0.456, 0.406], mean=[0.229, 0.224, 0.225]):
    plt.imshow(unnormalize(img, std, mean))
    plt.show()

def unnormalize(img, std=[0.485, 0.456, 0.406], mean=[0.229, 0.224, 0.225]):
    img = img.numpy().transpose((1, 2, 0))
    img = (img * std) + mean  # Unnormalize
    img = img.clip(0, 1)      # Clip values to [0, 1] range
    return img

if __name__ == "__main__":
    # Test if this thing works
    cwd = os.getcwd()
    image_folder = cwd + "\\src\\data\\crohme\\images"
    label_file = cwd + "\\src\\data\\crohme\\CROHME_math.txt"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = crohme.CROHMEDatset(image_folder, label_file, transform=transform)

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
    sample_images, sample_labels, _ = next(data_iter)

    # Display the first image and its label
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    plot_crohme_img(sample_images[0], mean, std)
    print("Label:", sample_labels[0])