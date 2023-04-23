import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import cv2
import os

import data_utils

def find_bounding_boxes(image_path=None, image=None):
    """
    Takes either an image path OR an image. Taking an image directly is
    kind of broken rn

    Args:
        - image_path (str): a string path to the single image file
        - image: has to be from the custom CROHME dataloader
    """
    if image_path and not isinstance(image, torch.Tensor):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    elif isinstance(image, torch.Tensor) and (image_path is None):
        image = tensor_to_grayscale(img)
    else:
        raise ValueError("find_bounding_boxes should only be given a path OR an image file")
    
    _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        bounding_boxes.append((x, y, w, h))

    return bounding_boxes

def tensor_to_grayscale(img):
    img_np = img.numpy().transpose(1, 2, 0)  # Convert tensor to NumPy array
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    img_gray = (img_gray * 255).astype('uint8')  # Convert to 8-bit image
    return img_gray

################################
#             Main             #
################################

if __name__ == "__main__":
    train_loader, test_loader, val_loader = data_utils.get_CROHME_loaders()

    # Get a sample from the train_loader
    sample_images, sample_labels, sample_fnames = next(iter(train_loader))
    img = sample_images[0]

    # Display the first image and its label
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    data_utils.plot_crohme_img(img, mean, std)
    print("Label:", sample_labels[0])

    # Find bounding boxes for a given image
    image_path = os.getcwd() + "\\src\\data\crohme\\images\\" + sample_fnames[0]
    print(image_path)
    bounding_boxes = find_bounding_boxes(image_path)
    
    # Visualize bounding boxes
    image = cv2.imread(image_path)
    for x, y, w, h in bounding_boxes:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Slice the image using the bounding box coordinates
        sub_image = image[y:y+h, x:x+w]

        # Convert the sub-image to RGB format
        sub_image_rgb = cv2.cvtColor(sub_image, cv2.COLOR_BGR2RGB)

        # Display the sub-image
        plt.imshow(sub_image_rgb)
        plt.show()


    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()