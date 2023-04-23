import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_utils import *
import os
import CROHMEDataset
import cv2

def find_bounding_boxes(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        bounding_boxes.append((x, y, w, h))

    return bounding_boxes

################################
#             Main             #
################################

if __name__ == "__main__":
    train_loader, test_loader, val_loader = CROHMEDataset.get_CROHME_loaders()

    # Get a sample from the train_loader
    data_iter = iter(train_loader)
    sample_images, sample_labels = next(data_iter)

    # Display the first image and its label
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    CROHMEDataset.print_sample(sample_images[0], mean, std)
    print("Label:", sample_labels[0])

    # Find bounding boxes for a given image
    image_path = r'src\data\crohme\images\0000000.png'
    bounding_boxes = find_bounding_boxes(image_path)

    # Visualize bounding boxes
    image = cv2.imread(image_path)
    for x, y, w, h in bounding_boxes:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()
