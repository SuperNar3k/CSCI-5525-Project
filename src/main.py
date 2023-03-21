# CSCI 5525 - Group 11

################################
#           Modules            #
################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

################################
#        Load HASYv2(s)        #
################################

# Be sure to unzip the .tar before running

# Load training data
## Note for the file path there are 10 folders (fold-1 through fold-10)
## each which have train.csv and test.csv
csv_filepath = "src\\data\\classification-task\\fold-1\\test.csv" # may need to be modified depending on your environment

print(f"Loading data from: '{csv_filepath}'")
symbol_index = hasy.generate_index(csv_filepath)                  # returns dict of labels
images,labels = hasy.load_images(csv_filepath,symbol_index)       # images has size (index,y,x,depth)
print(f"Loaded {len(images)} images...")


# plot a sample image
sample = hasy.thresholdize(images[0]) # sample has size (32,32), b&w img (1s and 0s)
plt.imshow(sample, cmap='gray')       # plot b&w image
plt.show()