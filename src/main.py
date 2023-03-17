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
from data import hasy_tools as hasy
# help(hasy) # prints some helpful tools, somewhat deprecated?

# Load training data
## Note for the file path there are 10 folders (fold-1 through fold-10)
## each which have train.csv and test.csv
csv_filepath = "src\\data\\classification-task\\fold-1\\test.csv" # may need to be modified depending on your environment

print("Loading data from: " + csv_filepath)
symbol_index = hasy.generate_index(csv_filepath)                  # returns dict of labels
images,labels = hasy.load_images(csv_filepath,symbol_index) 
print(f"Loaded {len(images)} images...")


# plot a sample image
sample = hasy.thresholdize(images[0]) # returns np array of size (32,32), b&w img
plt.imshow(sample, cmap='gray')
plt.show()