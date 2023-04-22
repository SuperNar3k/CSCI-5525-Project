import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from data.crohme.LatexDataset import LatexDataset, generate_tokenizer
from data.hasy import hasy_tools as hasy

################################
#        Load HASYv2(s)        #
################################

# Be sure to unzip the .tar before running
# help(hasy) # prints some helpful tools, somewhat deprecated?
def load_hasy():
    """
    Load HASYv2 data
        - Note for the file path there are 10 folders (fold-1 through fold-10)
          each which have train.csv and test.csv
    """
    csv_filepath = "src\\data\\hasy\\classification-task\\fold-1\\test.csv" # may need to be modified depending on your environment
    
    print(f"Loading HASYv2 data from: '{csv_filepath}'")
    symbol_index = hasy.generate_index(csv_filepath)            # returns dict of labels
    images,labels = hasy.load_images(csv_filepath,symbol_index) # images has size (index,y,x,depth)
    print(f"Loaded {len(images)} images...")


    # plot a sample image
    sample = hasy.thresholdize(images[0]) # sample has size (32,32), b&w img (1s and 0s)
    plt.imshow(sample, cmap='gray')       # plot b&w image
    plt.show()

    return images, labels

################################
#         Load CROHME          #
################################

# Load CROHME LaTeX data
## default batch sizes are 16
## see details within LatexDatset.py
def load_crohme():
    """
    Load CROHME LaTeX data
        - default batch sizes are 16
        - see details within LatexDatset.py
    """
    eqn_path = "src\\data\\crohme\\math.txt" # may need to be modified depending on your environment
    img_path = "src\\data\\crohme\\images"
    tokenizer_path = "src\\data\\crohme\\tokenizer.json"
    vocab_size = 8000

    print(f"Loading CROHME data from: {eqn_path}")
    generate_tokenizer(equations=[eqn_path], output=tokenizer_path, vocab_size=vocab_size)
    latex_dataloader = LatexDataset(equations=eqn_path, images=img_path, tokenizer=tokenizer_path) # torch style dataloader
    
    return latex_dataloader
