import re
import pandas as pd
from io import StringIO
from PIL import Image
import numpy as np
import time
import math
import torch

# Set device to GPU if this is available
use_cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('We are using GPU.' if use_cuda else 'We are using CPU.')

# Loading filenames and labels
dataset_path = "Dataset/"
FILENAME_TRAIN = 'train.csv'
FILENAME_TEST = 'test.csv'
with open(dataset_path + FILENAME_TRAIN) as file:
    lines = [re.sub(r'([^,])"(\s*[^\n])', r'\1/"\2', line) for line in file]
    df_train = pd.read_csv(StringIO(''.join(lines)), escapechar="/")
with open(dataset_path + FILENAME_TEST) as file:
    lines = [re.sub(r'([^,])"(\s*[^\n])', r'\1/"\2', line) for line in file]
    df_test = pd.read_csv(StringIO(''.join(lines)), escapechar="/")

# To get overview
#print(df_train.describe)

# Loading images
def pil_process_image_color(image):
  image = Image.open(image).convert("RGB")
  arr = np.asarray(image)
  return arr

image_path = dataset_path + "data/0.jpg"
image = pil_process_image_color(image_path)
print(image.shape)






