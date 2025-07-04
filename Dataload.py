from PIL import Image
import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import transforms

# Loading images
def pil_process_image_color(image):
  image = Image.open(image).convert("RGB")
  #arr = np.asarray(image)
  return image

# Encode target labels
def encode(y):
  y = list(map(int, y.split(" ")))
  y_encoded = np.zeros(19, dtype=int)
  for i in range(len(y)):
    y_encoded[int(y[i])-1] = 1
  return np.delete(y_encoded,11)

def get_class(array):
  class_pred = np.where(array > 0.5)[1] +1
  if class_pred >= 12:
     class_pred += 1
  return class_pred

def unencode(y_pred,to_cpu=True):
  labels = []
  for i in range(len(y_pred)):
    if to_cpu: # Move to cpu device
      temp = y_pred[i].cpu()
    temp = np.asarray(temp)
    temp = get_class(temp)
    temp = " ".join(str(item) for item in temp)
    labels.append(temp)
  return labels 

# Custom Dataset class for our data
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir,transform=None, target_transform=None,transform_aug = None,targets_available=True, use_captions=False):
        self.img_labels = annotations_file
        self.img_dir = img_dir
        self.transform = transform
        self.transform_aug = transform_aug
        self.target_transform = target_transform
        self.targets_available = targets_available
        self.use_captions = use_captions
      
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        if self.use_captions:
           # Still in testing phase
           caption = self.img_labels.iloc[idx,2]
           print("Caption: {}".format(caption))
           #caption_embedded = text_to_embed(caption)
           print("Caption Embedding Shape:", caption.shape)
            
        image = pil_process_image_color(img_path)
        if self.targets_available:
          label = self.img_labels.iloc[idx, 1]
        else:
          label = []

        if self.target_transform and self.targets_available:
            label = self.target_transform(label)
        
        if self.transform_aug and label[0]!=1:
            image = self.transform_aug(image)
        else:
            image = self.transform(image)
            
        # if self.use_caption:
        #   return image, caption, label
        return image, label