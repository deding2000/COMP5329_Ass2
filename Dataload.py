from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset
from torchvision.io import read_image

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
  return np.array(y_encoded)

# Custom Dataset class for our data
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = annotations_file
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = pil_process_image_color(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label