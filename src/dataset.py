from torch.utils.data import Dataset
import cv2
DIR_INPUT = 'input'
from PIL import Image
import torch 
import numpy as np

class PlantDataset(Dataset): 
    def __init__(self, df, transforms=None):
        self.df = df
        self.transforms=transforms
        
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        img_id, label, _ = self.df.loc[idx]
        image_src = DIR_INPUT + '/images/' + img_id + '.jpg'
        image = cv2.imread(image_src)
        labels = torch.tensor(label)
        # labels = labels.unsqueeze(-1)
        
        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed['image']

        return image, labels