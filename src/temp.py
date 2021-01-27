import dataset
from model import PlantModel
import engine
from torch.utils.data import DataLoader
import pandas as pd
from albumentations.pytorch import ToTensorV2
import albumentations
import torch
import torch.nn as nn

import pandas as pd 

if __name__=="__main__":
    device = 'cuda'
    DIR_INPUT = 'input'
    train_df = pd.read_csv(DIR_INPUT + '/train_fold.csv')
    transforms_train = albumentations.Compose([
    albumentations.RandomResizedCrop(height=256, width=256, p=1.0),
    ToTensorV2(p=1.0),
])
    dataset_train = dataset.PlantDataset(df=train_df, transforms=transforms_train)
    dataloader_train = DataLoader(dataset_train, batch_size=64)
    model = PlantModel(pretrained=True)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    plist = [{'params': model.parameters(), 'lr': 5e-5}]
    optimizer = torch.optim.Adam(plist, lr=5e-5)
    engine.train(dataloader_train, model, optimizer, device='cuda')