import dataset
from model import PlantModel
import engine
from torch.utils.data import DataLoader
import pandas as pd
from albumentations.pytorch import ToTensorV2
import albumentations
import torch
import torch.nn as nn



DIR_INPUT = 'input'
BATCH_SIZE = 64
epochs = 5


train_df = pd.read_csv(DIR_INPUT + '/train.csv')
train_df['sample_type'] = 'train'

sample_idx = train_df.sample(frac=0.2, random_state=42).index
train_df.loc[sample_idx, 'sample_type'] = 'valid'

valid_df = train_df[train_df['sample_type'] == 'valid']
valid_df.reset_index(drop=True, inplace=True)

train_df = train_df[train_df['sample_type'] == 'train']
train_df.reset_index(drop=True, inplace=True)

from albumentations.pytorch import ToTensorV2
transforms_train = albumentations.Compose([
    albumentations.RandomResizedCrop(height=256, width=256, p=1.0),
    albumentations.Flip(),
    albumentations.ShiftScaleRotate(rotate_limit=1.0, p=0.8),
    albumentations.Normalize(p=1.0),
    ToTensorV2(p=1.0),
])

transforms_valid = albumentations.Compose([
    albumentations.RandomResizedCrop(height=256, width=256, p=1.0),
    albumentations.Normalize(p=1.0),
    ToTensorV2(p=1.0),
])

dataset_train = dataset.PlantDataset(df=train_df, transforms=transforms_train)
dataset_valid = dataset.PlantDataset(df=valid_df, transforms=transforms_valid)

dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, num_workers=4, shuffle=True)
dataloader_valid = DataLoader(dataset_valid, batch_size=BATCH_SIZE, num_workers=4, shuffle=False)
device = torch.device("cpu")

model = PlantModel(num_classes=[1, 1, 1, 1])
model.to(device)

criterion = nn.BCEWithLogitsLoss()
plist = [{'params': model.parameters(), 'lr': 5e-5}]
optimizer = torch.optim.Adam(plist, lr=5e-5)

for epoch in range(epochs):
    engine.train(dataloader_train, model, optimizer, device)
    val_los = engine.evaluate(dataloader_valid, model, device)
    print(f'epoch = {epoch}, valid loss = {val_los}')