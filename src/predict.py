import engine
import pandas as pd 
from dataset import PlantDataset
from albumentations.pytorch import ToTensorV2
import albumentations
from torch.utils.data import DataLoader
from model import PlantModel
import torch 
import numpy as np 
from tqdm import tqdm


if __name__=="__main__":
    device = 'cuda'
    submission_df = pd.read_csv('input/sample_submission.csv')

    transforms_valid = albumentations.Compose([
        albumentations.RandomResizedCrop(height=256, width=256, p=1.0),
        albumentations.Normalize(p=1.0),
        ToTensorV2(p=1.0),
    ])
    dataset_test = PlantDataset(df=submission_df, transforms=transforms_valid)
    dataloader_test = DataLoader(dataset_test, batch_size=64, num_workers=4, shuffle=False)

    # loading saved model
    model_inf = PlantModel(num_classes=[1, 1, 1, 1])
    model_inf.load_state_dict(torch.load('models/model.pt'))
    model_inf = model_inf.to(device)
    
    prediction = engine.predict(dataloader_test, model_inf, device='cuda')
    submission_df[['healthy', 'multiple_diseases', 'rust', 'scab']] = prediction.numpy()
    submission_df.to_csv('submission.csv', index=False)