
from tqdm import tqdm
import model
import torch 
import torch.nn as nn 
import numpy as np 
criterion = nn.BCEWithLogitsLoss()

def train(dataloader, model, optimizer, device):
    model.train()
    tr_loss = 0
    tk0 = tqdm(dataloader, desc="Train")
    for step, batch in enumerate(tk0):
        inputs = batch[0]
        targets = batch[1]
        
        inputs = inputs.to(device, dtype=torch.float)
        targets = targets.to(device, dtype=torch.float)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        losses = []
        for i in range(4):
            losses.append(criterion(outputs[i], targets[:,i]))
        loss = losses[0]+losses[1]+losses[2]+losses[3]
        loss.backward()
        tr_loss += loss.item()
        optimizer.step()

def evaluate(data_loader, model, device):
    model.eval()
    val_loss = 0
    val_preds = None
    val_labels = None
    tk0 = tqdm(data_loader, desc='Validate')
    
    for step, batch in enumerate(tk0):
            
        inputs = batch[0]
        targets = batch[1]
        
        if val_labels is None:
            val_labels = targets.clone().squeeze(-1)
        else:
            val_labels = torch.cat((val_labels, targets.squeeze(-1)))
        
        inputs = inputs.to(device, dtype=torch.float)
        targets = targets.to(device, dtype=torch.float)
        with torch.no_grad():
            output = model(inputs)
            losses = []
            for i in range(4):
                losses.append(criterion(output[i], targets[:,i]))
            loss = losses[0]+losses[1]+losses[2]+losses[3]
            val_loss += loss.item()
            
            preds = torch.sigmoid(torch.stack(output).permute(1, 0, 2).cpu().squeeze(-1))
            
            if val_preds is None:
                val_preds = preds
            else:
                val_preds = torch.cat((val_preds, preds), dim=0)
        return val_loss

def predict(dataloader, model, device):
    model.eval()
    tk0 = tqdm(dataloader, desc="Predict")
    test_preds = None
    for step, batch in enumerate(tk0):
        images = batch[0]
        images = images.to(device, dtype=torch.float)
        with torch.no_grad():
            outputs = model(images)
            preds = torch.sigmoid(torch.stack(outputs).permute(1, 0, 2).cpu().squeeze(-1))
            if test_preds is None:
                test_preds = preds
            else:
                test_preds = torch.cat((test_preds, preds), dim=0)
    return test_preds