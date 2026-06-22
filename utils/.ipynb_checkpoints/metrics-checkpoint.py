import torch

def calculate_metrics(inputs, targets, threshold=0.5, smooth=1e-5):

    inputs = torch.sigmoid(inputs)
    inputs = (inputs > threshold).float()
    
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    
    TP = (inputs * targets).sum()                
    FP = ((inputs == 1) & (targets == 0)).sum()  
    FN = ((inputs == 0) & (targets == 1)).sum()   
    
    dice = (2. * TP + smooth) / (2. * TP + FP + FN + smooth)
    iou = (TP + smooth) / (TP + FP + FN + smooth)
    recall = (TP + smooth) / (TP + FN + smooth)      
    precision = (TP + smooth) / (TP + FP + smooth)   
    
    return {
        "dice": dice.item(),
        "iou": iou.item(),
        "recall": recall.item(),
        "precision": precision.item()
    }
