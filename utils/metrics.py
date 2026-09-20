import torch


def calculate_metrics(inputs, targets, threshold=0.5, is_logits=True):
    """
    Calculate mean per-image Dice, IoU, Recall, and Precision.

    inputs:
        [H, W], [B, H, W], or [B, 1, H, W]
        Raw logits if is_logits=True, probabilities otherwise.

    targets:
        Binary masks, same shape layout as inputs.
    """
    inputs = inputs.float()
    targets = targets.float()

  
    if inputs.ndim == 2:
        inputs = inputs.unsqueeze(0).unsqueeze(0)
    elif inputs.ndim == 3:
        inputs = inputs.unsqueeze(1)

    if targets.ndim == 2:
        targets = targets.unsqueeze(0).unsqueeze(0)
    elif targets.ndim == 3:
        targets = targets.unsqueeze(1)

    if is_logits:
        inputs = torch.sigmoid(inputs)

    inputs = (inputs > threshold).float()
    targets = (targets > 0.5).float()

    batch_size = inputs.shape[0]

    inputs = inputs.reshape(batch_size, -1)
    targets = targets.reshape(batch_size, -1)

    tp = (inputs * targets).sum(dim=1)
    fp = (inputs * (1.0 - targets)).sum(dim=1)
    fn = ((1.0 - inputs) * targets).sum(dim=1)

    pred_sum = inputs.sum(dim=1)
    target_sum = targets.sum(dim=1)

 
    dice = (2.0 * tp + 1e-5) / (pred_sum + target_sum + 1e-5)
    iou = (tp + 1e-5) / (tp + fp + fn + 1e-5)

  
    one = torch.ones_like(pred_sum)
    zero = torch.zeros_like(pred_sum)

 
    precision = torch.where(
        pred_sum == 0,
        torch.where(target_sum == 0, one, zero),
        tp / (pred_sum + 1e-8)
    )

  
    recall = torch.where(
        target_sum == 0,
        torch.where(pred_sum == 0, one, zero),
        tp / (target_sum + 1e-8)
    )

    return {
        "dice": dice.mean().item(),
        "iou": iou.mean().item(),
        "recall": recall.mean().item(),
        "precision": precision.mean().item(),
    }