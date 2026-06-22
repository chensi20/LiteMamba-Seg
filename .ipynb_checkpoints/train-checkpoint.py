import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2
import os
import numpy as np

try:
    from torch.amp import GradScaler
except ImportError:
    from torch.cuda.amp import GradScaler
from tqdm import tqdm

from config import Config
from dataset import get_loaders
from utils.losses import StructureLoss
from utils.metrics import calculate_metrics
from utils.tools import seed_everything
from models.MambaSeg import MambaSeg_UNet as UNet


class BoundaryIoULoss(nn.Module):
    def __init__(self):
        super(BoundaryIoULoss, self).__init__()

    def forward(self, pred, target):
        """
        IoU Loss
        pred: Logits
        target: Binary Mask
        """
        pred = torch.sigmoid(pred)

        if target.ndim == 3:
            target = target.unsqueeze(1)
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)

        target = target.float()

        pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        eroded = 1 - pool(1 - target)
        dilated = pool(target)
        gt_boundary = dilated - eroded

        intersection = (pred * target * gt_boundary).sum()
        union = (pred * gt_boundary).sum() + (target * gt_boundary).sum()

        boundary_iou = (intersection + 1e-6) / (union - intersection + 1e-6)

        return 1 - boundary_iou


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def train_one_epoch(model, loader, criterion_main, criterion_bound, optimizer, scaler, device, epoch):
    model.train()
    running_loss = 0.0

    loop = tqdm(loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS} [Train]", leave=True)

    for images, masks in loop:
        images = images.to(device)
        masks = masks.to(device)

        if masks.ndim == 3:
            masks = masks.unsqueeze(1)
        elif masks.ndim == 4 and masks.shape[-1] == 1:
            masks = masks.permute(0, 3, 1, 2)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            preds = model(images)

            if isinstance(preds, (tuple, list)):
                out1, out2, out3, out4 = preds

                loss1 = criterion_main(out1, masks) + 0.3 * criterion_bound(out1, masks)
                loss2 = criterion_main(out2, masks)
                loss3 = criterion_main(out3, masks)
                loss4 = criterion_main(out4, masks)

                loss = loss1 + 0.3 * loss2 + 0.2 * loss3 + 0.1 * loss4
            else:
                loss_main = criterion_main(preds, masks)
                loss_bound = criterion_bound(preds, masks)
                loss = loss_main + 0.5 * loss_bound

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        loop.set_postfix(loss=loss.item(), lr=get_lr(optimizer))

    return running_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0

    metrics_score = {"dice": 0.0, "iou": 0.0, "recall": 0.0, "precision": 0.0}

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="[Val]", leave=False):
            images = images.to(device)
            masks = masks.to(device)

            if masks.ndim == 3:
                masks = masks.unsqueeze(1)
            elif masks.ndim == 4 and masks.shape[-1] == 1:
                masks = masks.permute(0, 3, 1, 2)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                preds = model(images)

                if isinstance(preds, (tuple, list)):
                    preds = preds[0]

                loss = criterion(preds, masks)

            running_loss += loss.item()

            preds_prob = torch.sigmoid(preds)
            preds_bin = (preds_prob > 0.5).float()

            batch_metrics = calculate_metrics(preds_bin, masks.float())

            for k in metrics_score:
                if k in batch_metrics:
                    metrics_score[k] += batch_metrics[k]

    epoch_loss = running_loss / len(loader)
    for k in metrics_score:
        metrics_score[k] /= len(loader)

    return epoch_loss, metrics_score


if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    seed_everything(Config.SEED)
    device = Config.DEVICE

    print(f"Starting training | Device: {device} | Model: MambaSeg")
    print("=" * 60)
    print(f"Experiment Name      : {Config.EXP_NAME}")
    print(f"Seed                 : {Config.SEED}")
    print(f"IMG_SIZE             : {Config.IMG_SIZE}")
    print(f"Layer3 Mamba         : {Config.USE_LAYER3_MAMBA}")
    print(f"Bottleneck Mamba     : {Config.USE_BOTTLENECK_MAMBA}")
    print(f"D4 Mamba             : {Config.USE_D4_MAMBA}")
    print(f"Conv Refine          : {Config.USE_CONV_REFINE}")
    print("=" * 60)

    train_loader, val_loader, _ = get_loaders(Config)
    print(f"Data loading complete: Train set {len(train_loader)} batches | Val set {len(val_loader)} batches")

    model = UNet(
        n_classes=Config.NUM_CLASSES,
        use_layer3_mamba=Config.USE_LAYER3_MAMBA,
        use_d4_mamba=Config.USE_D4_MAMBA,
        use_bottleneck_mamba=Config.USE_BOTTLENECK_MAMBA,
        use_conv_refine=Config.USE_CONV_REFINE
    ).to(device)

    print(f"!!! CURRENT MODEL PARAMS: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f} M")

    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=1e-6
    )

    criterion_main = StructureLoss()
    criterion_bound = BoundaryIoULoss().to(device)

    scaler = GradScaler()
    best_dice = 0.0

    best_ckpt_path = f"checkpoints/best_model_{Config.EXP_NAME}.pth"
    last_ckpt_path = f"checkpoints/last_model_{Config.EXP_NAME}.pth"

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion_main=criterion_main,
            criterion_bound=criterion_bound,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
        )

        val_loss, metrics = validate(
            model=model,
            loader=val_loader,
            criterion=criterion_main,
            device=device,
        )

        scheduler.step()
        current_lr = get_lr(optimizer)

        save_msg = ""
        if metrics["dice"] > best_dice:
            best_dice = metrics["dice"]
            torch.save(model.state_dict(), best_ckpt_path)
            save_msg = "Best Dice!"

        torch.save(model.state_dict(), last_ckpt_path)

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"   Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"   Dice: {metrics['dice']:.4f} | IoU: {metrics['iou']:.4f}")
        print(f"   Recall: {metrics['recall']:.4f} | Precision: {metrics['precision']:.4f}")
        print(f"   LR: {current_lr:.6f} | {save_msg}")
        print("-" * 60)

    print(f"\nFinish! Best Dice: {best_dice:.4f}")
    print(f"Best checkpoint saved to: {best_ckpt_path}")
    print(f"Last checkpoint saved to: {last_ckpt_path}")