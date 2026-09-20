import csv
import datetime
import glob
import os
from PIL import Image
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import Config
from models.MambaSeg import MambaSeg_UNet as LiteMamba

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_final_log(
    exp_name: str,
    dataset_name: str,
    mean_dice: float,
    mean_iou: float,
    mean_recall: float,
    mean_precision: float,
    model_name: str = "LiteMamba",
) -> None:
    ensure_dir("./experiment_logs")
    csv_path = f"./experiment_logs/final_results_{exp_name}.csv"
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "Timestamp",
                "Experiment",
                "Model",
                "Dataset",
                "Mean Dice",
                "Mean IoU",
                "Mean Recall",
                "Mean Precision",
                "Seed",
                "Layer3 Mamba",
                "Bottleneck Mamba",
                "D4 Mamba",
                "Conv Refine",
                "Note",
            ])

        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([
            now,
            exp_name,
            model_name,
            dataset_name,
            f"{mean_dice:.4f}",
            f"{mean_iou:.4f}",
            f"{mean_recall:.4f}",
            f"{mean_precision:.4f}",
            Config.SEED,
            Config.USE_LAYER3_MAMBA,
            Config.USE_BOTTLENECK_MAMBA,
            Config.USE_D4_MAMBA,
            Config.USE_CONV_REFINE,
            "",
        ])

    print(f"✅ [{dataset_name}] results appended to {csv_path}")


def save_detailed_log(
    exp_name: str,
    dataset_name: str,
    img_name: str,
    dice: float,
    iou: float,
    recall: float,
    precision: float,
) -> None:
    detail_dir = f"./experiment_logs/details/{exp_name}/{dataset_name}"
    ensure_dir(detail_dir)

    txt_path = os.path.join(detail_dir, "scores.txt")
    with open(txt_path, "a", encoding="utf-8") as f:
        f.write(
            f"{img_name}: "
            f"Dice={dice:.4f}, IoU={iou:.4f}, Recall={recall:.4f}, Precision={precision:.4f}\n"
        )


def calculate_metrics_simple(pred: torch.Tensor, target: torch.Tensor):
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()

    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()

    pred_sum = pred.sum()
    target_sum = target.sum()

    dice = (2.0 * tp + 1e-5) / (pred_sum + target_sum + 1e-5)
    iou = (tp + 1e-5) / (tp + fp + fn + 1e-5)

    if target_sum.item() == 0:
        recall = 1.0 if pred_sum.item() == 0 else 0.0
    else:
        recall = float((tp / target_sum).item())

    if pred_sum.item() == 0:
        precision = 1.0 if target_sum.item() == 0 else 0.0
    else:
        precision = float((tp / pred_sum).item())

    return (
        float(dice.item()),
        float(iou.item()),
        recall,
        precision,
    )


def test_dataset(model, img_dir: str, mask_dir: str, dataset_name: str, exp_name: str):
    print(f"\n🚀 Testing: {dataset_name} ...")

    save_pred_dir = f"./results/LiteMamba_Visuals/{exp_name}/{dataset_name}/Preds"
    ensure_dir(save_pred_dir)

    detail_dir = f"./experiment_logs/details/{exp_name}/{dataset_name}"
    ensure_dir(detail_dir)
    detail_log_path = os.path.join(detail_dir, "scores.txt")
    if os.path.exists(detail_log_path):
        os.remove(detail_log_path)

    img_paths = sorted(glob.glob(os.path.join(img_dir, "*")))
    img_paths = [p for p in img_paths if p.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif"))]

    if not img_paths:
        print(f"⚠️ Image not found: {img_dir}")
        return None, None, None, None

    dice_list, iou_list, recall_list, precision_list = [], [], [], []
    missing_count, mismatch_count = 0, 0

    target_size = Config.IMG_SIZE if isinstance(Config.IMG_SIZE, tuple) else (Config.IMG_SIZE, Config.IMG_SIZE)

    model.eval()

    for img_path in tqdm(img_paths, desc=f"Evaluating {dataset_name}", leave=False):
        try:
            image = Image.open(img_path).convert("RGB")
            original_w, original_h = image.size
            image_resized = image.resize(target_size, Image.BILINEAR)

            img_np = np.array(image_resized).astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_norm = (img_np - mean) / std
            img_tensor = torch.tensor(img_norm).permute(2, 0, 1).unsqueeze(0).float().to(device)

            base_name = os.path.basename(img_path)
            name_no_ext = os.path.splitext(base_name)[0]

            mask_path = None
            possible_mask_names = [base_name, name_no_ext + ".png", name_no_ext + ".jpg"]
            for m_name in possible_mask_names:
                candidate = os.path.join(mask_dir, m_name)
                if os.path.exists(candidate):
                    mask_path = candidate
                    break

            if not mask_path:
                print(f"\n⚠️ Mask not found for {base_name}")
                missing_count += 1
                continue

            mask_gt = Image.open(mask_path).convert("L")
            mask_np = np.array(mask_gt).astype(np.float32) / 255.0
            mask_tensor = torch.tensor(mask_np > 0.5).float().to(device)

            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    pred = model(img_tensor)
                    if isinstance(pred, (tuple, list)):
                        pred = pred[0]

              
                pred = pred.float()

            
                pred = F.interpolate(
                    pred.unsqueeze(0) if pred.ndim == 3 else pred,
                    size=(original_h, original_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze()

                pred = torch.sigmoid(pred)

            if pred.shape != mask_tensor.shape:
                print(f"\n⚠️ Shape mismatch on {base_name}: pred {pred.shape} vs mask {mask_tensor.shape}")
                mismatch_count += 1
                continue

            d, i, r, p = calculate_metrics_simple(pred, mask_tensor)
            dice_list.append(d)
            iou_list.append(i)
            recall_list.append(r)
            precision_list.append(p)

            save_detailed_log(exp_name, dataset_name, base_name, d, i, r, p)

            pred_save = (pred > 0.5).float().cpu().numpy()
            pred_save = (pred_save * 255).astype(np.uint8)
            save_name = name_no_ext + ".png"
            cv2.imwrite(os.path.join(save_pred_dir, save_name), pred_save)

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

    if missing_count > 0 or mismatch_count > 0:
        print(f"\n⚠️ [Warning] {dataset_name}: Missed {missing_count} masks, Skipped {mismatch_count} shape mismatches.")

    if dice_list:
        mean_dice = np.mean(dice_list)
        mean_iou = np.mean(iou_list)
        mean_recall = np.mean(recall_list)
        mean_precision = np.mean(precision_list)

        save_final_log(
            exp_name=exp_name,
            dataset_name=dataset_name,
            mean_dice=mean_dice,
            mean_iou=mean_iou,
            mean_recall=mean_recall,
            mean_precision=mean_precision,
        )
        return mean_dice, mean_iou, mean_recall, mean_precision

    return None, None, None, None


if __name__ == "__main__":
    print("=" * 60)
    print(f"Experiment Name      : {Config.EXP_NAME}")
    print(f"Seed                 : {Config.SEED}")
    print(f"IMG_SIZE             : {Config.IMG_SIZE}")
    print(f"Layer3 Mamba         : {Config.USE_LAYER3_MAMBA}")
    print(f"Bottleneck Mamba     : {Config.USE_BOTTLENECK_MAMBA}")
    print(f"D4 Mamba             : {Config.USE_D4_MAMBA}")
    print(f"Conv Refine          : {Config.USE_CONV_REFINE}")
    print("=" * 60)

    model = LiteMamba(
        n_classes=Config.NUM_CLASSES,
        use_layer3_mamba=Config.USE_LAYER3_MAMBA,
        use_d4_mamba=Config.USE_D4_MAMBA,
        use_bottleneck_mamba=Config.USE_BOTTLENECK_MAMBA,
        use_conv_refine=Config.USE_CONV_REFINE,
    ).to(device)

    weights_path = f"./checkpoints/best_model_{Config.EXP_NAME}.pth"

    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"✅ Loaded best weights from: {weights_path}")
    else:
        print(f"❌ Error: Model weights not found at {weights_path}")
        exit()

    test_datasets = ["CVC-ClinicDB", "CVC-ColonDB", "ETIS-LaribPolypDB", "Kvasir"]
    results_summary = {}

    for dataset_name in test_datasets:
        img_dir = f"{Config.BASE_DIR}/TestDataset/{dataset_name}/images"
        mask_dir = f"{Config.BASE_DIR}/TestDataset/{dataset_name}/masks"

        if os.path.exists(img_dir):
            m_dice, m_iou, m_recall, m_precision = test_dataset(
                model=model,
                img_dir=img_dir,
                mask_dir=mask_dir,
                dataset_name=dataset_name,
                exp_name=Config.EXP_NAME,
            )
            if m_dice is not None:
                results_summary[dataset_name] = {
                    "dice": m_dice,
                    "iou": m_iou,
                    "recall": m_recall,
                    "precision": m_precision,
                }
        else:
            print(f"⚠️ Skipping {dataset_name}: Path not found ({img_dir})")

    print("\n" + "=" * 72)
    print(f"🎉 All Datasets Tested! Experiment: {Config.EXP_NAME}")
    print(f"| {'Dataset':<18} | {'Dice':<6} | {'IoU':<6} | {'Recall':<6} | {'Prec':<6} |")
    print("|" + "-" * 19 + "|" + "-" * 8 + "|" + "-" * 8 + "|" + "-" * 8 + "|" + "-" * 8 + "|")
    for name, m in results_summary.items():
        print(
            f"| {name:<18} | {m['dice']:.4f} | {m['iou']:.4f} | "
            f"{m['recall']:.4f} | {m['precision']:.4f} |"
        )
    print("=" * 72)