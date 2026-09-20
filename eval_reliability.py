import os
import io
import glob
import cv2
import torch
import numpy as np
import torch.nn.functional as F

from PIL import Image, ImageEnhance, ImageFilter
from scipy.ndimage import distance_transform_edt
from torchvision import transforms

from config import Config
from models.MambaSeg import MambaSeg_UNet


# ============================================================
# Settings
# ============================================================

DATASETS = [
    "Kvasir",
    "CVC-ClinicDB",
    "CVC-ColonDB",
    "ETIS-LaribPolypDB",
]

PERTURBATIONS = [
    "clean",
    "blur",
    "jpeg",
    "brightness",
    "contrast",
    "low_resolution",
]

THRESHOLD = 0.5
BOUNDARY_RATIO = 0.02
ECE_BINS = 15


MODEL_CONFIGS = {
    "LiteMamba-Seg": {
        "checkpoint": "./checkpoints/best_model_2025.pth",
        "layer3": False,
        "d4": True,
        "bottleneck": True,
        "conv": True,
    },

    "Baseline": {
        "checkpoint": "./checkpoints/best_model_Baseline.pth",
        "layer3": False,
        "d4": False,
        "bottleneck": False,
        "conv": False,
    },
}


transform = transforms.Compose([
    transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# ============================================================
# Model
# ============================================================

def build_model(cfg, device):
    model = MambaSeg_UNet(
        n_classes=Config.NUM_CLASSES,
        use_layer3_mamba=cfg["layer3"],
        use_d4_mamba=cfg["d4"],
        use_bottleneck_mamba=cfg["bottleneck"],
        use_conv_refine=cfg["conv"],
    ).to(device)

    ckpt = cfg["checkpoint"]

    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    model.load_state_dict(
        torch.load(ckpt, map_location=device)
    )

    model.eval()
    return model


# ============================================================
# Dataset pairing
# ============================================================

def collect_pairs(dataset_name):
    img_dir = os.path.join(
        Config.BASE_DIR,
        "TestDataset",
        dataset_name,
        "images"
    )

    mask_dir = os.path.join(
        Config.BASE_DIR,
        "TestDataset",
        dataset_name,
        "masks"
    )

    imgs = sorted(glob.glob(os.path.join(img_dir, "*.*")))
    masks = sorted(glob.glob(os.path.join(mask_dir, "*.*")))

    mask_dict = {
        os.path.splitext(os.path.basename(p))[0]: p
        for p in masks
    }

    pairs = []

    for img in imgs:
        stem = os.path.splitext(os.path.basename(img))[0]

        if stem in mask_dict:
            pairs.append((img, mask_dict[stem]))

    return pairs


# ============================================================
# Perturbations
# ============================================================

def perturb(image, mode):
    if mode == "clean":
        return image

    if mode == "blur":
        return image.filter(
            ImageFilter.GaussianBlur(radius=2.0)
        )

    if mode == "jpeg":
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=50)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    if mode == "brightness":
        return ImageEnhance.Brightness(
            image
        ).enhance(1.2)

    if mode == "contrast":
        return ImageEnhance.Contrast(
            image
        ).enhance(0.8)

    if mode == "low_resolution":
        original_size = image.size

        low = image.resize(
            (176, 176),
            Image.BILINEAR
        )

        return low.resize(
            original_size,
            Image.BILINEAR
        )

    raise ValueError(mode)


# ============================================================
# Prediction
# ============================================================

def predict(model, image, h, w, device):
    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x).float()

        logits = F.interpolate(
            logits,
            size=(h, w),
            mode="bilinear",
            align_corners=False
        )

        prob = torch.sigmoid(logits)

    prob = prob.squeeze().cpu().numpy()

    pred = (prob >= THRESHOLD).astype(np.uint8)

    return prob, pred


# ============================================================
# Metrics
# ============================================================

def dice(gt, pred, smooth=1e-5):
    gt = gt.astype(np.float32)
    pred = pred.astype(np.float32)

    inter = (gt * pred).sum()

    return float(
        (2 * inter + smooth)
        /
        (gt.sum() + pred.sum() + smooth)
    )


def boundary(mask):
    mask = mask.astype(np.uint8)

    h, w = mask.shape

    k = max(
        1,
        int(round(
            np.sqrt(h * h + w * w)
            * BOUNDARY_RATIO
        ))
    )

    if k % 2 == 0:
        k += 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (k, k)
    )

    dilated = cv2.dilate(mask, kernel)
    eroded = cv2.erode(mask, kernel)

    return (dilated - eroded) > 0


def boundary_iou(gt, pred):
    bg = boundary(gt)
    bp = boundary(pred)

    inter = np.logical_and(bg, bp).sum()
    union = np.logical_or(bg, bp).sum()

    return 1.0 if union == 0 else float(inter / union)


def contour(mask):
    mask = mask.astype(np.uint8)

    eroded = cv2.erode(
        mask,
        np.ones((3, 3), np.uint8)
    )

    return mask.astype(bool) ^ eroded.astype(bool)


def hd95(gt, pred):
    h, w = gt.shape
    max_dist = np.sqrt(h * h + w * w)

    if gt.sum() == 0 and pred.sum() == 0:
        return 0.0

    if gt.sum() == 0 or pred.sum() == 0:
        return float(max_dist)

    bg = contour(gt)
    bp = contour(pred)

    if bg.sum() == 0 or bp.sum() == 0:
        return float(max_dist)

    dt_gt = distance_transform_edt(~bg)
    dt_pred = distance_transform_edt(~bp)

    d1 = dt_gt[bp]
    d2 = dt_pred[bg]

    return float(
        np.percentile(
            np.concatenate([d1, d2]),
            95
        )
    )


def ece(prob, gt, bins=ECE_BINS):
    prob = prob.reshape(-1)
    gt = gt.reshape(-1)

    edges = np.linspace(0, 1, bins + 1)

    total = 0.0

    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]

        if i == bins - 1:
            m = (prob >= lo) & (prob <= hi)
        else:
            m = (prob >= lo) & (prob < hi)

        if not np.any(m):
            continue

        confidence = prob[m].mean()
        accuracy = gt[m].mean()

        total += (
            m.mean()
            * abs(confidence - accuracy)
        )

    return float(total)


def lesion_fp(gt, pred):
    n_labels, labels = cv2.connectedComponents(
        pred.astype(np.uint8),
        connectivity=8
    )

    count = 0

    for label_id in range(1, n_labels):
        component = labels == label_id

        if np.logical_and(
            component,
            gt > 0
        ).sum() == 0:
            count += 1

    return count


# ============================================================
# Dataset evaluation
# ============================================================

def evaluate(model, dataset_name, device):
    pairs = collect_pairs(dataset_name)

    if not pairs:
        print(f"No data found: {dataset_name}")
        return None

    robustness = {
        p: []
        for p in PERTURBATIONS
    }

    biou_values = []
    hd95_values = []
    ece_values = []
    fp_values = []

    for i, (img_path, mask_path) in enumerate(pairs):
        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        gt = np.asarray(
            Image.open(mask_path).convert("L"),
            dtype=np.float32
        )

        if gt.max() > 1:
            gt /= 255.0

        if gt.shape != (h, w):
            gt = cv2.resize(
                gt,
                (w, h),
                interpolation=cv2.INTER_NEAREST
            )

        gt = (gt >= THRESHOLD).astype(np.uint8)

        clean_prob = None
        clean_pred = None

        for mode in PERTURBATIONS:
            test_image = perturb(image, mode)

            prob, pred = predict(
                model,
                test_image,
                h,
                w,
                device
            )

            robustness[mode].append(
                dice(gt, pred)
            )

            if mode == "clean":
                clean_prob = prob
                clean_pred = pred

        biou_values.append(
            boundary_iou(gt, clean_pred)
        )

        hd95_values.append(
            hd95(gt, clean_pred)
        )

        ece_values.append(
            ece(clean_prob, gt)
        )

        fp_values.append(
            lesion_fp(gt, clean_pred)
        )

        if (i + 1) % 50 == 0 or i + 1 == len(pairs):
            print(
                f"{dataset_name}: "
                f"{i + 1}/{len(pairs)}"
            )

    result = {
        p: float(np.mean(v))
        for p, v in robustness.items()
    }

    perturbed = [
        result[p]
        for p in PERTURBATIONS
        if p != "clean"
    ]

    result["mean_perturbed"] = float(
        np.mean(perturbed)
    )

    result["dice_drop"] = (
        result["clean"]
        - result["mean_perturbed"]
    )

    result["boundary_iou"] = float(
        np.mean(biou_values)
    )

    result["hd95"] = float(
        np.mean(hd95_values)
    )

    result["ece"] = float(
        np.mean(ece_values)
    )

    result["lesion_fp"] = float(
        np.mean(fp_values)
    )

    return result


# ============================================================
# Print
# ============================================================

def print_results(results):
    print("\nROBUSTNESS")
    print(
        f"{'Model':<16}{'Dataset':<12}"
        f"{'Clean':>8}{'Blur':>8}{'JPEG':>8}"
        f"{'Bright':>8}{'Contrast':>10}"
        f"{'LowRes':>8}{'Mean':>8}{'Drop':>8}"
    )

    for model_name, datasets in results.items():
        for ds, r in datasets.items():
            print(
                f"{model_name:<16}{ds:<12}"
                f"{r['clean']:>8.4f}"
                f"{r['blur']:>8.4f}"
                f"{r['jpeg']:>8.4f}"
                f"{r['brightness']:>8.4f}"
                f"{r['contrast']:>10.4f}"
                f"{r['low_resolution']:>8.4f}"
                f"{r['mean_perturbed']:>8.4f}"
                f"{r['dice_drop']:>8.4f}"
            )

    print("\nBOUNDARY / CALIBRATION / FALSE POSITIVES")
    print(
        f"{'Model':<16}{'Dataset':<12}"
        f"{'BIoU':>10}{'HD95':>10}"
        f"{'ECE':>10}{'FP/img':>10}"
    )

    for model_name, datasets in results.items():
        for ds, r in datasets.items():
            print(
                f"{model_name:<16}{ds:<12}"
                f"{r['boundary_iou']:>10.4f}"
                f"{r['hd95']:>10.2f}"
                f"{r['ece']:>10.4f}"
                f"{r['lesion_fp']:>10.4f}"
            )


# ============================================================
# Main
# ============================================================

def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))

    results = {}

    for model_name, cfg in MODEL_CONFIGS.items():
        print(f"\nEvaluating {model_name}")

        model = build_model(
            cfg,
            device
        )

        results[model_name] = {}

        for dataset in DATASETS:
            r = evaluate(
                model,
                dataset,
                device
            )

            if r is not None:
                results[model_name][dataset] = r

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print_results(results)


if __name__ == "__main__":
    main()