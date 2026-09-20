import os
import glob
import cv2
import torch
import numpy as np
import torch.nn.functional as F

from scipy.ndimage import distance_transform_edt
from torchvision import transforms
from PIL import Image

from config import Config
from models.MambaSeg import MambaSeg_UNet


# ============================================================
# 1. Boundary metric functions
# ============================================================
def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    Convert a binary mask to a boundary band.

    Boundary width is defined relative to 2% of the image diagonal.
    """
    mask = mask.astype(np.uint8)

    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)

    dilation_size = max(
        1,
        int(round(img_diag * dilation_ratio))
    )

    # Use odd kernel size for symmetric morphology
    if dilation_size % 2 == 0:
        dilation_size += 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (dilation_size, dilation_size)
    )

    dilated = cv2.dilate(
        mask,
        kernel,
        iterations=1
    )

    eroded = cv2.erode(
        mask,
        kernel,
        iterations=1
    )

    boundary = (dilated - eroded) > 0

    return boundary


def compute_boundary_iou(
    gt,
    pred,
    dilation_ratio=0.02
):
    """
    Boundary IoU.

    Both prediction and ground truth are binarized at 0.5.
    """
    gt_bin = gt > 0.5
    pred_bin = pred > 0.5

    # Both empty
    if gt_bin.sum() == 0 and pred_bin.sum() == 0:
        return 1.0

    gt_boundary = mask_to_boundary(
        gt_bin,
        dilation_ratio
    )

    pred_boundary = mask_to_boundary(
        pred_bin,
        dilation_ratio
    )

    intersection = np.logical_and(
        gt_boundary,
        pred_boundary
    ).sum()

    union = np.logical_or(
        gt_boundary,
        pred_boundary
    ).sum()

    if union == 0:
        return 1.0

    return float(intersection / union)


def extract_boundary(mask):
    """
    Extract a one-pixel inner contour for HD95.
    """
    mask = mask.astype(np.uint8)

    kernel = np.ones(
        (3, 3),
        dtype=np.uint8
    )

    eroded = cv2.erode(
        mask,
        kernel,
        iterations=1
    )

    boundary = (
        mask.astype(bool)
        ^ eroded.astype(bool)
    )

    return boundary


def compute_hd95(gt, pred):
    """
    Compute symmetric HD95 in pixel units.

    If one mask is empty while the other contains foreground,
    the image diagonal is used as a penalty distance.
    """
    gt_bin = gt > 0.5
    pred_bin = pred > 0.5

    h, w = gt_bin.shape
    max_distance = np.sqrt(
        h ** 2 + w ** 2
    )

    # Both empty
    if gt_bin.sum() == 0 and pred_bin.sum() == 0:
        return 0.0

    # One empty, one non-empty
    if gt_bin.sum() == 0 or pred_bin.sum() == 0:
        return float(max_distance)

    gt_boundary = extract_boundary(
        gt_bin
    )

    pred_boundary = extract_boundary(
        pred_bin
    )

    # Extremely rare degenerate case
    if (
        gt_boundary.sum() == 0
        or pred_boundary.sum() == 0
    ):
        return float(max_distance)

    # Distance to closest GT boundary pixel
    dt_gt = distance_transform_edt(
        ~gt_boundary
    )

    # Distance to closest predicted boundary pixel
    dt_pred = distance_transform_edt(
        ~pred_boundary
    )

    pred_to_gt = dt_gt[
        pred_boundary
    ]

    gt_to_pred = dt_pred[
        gt_boundary
    ]

    all_distances = np.concatenate(
        [
            pred_to_gt,
            gt_to_pred
        ]
    )

    return float(
        np.percentile(
            all_distances,
            95
        )
    )


# ============================================================
# 2. Image-mask pairing
# ============================================================
def get_file_stem(path):
    return os.path.splitext(
        os.path.basename(path)
    )[0]


def collect_image_mask_pairs(
    img_dir,
    mask_dir
):
    image_paths = sorted(
        glob.glob(
            os.path.join(
                img_dir,
                "*.*"
            )
        )
    )

    mask_paths = sorted(
        glob.glob(
            os.path.join(
                mask_dir,
                "*.*"
            )
        )
    )

    mask_dict = {
        get_file_stem(path): path
        for path in mask_paths
    }

    pairs = []

    for img_path in image_paths:
        stem = get_file_stem(
            img_path
        )

        if stem not in mask_dict:
            print(
                f"[Warning] No matching mask for: "
                f"{img_path}"
            )
            continue

        pairs.append(
            (
                img_path,
                mask_dict[stem]
            )
        )

    return pairs


# ============================================================
# 3. Evaluate one dataset
# ============================================================
def evaluate_dataset(
    model,
    dataset_name,
    data_dir,
    device,
    dilation_ratio=0.02
):
    img_dir = os.path.join(
        data_dir,
        dataset_name,
        "images"
    )

    gt_dir = os.path.join(
        data_dir,
        dataset_name,
        "masks"
    )

    pairs = collect_image_mask_pairs(
        img_dir,
        gt_dir
    )

    if len(pairs) == 0:
        print(
            f"[{dataset_name}] "
            f"No valid image-mask pairs found."
        )
        return None

    print(
        f"[{dataset_name}] "
        f"{len(pairs)} image-mask pairs"
    )

    img_transform = transforms.Compose(
        [
            transforms.Resize(
                (
                    Config.IMG_SIZE,
                    Config.IMG_SIZE
                )
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406
                ],
                std=[
                    0.229,
                    0.224,
                    0.225
                ]
            )
        ]
    )

    boundary_iou_values = []
    hd95_values = []

    empty_prediction_count = 0

    with torch.no_grad():

        for img_path, gt_path in pairs:

            # ------------------------------------------------
            # Load original image
            # ------------------------------------------------
            raw_img = Image.open(
                img_path
            ).convert("RGB")

            w_orig, h_orig = raw_img.size

            input_tensor = (
                img_transform(
                    raw_img
                )
                .unsqueeze(0)
                .to(device)
            )

            # ------------------------------------------------
            # Forward pass
            # ------------------------------------------------
            logits = model(
                input_tensor
            ).float()

            # Frozen evaluation protocol:
            # raw logits -> resize -> sigmoid -> threshold
            logits = F.interpolate(
                logits,
                size=(
                    h_orig,
                    w_orig
                ),
                mode="bilinear",
                align_corners=False
            )

            prob = torch.sigmoid(
                logits
            )

            prob = (
                prob
                .squeeze(0)
                .squeeze(0)
                .cpu()
                .numpy()
            )

            # ------------------------------------------------
            # Load GT
            # ------------------------------------------------
            gt_img = Image.open(
                gt_path
            ).convert("L")

            gt_arr = np.asarray(
                gt_img,
                dtype=np.float32
            )

            # Handle masks stored as [0,255]
            if gt_arr.max() > 1.0:
                gt_arr = gt_arr / 255.0

            # Safety check:
            # GT and prediction must share original resolution
            if gt_arr.shape != prob.shape:

                gt_arr = cv2.resize(
                    gt_arr,
                    (
                        w_orig,
                        h_orig
                    ),
                    interpolation=cv2.INTER_NEAREST
                )

            # ------------------------------------------------
            # Binary masks
            # ------------------------------------------------
            gt_bin = (
                gt_arr > 0.5
            ).astype(
                np.uint8
            )

            pred_bin = (
                prob > 0.5
            ).astype(
                np.uint8
            )

            if pred_bin.sum() == 0:
                empty_prediction_count += 1

            # ------------------------------------------------
            # Metrics
            # ------------------------------------------------
            boundary_iou = compute_boundary_iou(
                gt_bin,
                pred_bin,
                dilation_ratio=dilation_ratio
            )

            hd95 = compute_hd95(
                gt_bin,
                pred_bin
            )

            boundary_iou_values.append(
                boundary_iou
            )

            hd95_values.append(
                hd95
            )

    mean_boundary_iou = float(
        np.mean(
            boundary_iou_values
        )
    )

    std_boundary_iou = float(
        np.std(
            boundary_iou_values,
            ddof=1
        )
    )

    mean_hd95 = float(
        np.mean(
            hd95_values
        )
    )

    std_hd95 = float(
        np.std(
            hd95_values,
            ddof=1
        )
    )

    return {
        "dataset": dataset_name,
        "n": len(pairs),
        "boundary_iou_mean": mean_boundary_iou,
        "boundary_iou_std": std_boundary_iou,
        "hd95_mean": mean_hd95,
        "hd95_std": std_hd95,
        "empty_predictions": empty_prediction_count,
    }


# ============================================================
# 4. Main
# ============================================================
def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    if torch.cuda.is_available():
        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # Final LiteMamba-Seg
    # --------------------------------------------------------
    model = MambaSeg_UNet(
        n_classes=Config.NUM_CLASSES,

        use_layer3_mamba=False,

        use_d4_mamba=True,

        use_bottleneck_mamba=True,

        use_conv_refine=True,
    ).to(device)

    # --------------------------------------------------------
    # Load final seed-2025 checkpoint
    # --------------------------------------------------------
    weights_path = (
        "./checkpoints/"
        "best_model_2025.pth"
    )

    if not os.path.exists(
        weights_path
    ):
        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{weights_path}"
        )

    print(
        f"Loading checkpoint: "
        f"{weights_path}"
    )

    checkpoint = torch.load(
        weights_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint
    )

    model.eval()

    # --------------------------------------------------------
    # Test dataset root
    #
    # Change this to exactly the same location used by test.py
    # --------------------------------------------------------
    data_dir = (
        "./data/TestDataset"
    )

    datasets = [
        "Kvasir",
        "ClinicDB",
        "ColonDB",
        "ETIS"
    ]

    # Boundary width = 2% of image diagonal
    dilation_ratio = 0.02

    print("\n")
    print("=" * 100)

    print(
        f"{'Dataset':<18}"
        f"{'Boundary IoU ↑':>20}"
        f"{'HD95 (px) ↓':>20}"
        f"{'Empty Pred.':>15}"
    )

    print("=" * 100)

    all_results = []

    for dataset_name in datasets:

        result = evaluate_dataset(
            model=model,
            dataset_name=dataset_name,
            data_dir=data_dir,
            device=device,
            dilation_ratio=dilation_ratio
        )

        if result is None:
            continue

        all_results.append(
            result
        )

        print(
            f"{dataset_name:<18}"
            f"{result['boundary_iou_mean']:>10.4f}"
            f" ± "
            f"{result['boundary_iou_std']:<7.4f}"
            f"{result['hd95_mean']:>10.2f}"
            f" ± "
            f"{result['hd95_std']:<7.2f}"
            f"{result['empty_predictions']:>15d}"
        )

    print("=" * 100)

    print(
        "\nBoundary IoU settings:"
    )

    print(
        f"- Boundary width: "
        f"{dilation_ratio * 100:.1f}% "
        f"of image diagonal"
    )

    print(
        "- Prediction threshold: 0.5"
    )

    print(
        "- HD95 unit: pixels"
    )

    print(
        "- Empty prediction with non-empty GT: "
        "image diagonal used as HD95 penalty"
    )


if __name__ == "__main__":
    main()