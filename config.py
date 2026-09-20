import torch
import os


class Config:
    IMG_SIZE = 352
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    NUM_WORKERS = 2
    EPOCHS = 100
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Multi-seed experiments: [42, 3407, 2025, 666, 1234]
    SEED = 2025
    MULTI_SEEDS = [42, 3407, 2025, 666, 1234]

    # Fixed data split seed
    # Keep this unchanged for all experiments
    SPLIT_SEED = 42

    # Internal validation ratio
    # 1450 development images -> 1305 train + 145 validation
    VAL_RATIO = 0.10
    NUM_CLASSES = 1

    # Main LiteMamba-Seg architecture switches
    USE_LAYER3_MAMBA = False
    USE_D4_MAMBA = True
    USE_BOTTLENECK_MAMBA = True
    USE_CONV_REFINE = True

    # Reviewer #5: ImageNet-pretrained ResNet34
    # True = ImageNet pretrained
    # False = random initialization
    USE_PRETRAINED = True

    # Controlled D4 placement experiment only:
    # "none", "pre_up", "pre_fusion", "post_fusion", "post_conv"
    D4_MAMBA_POS = "post_conv"

    # Experiment name
    EXP_NAME = "2025"

    # Evaluation settings
    PRED_THRESHOLD = 0.5
    METRIC_SMOOTH = 1e-5

    # Profiling settings
    PROFILE_BATCH_SIZE = 1
    PROFILE_WARMUP = 100
    PROFILE_REPEATS = 300
    PROFILE_FP16 = True

    # Optional analysis switches
    EVAL_BOUNDARY_IOU = True
    EVAL_HD95 = True
    EVAL_CALIBRATION = True
    EVAL_LESION_FP = True
    ECE_BINS = 15

    # Robustness evaluation
    RUN_ROBUSTNESS_TEST = True
    ROBUSTNESS_TESTS = [
        "resolution",
        "jpeg",
        "gaussian_blur",
        "brightness",
        "contrast",
    ]

    BASE_DIR = "/root/autodl-tmp/LiteMamba-Seg/datasets"

    # Original PraNet training pool
    TRAIN_IMG_DIR = f"{BASE_DIR}/TrainDataset/image"
    TRAIN_MASK_DIR = f"{BASE_DIR}/TrainDataset/masks"

    SAVE_DIR = "./checkpoints"
    os.makedirs(SAVE_DIR, exist_ok=True)