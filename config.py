import torch
import os

class Config:
    IMG_SIZE = 352
    BATCH_SIZE = 16   
    LEARNING_RATE = 1e-4
    NUM_WORKERS = 2
    EPOCHS = 100
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42 #SEEDS = [42, 3407, 2025, 666, 1234]
    NUM_CLASSES = 1
    USE_LAYER3_MAMBA = False
    USE_D4_MAMBA = True
    USE_BOTTLENECK_MAMBA = True
    USE_CONV_REFINE = True
    EXP_NAME = "42"

    BASE_DIR = "/root/autodl-tmp/LiteMamba-Seg/datasets"
    
    TRAIN_IMG_DIR = f"{BASE_DIR}/TrainDataset/image"
    TRAIN_MASK_DIR = f"{BASE_DIR}/TrainDataset/masks"
    VAL_IMG_DIR = f"{BASE_DIR}/TestDataset/CVC-ClinicDB/images"
    VAL_MASK_DIR = f"{BASE_DIR}/TestDataset/CVC-ClinicDB/masks"
    

    TEST_IMG_DIR = f"{BASE_DIR}/TestDataset/ETIS-LaribPolypDB/images"
    TEST_MASK_DIR = f"{BASE_DIR}/TestDataset/ETIS-LaribPolypDB/masks"
    

    SAVE_DIR = "./checkpoints"
    os.makedirs(SAVE_DIR, exist_ok=True)
