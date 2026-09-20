import glob
import os
import random

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class PolypDataset(Dataset):
    def __init__(self, data_pairs, transform=None):
        self.data_pairs = data_pairs
        self.transform = transform

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, index):
        pair = self.data_pairs[index]
        img_path = pair["image"]
        mask_path = pair["mask"]

        try:
            image = np.array(Image.open(img_path).convert("RGB"))
            mask = np.array(
                Image.open(mask_path).convert("L"), dtype=np.float32
            )

        
            if mask.max() > 1.0:
                mask = mask / 255.0


            if self.transform is not None:
                augmentations = self.transform(image=image, mask=mask)
                image = augmentations["image"]
                mask = augmentations["mask"]

      
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask)

            mask = (mask > 0.5).float()

            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            elif mask.ndim == 3 and mask.shape[-1] == 1:
                mask = mask.permute(2, 0, 1)

            return image, mask

        except Exception as e:
            raise RuntimeError(
                f"Failed to load sample: image={img_path}, mask={mask_path}. Reason: {e}"
            ) from e


def collect_images_and_masks(img_dir, mask_dir, dataset_name="Dataset"):
    valid_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif"]

    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Image directory does not exist: {img_dir}")
    if not os.path.exists(mask_dir):
        raise FileNotFoundError(f"Mask directory does not exist: {mask_dir}")

    image_paths = []
    for ext in valid_extensions:
        image_paths.extend(glob.glob(os.path.join(img_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(img_dir, ext.upper())))

    image_paths = sorted(set(image_paths))
    data_pairs = []

    for img_p in image_paths:
        base_name = os.path.basename(img_p)
        name_no_ext = os.path.splitext(base_name)[0]

        potential_mask_names = [
            base_name,
            name_no_ext + ".png",
            name_no_ext + ".PNG",
            name_no_ext + ".jpg",
            name_no_ext + ".JPG",
            name_no_ext + ".jpeg",
            name_no_ext + ".JPEG",
            name_no_ext + ".tif",
            name_no_ext + ".TIF",
            name_no_ext + ".bmp",
            name_no_ext + ".BMP",
        ]

        final_mask_path = None
        for m_name in potential_mask_names:
            candidate = os.path.join(mask_dir, m_name)
            if os.path.exists(candidate):
                final_mask_path = candidate
                break

        if final_mask_path is not None:
            data_pairs.append({"image": img_p, "mask": final_mask_path})

    print(f"[{dataset_name}] Found {len(data_pairs)} sample pairs.")
    return data_pairs


def save_split_files(train_pairs, val_pairs, split_dir="./splits"):
    os.makedirs(split_dir, exist_ok=True)
    train_split_path = os.path.join(split_dir, "train_split.txt")
    val_split_path = os.path.join(split_dir, "val_split.txt")

    with open(train_split_path, "w", encoding="utf-8") as f:
        for pair in train_pairs:
            f.write(os.path.basename(pair["image"]) + "\n")

    with open(val_split_path, "w", encoding="utf-8") as f:
        for pair in val_pairs:
            f.write(os.path.basename(pair["image"]) + "\n")

    print(f"Saved train split: {train_split_path}")
    print(f"Saved validation split: {val_split_path}")


def get_loaders(config):
    print("--- Loading PraNet Development Pool ---")

    all_train_pairs = collect_images_and_masks(
        config.TRAIN_IMG_DIR, config.TRAIN_MASK_DIR, "Development Pool"
    )

    if len(all_train_pairs) == 0:
        raise ValueError("Error: No training data found!")

    all_train_pairs = sorted(all_train_pairs, key=lambda x: x["image"])

    split_seed = getattr(config, "SPLIT_SEED", 42)
    val_ratio = getattr(config, "VAL_RATIO", 0.10)

    rng = random.Random(split_seed)
    shuffled_pairs = all_train_pairs.copy()
    rng.shuffle(shuffled_pairs)

    val_size = int(len(shuffled_pairs) * val_ratio)
    val_pairs = shuffled_pairs[:val_size]
    train_pairs = shuffled_pairs[val_size:]

    print("=" * 60)
    print(f"Total Development Pool : {len(all_train_pairs)}")
    print(f"Training Samples        : {len(train_pairs)}")
    print(f"Validation Samples      : {len(val_pairs)}")
    print(f"Validation Ratio        : {val_ratio:.2f}")
    print(f"Split Seed              : {split_seed}")
    print("=" * 60)

    save_split_files(
        train_pairs=train_pairs, val_pairs=val_pairs, split_dir="./splits"
    )

    train_transform = A.Compose(
        [
            A.Resize(height=config.IMG_SIZE, width=config.IMG_SIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.15,
                rotate_limit=45,
                p=0.5,
            ),
            A.OneOf(
                [
                    A.GridDistortion(
                        num_steps=5, distort_limit=0.05, p=1.0
                    ),
                    A.ElasticTransform(
                        alpha=1,
                        sigma=50,
                        alpha_affine=50,
                        p=1.0,
                    ),
                ],
                p=0.3,
            ),
            A.OneOf(
                [
                    A.CLAHE(
                        clip_limit=4.0, tile_grid_size=(8, 8), p=1.0
                    ),
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2, contrast_limit=0.2, p=1.0
                    ),
                    A.ColorJitter(
                        brightness=0.1,
                        contrast=0.1,
                        saturation=0.2,
                        hue=0.1,
                        p=1.0,
                    ),
                ],
                p=0.4,
            ),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.MotionBlur(blur_limit=5, p=1.0),
                    A.Sharpen(
                        alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0
                    ),
                ],
                p=0.3,
            ),
            
            A.CoarseDropout(
                max_holes=8,
                max_height=32,
                max_width=32,
                min_holes=1,
                min_height=8,
                min_width=8,
                fill_value=0,
                mask_fill_value=None,  
                p=0.3,
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [
            A.Resize(height=config.IMG_SIZE, width=config.IMG_SIZE),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )

    train_ds = PolypDataset(train_pairs, transform=train_transform)
    val_ds = PolypDataset(val_pairs, transform=val_transform)

    g = torch.Generator()
    g.manual_seed(config.SEED)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, val_loader