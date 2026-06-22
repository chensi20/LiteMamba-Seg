import os
import random 
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import glob

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
        img_path = pair['image']
        mask_path = pair['mask']

        try:
            image = np.array(Image.open(img_path).convert("RGB"))
            
            if os.path.exists(mask_path):
                mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
            else:
                h, w = image.shape[:2]
                mask = np.zeros((h, w), dtype=np.float32)

            if mask.max() > 1:
                mask = mask / 255.0
            
            mask[mask >= 0.5] = 1.0
            mask[mask < 0.5] = 0.0

            if self.transform is not None:
                augmentations = self.transform(image=image, mask=mask)
                image = augmentations["image"]
                mask = augmentations["mask"]
                
            if isinstance(mask, torch.Tensor):
                if mask.ndim == 2:
                    mask = mask.unsqueeze(0)
            elif isinstance(mask, np.ndarray):
                if mask.ndim == 2:
                    mask = torch.from_numpy(mask).unsqueeze(0)
                else:
                    mask = torch.from_numpy(mask).permute(2, 0, 1)

            return image, mask
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return torch.zeros((3, 352, 352)), torch.zeros((1, 352, 352))

def collect_images_and_masks(img_dir, mask_dir, dataset_name="Dataset"):
    valid_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif']
    image_paths = []
    
    if not os.path.exists(img_dir):
        print(f"Warning: Directory does not exist: {img_dir}")
        return []

    for ext in valid_extensions:
        image_paths.extend(glob.glob(os.path.join(img_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(img_dir, ext.upper())))
        
    data_pairs = []
    found_count = 0
    
    for img_p in image_paths:
        base_name = os.path.basename(img_p)
        name_no_ext = os.path.splitext(base_name)[0]
        
        potential_mask_names = [
            base_name,
            name_no_ext + ".png",
            name_no_ext + ".jpg",
            name_no_ext + ".jpeg",
            name_no_ext + ".tif"
        ]
        
        final_mask_path = None
        for m_name in potential_mask_names:
            candidate = os.path.join(mask_dir, m_name)
            if os.path.exists(candidate):
                final_mask_path = candidate
                break
        
        if final_mask_path:
            data_pairs.append({'image': img_p, 'mask': final_mask_path})
            found_count += 1
            
    print(f"[{dataset_name}] Found {found_count} sample pairs.")
    return data_pairs

def get_loaders(config):
    print("--- Loading Standard PraNet Dataset ---")
    train_pairs = collect_images_and_masks(config.TRAIN_IMG_DIR, config.TRAIN_MASK_DIR, "Train (1450)")
    
  
    val_pairs = collect_images_and_masks(config.VAL_IMG_DIR, config.VAL_MASK_DIR, "Validation (ClinicDB-Test)")

    test_pairs = collect_images_and_masks(config.TEST_IMG_DIR, config.TEST_MASK_DIR, "Test (External)")

    if len(train_pairs) == 0:
        raise ValueError("Error: No training data found! Please check Kaggle paths in config.py.")

    train_transform = A.Compose([
        A.Resize(height=config.IMG_SIZE, width=config.IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.15, rotate_limit=45, p=0.5),
        A.OneOf([
            A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0),
        ], p=0.3),
        A.OneOf([
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.2, hue=0.1, p=1.0),
        ], p=0.4),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0), 
        ], p=0.3),
        A.CoarseDropout(
            max_holes=8, max_height=32, max_width=32, 
            min_holes=1, min_height=8, min_width=8, 
            fill_value=0, mask_fill_value=0, p=0.3
        ),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
        ToTensorV2(),
    ])

    val_test_transform = A.Compose([
        A.Resize(height=config.IMG_SIZE, width=config.IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
        ToTensorV2(),
    ])

    # Dataset
    train_ds = PolypDataset(train_pairs, transform=train_transform)
    val_ds = PolypDataset(val_pairs, transform=val_test_transform)
    test_ds = PolypDataset(test_pairs, transform=val_test_transform) 

 
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
        generator=g               
    )
    
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, pin_memory=True, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, pin_memory=True, shuffle=False)

    return train_loader, val_loader, test_loader