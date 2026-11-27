
from pathlib import Path
import random

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode


# ==================================================
# 1. 공통 설정
# ==================================================

SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 8

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

PADDING_COLOR = tuple(round(m * 255) for m in MEAN)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


# ==================================================
# 2. Random Seed
# ==================================================

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==================================================
# 3. 메타데이터 로딩 및 데이터 분할
# ==================================================

def load_data(data_dir=DATA_DIR):
    df = pd.read_csv(Path(data_dir) / "metadata.csv")

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    return train_df, val_df, test_df


# ==================================================
# 4. Resize + Padding
# ==================================================

def resize_with_padding(
    image,
    size=IMAGE_SIZE,
    fill=PADDING_COLOR
):
    width, height = image.size

    scale = size / max(width, height)

    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))

    image = TF.resize(
        image,
        [new_height, new_width],
        interpolation=InterpolationMode.BILINEAR
    )

    pad_left = (size - new_width) // 2
    pad_top = (size - new_height) // 2

    pad_right = size - new_width - pad_left
    pad_bottom = size - new_height - pad_top

    image = TF.pad(
        image,
        [pad_left, pad_top, pad_right, pad_bottom],
        fill=fill,
        padding_mode="constant"
    )

    return image


# ==================================================
# 5. 공통 Transform
# ==================================================

def get_transform():
    return transforms.Compose([
        transforms.Lambda(resize_with_padding),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])


# ==================================================
# 6. Custom Dataset
# ==================================================

class ActionDataset(Dataset):

    def __init__(self, dataframe, data_dir, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = self.data_dir / row["image_path"]

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        label = int(row["target_id"])

        if self.transform is not None:
            image = self.transform(image)

        return image, label


# ==================================================
# 7. DataLoader
# ==================================================

def get_dataloaders(
    data_dir=DATA_DIR,
    batch_size=BATCH_SIZE,
    seed=SEED
):
    train_df, val_df, test_df = load_data(data_dir)

    transform = get_transform()

    train_dataset = ActionDataset(train_df, data_dir, transform)
    val_dataset = ActionDataset(val_df, data_dir, transform)
    test_dataset = ActionDataset(test_df, data_dir, transform)

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
        generator=generator
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )

    return train_loader, val_loader, test_loader
