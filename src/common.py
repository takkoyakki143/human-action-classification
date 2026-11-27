# src/common.py — 여러 노트북이 함께 쓰는 경로·데이터 로딩·전처리 코드
# 02_preprocessing에서 확정한 전처리를 이 파일 하나에 두고, 03_modeling은 여기서 불러 씁니다.
# → 노트북마다 복사하면 한쪽만 고쳐지는 실수가 생길 수 있어, "세 방식 동일 전처리"(규칙 2)를 코드로 보장하기 위함
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights

# ── 경로 ──
# 이 파일(src/) 기준 한 단계 위 = 프로젝트 최상위 폴더
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"
FIGURE_DIR = ROOT / "figures"

# ── 02_preprocessing에서 확정한 설정값 ──
SEED = 42          # 기본 Seed
IMG_SIZE = 224     # ResNet18 기본 입력 크기
BATCH_SIZE = 32    # 한 번에 모델에 넣는 사진 수

WEIGHTS = ResNet18_Weights.IMAGENET1K_V1              # 사용할 사전학습 가중치 버전 (규칙 1)
MEAN = list(WEIGHTS.transforms().mean)                 # ImageNet 채널별 평균 [R, G, B]
STD = list(WEIGHTS.transforms().std)                   # ImageNet 채널별 표준편차 [R, G, B]
PAD_COLOR = tuple(round(m * 255) for m in MEAN)        # 평균값을 0~255 색으로 바꾼 것 → (124, 116, 104)


def load_metadata():
    # utf-8-sig: CSV 맨 앞의 숨은 문자(BOM) 때문에 첫 컬럼명이 깨지는 것을 방지
    df = pd.read_csv(DATA_DIR / "metadata.csv", encoding="utf-8-sig")
    df["full_path"] = df["image_path"].apply(lambda p: DATA_DIR / p)  # 실제 이미지 경로
    return df


def load_class_names():
    """0~7번 순서대로 (영어 클래스명 목록, 한국어 클래스명 목록)을 돌려준다."""
    cm = pd.read_csv(DATA_DIR / "class_mapping.csv", encoding="utf-8-sig").sort_values("target_id")
    return cm["class_name"].tolist(), cm["class_name_ko"].tolist()


class ResizePad:
    """비율을 유지하며 긴 변을 size에 맞추고, 남는 곳을 fill 색으로 채워 정사각형으로 만든다."""

    def __init__(self, size=IMG_SIZE, fill=PAD_COLOR):
        self.size = size
        self.fill = fill

    def __call__(self, img):
        w, h = img.size                                   # 원본 가로, 세로
        scale = self.size / max(w, h)                     # 긴 변이 size가 되도록 하는 배율
        new_w = max(1, round(w * scale))
        new_h = max(1, round(h * scale))
        img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)   # 비율 유지 리사이즈
        canvas = Image.new("RGB", (self.size, self.size), self.fill)  # size x size 빈 도화지
        left = (self.size - new_w) // 2                   # 가운데에 놓기 위한 왼쪽 여백
        top = (self.size - new_h) // 2                    # 가운데에 놓기 위한 위쪽 여백
        canvas.paste(img, (left, top))
        return canvas

    def __repr__(self):
        return f"ResizePad(size={self.size}, fill={self.fill})"


# Validation / Test용 고정 전처리 (무작위 요소 없음, 규칙 7)
eval_transform = transforms.Compose([
    ResizePad(IMG_SIZE),
    transforms.ToTensor(),              # 0~255 픽셀 → 0~1 실수, 모양은 (채널, 세로, 가로)
    transforms.Normalize(MEAN, STD),    # (값 - 평균) / 표준편차 → ImageNet 기준에 맞춤
])

# Train용: baseline은 증강 없음 → eval과 동일
# 나중에 개선 실험에서 증강을 넣을 때는 노트북에서 train_transform만 새로 만들어 쓰면 됩니다.
train_transform = eval_transform


class ActionDataset(Dataset):
    """표의 i번째 행을 주면 (전처리된 사진, 정답 번호) 한 쌍을 돌려준다. 입력은 픽셀뿐(규칙 4)."""

    def __init__(self, frame, transform):
        self.rows = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows.iloc[idx]
        with Image.open(DATA_DIR / row["image_path"]) as img:
            x = self.transform(img.convert("RGB"))   # 흑백·CMYK 사진도 RGB 3채널로 통일
        return x, int(row["target_id"])              # (사진 텐서, 정답 번호)
