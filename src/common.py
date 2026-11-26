# src/common.py — 세 노트북이 같이 쓰는 경로·데이터 로딩 코드
from pathlib import Path
import pandas as pd

# 이 파일(src/) 기준 한 단계 위 = 프로젝트 최상위 폴더
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"
FIGURE_DIR = ROOT / "figures"


def load_metadata():
    # utf-8-sig: CSV 맨 앞의 숨은 문자(BOM) 때문에 첫 컬럼명이 깨지는 것을 방지
    df = pd.read_csv(DATA_DIR / "metadata.csv", encoding="utf-8-sig")
    df["full_path"] = df["image_path"].apply(lambda p: DATA_DIR / p)  # 실제 이미지 경로
    return df