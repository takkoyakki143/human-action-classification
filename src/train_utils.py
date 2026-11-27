"""
src/train_utils.py
03_baseline과 04_modeling이 함께 쓰는 학습·평가 함수 모음.

[원칙] 03_baseline 노트북의 함수를 '옮기기만' 했다.
학습 결과에 영향을 주는 계산(모델 생성, 레이어 얼리기, BN 고정, 옵티마이저, 학습 루프,
Early Stopping, 난수 사용 순서)은 03과 한 줄도 다르지 않게 유지했다.

03에 없던 것으로 추가한 기능은 모두 '학습에 영향을 주지 않는' 기록·안전장치뿐이다.
  (1) get_config      : 기준선 조건을 복사해 바꿀 항목만 고치고, 바꾼 내용을 자동 기록
  (2) 예측 저장        : 실행마다 Val 이미지별 예측을 CSV로 저장 (오분류 분석용)
  (3) 덮어쓰기 방지    : 같은 이름의 결과가 있으면 학습을 시작하지 않고 멈춤
  (4) 전처리 자동 기록 : 실제로 쓴 Train/Val 전처리 내용을 summary에 함께 저장
"""
import copy
import json
import pickle
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from sklearn.metrics import f1_score

from src.common import BATCH_SIZE, IMG_SIZE, WEIGHTS, MODEL_DIR, RESULT_DIR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # GPU가 있으면 GPU 사용
GPU_NAME = torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "CPU"


# ════════════════════════════════════════════════════════════
# 1. 실험 조건 (03_baseline의 COMMON, EXPERIMENTS와 같은 값)
# ════════════════════════════════════════════════════════════
BASELINE_COMMON = {
    "loss": "CrossEntropyLoss",
    "optimizer": "AdamW",
    "weight_decay": 1e-4,          # 가중치가 너무 커지지 않게 살짝 눌러주는 값 (과적합 완화)
    "scheduler": "없음 (고정 학습률)",
    "monitor": "val_macro_f1",     # 최고 모델을 고르는 기준
    "patience": 15,                # 15 epoch 동안 최고점 갱신이 없으면 학습 중단
    "batch_size": BATCH_SIZE,
    "img_size": IMG_SIZE,          # 기록용. 실제 크기는 Dataset의 전처리가 정함 (run_experiment에서 일치 여부 검사)
    "train_augmentation": "없음 (baseline)",   # 기록용. 실제 증강은 train_set의 전처리가 정함
}

BASELINE_EXPERIMENTS = {
    "scratch": {
        "pretrained": False,                  # 무작위 가중치에서 시작
        "train_scope": "all",                 # 전체 레이어 학습
        "lr": {"all": 1e-3},
        "max_epochs": 100,                    # 처음부터 배우므로 수렴이 느려 epoch를 더 줌
    },
    "feature_extraction": {
        "pretrained": True,                   # ImageNet 가중치에서 시작
        "train_scope": "fc",                  # 마지막 분류기(fc)만 학습
        "lr": {"fc": 1e-3},
        "max_epochs": 30,
    },
    "fine_tuning": {
        "pretrained": True,
        "train_scope": "layer4+fc",           # 마지막 블록(layer4)과 분류기만 학습
        "lr": {"layer4": 1e-4, "fc": 1e-3},   # 이미 배운 백본은 조금만, 새 분류기는 크게
        "max_epochs": 30,
    },
}

SEEDS = [42, 43, 44]   # 여러 번 반복해 점수의 흔들림을 확인할 Seed 목록

# 코드가 실제로 읽지 않는 '설명용' 항목. 여기 값만 바꾸면 기록과 실제가 어긋나므로 get_config에서 막는다.
DESCRIPTION_ONLY = {"loss", "optimizer", "scheduler", "monitor"}


def get_config(base, name=None, **changes):
    """기준선 방식(base)의 조건을 복사하고, 바꿀 항목(changes)만 덮어쓴 조건 묶음을 돌려준다.

    예) 기준선 그대로      : get_config("fine_tuning")
        조건을 바꾼 실험   : get_config("fine_tuning", name="ft_wd1e-2", weight_decay=1e-2)
    바꾼 항목은 summary의 changed_from_baseline에 자동으로 남는다 (규칙 2: 달라진 조건 기록).
    """
    if base not in BASELINE_EXPERIMENTS:
        raise KeyError(f"기준선 방식 이름이 아닙니다: {base} (가능: {list(BASELINE_EXPERIMENTS)})")
    cfg = copy.deepcopy({**BASELINE_COMMON, **BASELINE_EXPERIMENTS[base]})

    unknown = [k for k in changes if k not in cfg]           # 오타 방지: 없는 항목은 조용히 무시되지 않게 멈춤
    if unknown:
        raise KeyError(f"기준선에 없는 항목입니다: {unknown}. 오타인지 확인하고, "
                       f"새 항목이 필요하면 train_utils.py에 기능을 먼저 추가해야 합니다.")
    desc_only = [k for k in changes if k in DESCRIPTION_ONLY]
    if desc_only:
        raise ValueError(f"{desc_only}는 코드가 읽지 않는 설명용 값입니다. "
                         f"바꾸려면 train_utils.py의 해당 코드를 먼저 고쳐야 합니다.")
    if changes and (name is None or name in BASELINE_EXPERIMENTS):
        raise ValueError("조건을 바꾼 실험에는 기준선과 다른 새 이름(name)을 붙여야 합니다. "
                         "같은 이름이면 기준선 결과와 섞이거나 덮어쓸 수 있습니다.")

    cfg.update(copy.deepcopy(changes))
    cfg["experiment"] = name or base                 # 결과 파일 이름에 쓰임
    cfg["base_experiment"] = base                    # 어느 기준선에서 출발했는지
    cfg["changed_from_baseline"] = copy.deepcopy(changes)
    return cfg


# ════════════════════════════════════════════════════════════
# 2. 재현성, 모델, 옵티마이저 (03의 04번 셀)
# ════════════════════════════════════════════════════════════
def set_seed(seed):
    """무작위 결과를 재현하기 위해 모든 난수 생성기를 같은 값으로 맞춘다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True   # GPU 연산 방식을 고정 (조금 느려질 수 있음)
    torch.backends.cudnn.benchmark = False       # 실행마다 연산 방식을 바꾸는 자동 최적화 끄기


def build_model(cfg, num_classes):
    """실험 조건(cfg)에 맞는 ResNet18을 만든다."""
    model = resnet18(weights=WEIGHTS if cfg["pretrained"] else None)

    if cfg["train_scope"] != "all":                 # FE, FT: 일단 전부 얼림
        for p in model.parameters():
            p.requires_grad = False
    if cfg["train_scope"] == "layer4+fc":            # FT: layer4만 다시 풀어 줌
        for p in model.layer4.parameters():
            p.requires_grad = True

    # 원래 fc(1000개 ImageNet 클래스)를 8개 클래스용 새 fc로 교체. 새로 만든 층은 기본적으로 학습 가능
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(DEVICE)


def freeze_bn_of_frozen_layers(model):
    """가중치가 얼려진 BatchNorm을 평가 모드로 돌려 running mean/var가 바뀌지 않게 한다."""
    n_frozen = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) and not any(p.requires_grad for p in m.parameters()):
            m.eval()
            n_frozen += 1
    return n_frozen


def make_optimizer(model, cfg):
    """학습할 부분에만 옵티마이저를 연결한다. FT는 layer4와 fc의 학습률을 다르게 준다."""
    lr = cfg["lr"]
    if cfg["train_scope"] == "all":
        groups = [{"params": model.parameters(), "lr": lr["all"]}]
    elif cfg["train_scope"] == "fc":
        groups = [{"params": model.fc.parameters(), "lr": lr["fc"]}]
    elif cfg["train_scope"] == "layer4+fc":
        groups = [
            {"params": model.layer4.parameters(), "lr": lr["layer4"]},
            {"params": model.fc.parameters(), "lr": lr["fc"]},
        ]
    else:   # 03에는 없던 방어 코드: 오타 난 범위로 엉뚱하게 학습되는 것을 막음 (결과에는 영향 없음)
        raise ValueError(f"알 수 없는 train_scope: {cfg['train_scope']}")
    return torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ════════════════════════════════════════════════════════════
# 3. DataLoader (03의 03번 셀)
# ════════════════════════════════════════════════════════════
def make_train_loader(train_set, seed, batch_size):
    """Seed에 따라 섞는 순서가 정해지는 Train DataLoader를 만든다."""
    g = torch.Generator().manual_seed(seed)
    return DataLoader(train_set, batch_size=batch_size, shuffle=True, generator=g, num_workers=0)


def make_eval_loader(dataset):
    """Val/Test용 DataLoader. 순서를 섞지 않아 예측을 df의 행 순서와 그대로 맞춰 볼 수 있다.
    채점 배치 크기는 학습 배치 크기를 바꾸는 실험에서도 03과 같게 BATCH_SIZE로 고정한다."""
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


# ════════════════════════════════════════════════════════════
# 4. 학습·평가 (03의 05번 셀)
# ════════════════════════════════════════════════════════════
loss_fn = nn.CrossEntropyLoss()


def train_one_epoch(model, loader, optimizer):
    model.train()                       # 학습 모드
    freeze_bn_of_frozen_layers(model)   # 얼린 레이어의 BN은 다시 평가 모드로 (기록장 보호)
    total_loss, correct, n = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)   # 직전 배치의 기울기 비우기
        logits = model(images)                  # 사진마다 8개 클래스 점수 계산
        loss = loss_fn(logits, labels)          # 정답과 얼마나 어긋났는지
        loss.backward()                         # 기울기 계산
        optimizer.step()                        # 가중치 수정
        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        n += len(labels)
    return total_loss / n, correct / n


@torch.inference_mode()                 # 채점 중에는 기울기를 계산하지 않음 (빠르고 메모리 절약)
def evaluate(model, loader):
    model.eval()                        # 모든 BN이 기록장 값을 사용
    num_classes = model.fc.out_features
    total_loss, y_true, y_pred, probs = 0.0, [], [], []
    for images, labels in loader:
        logits = model(images.to(DEVICE))
        total_loss += loss_fn(logits, labels.to(DEVICE)).item() * len(labels)
        p = logits.softmax(dim=1).cpu()          # 점수를 확률(합이 1)로 변환
        probs.append(p)
        y_true.extend(labels.tolist())
        y_pred.extend(p.argmax(dim=1).tolist())
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return {
        "loss": total_loss / len(y_true),
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro",
                                   labels=list(range(num_classes)), zero_division=0)),
        "y_true": y_true, "y_pred": y_pred, "probs": torch.cat(probs).numpy(),
    }


def predictions_frame(result, df, class_names, experiment, seed):
    """evaluate 결과를 '사진 한 장 = 한 줄' 표로 만든다. (03의 12번 셀 표 + 클래스별 확률)"""
    if not np.array_equal(df["target_id"].to_numpy(), result["y_true"]):
        raise ValueError("df의 행 순서가 DataLoader 순서와 다릅니다. shuffle=False로 만든 loader와 같은 df를 넣으세요.")
    frame = df[["image_id", "image_path", "target_id", "class_name"]].copy().reset_index(drop=True)
    frame["experiment"] = experiment
    frame["seed"] = seed
    frame["pred_id"] = result["y_pred"]
    frame["pred_name"] = [class_names[i] for i in result["y_pred"]]
    frame["confidence"] = result["probs"].max(axis=1).round(4)                              # 예측한 클래스의 확률
    frame["true_prob"] = result["probs"][np.arange(len(frame)), frame["target_id"]].round(4)  # 정답 클래스의 확률
    frame["correct"] = frame["pred_id"] == frame["target_id"]
    for i, c in enumerate(class_names):                                                     # 클래스별 확률 전체
        frame[f"prob_{c}"] = result["probs"][:, i].round(4)
    return frame


def _check_img_size(val_set, cfg):
    """기록된 img_size와 실제 Val 전처리 결과 크기가 같은지 확인한다 (기록-실제 불일치 방지)."""
    h, w = val_set[0][0].shape[-2:]
    if (h, w) != (cfg["img_size"], cfg["img_size"]):
        raise ValueError(f"cfg의 img_size({cfg['img_size']})와 실제 Val 이미지 크기({h}×{w})가 다릅니다. "
                         f"get_config에서 img_size를 바꿨다면 Dataset의 전처리도 같이 바꿔야 합니다.")


def run_experiment(cfg, seed, train_set, val_set, val_df, class_names,
                   save=True, overwrite=False, verbose=True):
    """한 조건(cfg)을 한 Seed로 학습하고, 최고 시점 모델·기록·요약·Val 예측을 저장해 돌려준다.

    save=False   : 파일을 하나도 저장하지 않음 (재현성 확인용)
    overwrite    : 같은 이름의 결과가 이미 있을 때 덮어쓸지. 기본값 False (기준선 보호)
    """
    name = cfg["experiment"]
    tag = f"{name}_seed{seed}"
    if save and not overwrite and (RESULT_DIR / f"summary_{tag}.json").exists():
        raise FileExistsError(f"{tag} 결과가 이미 있습니다. 다른 이름을 쓰거나, 정말 다시 돌릴 때만 overwrite=True로 실행하세요.")
    _check_img_size(val_set, cfg)
    val_loader = make_eval_loader(val_set)

    # ── 여기부터 03과 같은 순서: Seed 고정 → 모델 → 옵티마이저 → Train loader ──
    set_seed(seed)                       # 모델 초기값·fc 초기값·섞는 순서를 모두 Seed로 고정
    model = build_model(cfg, len(class_names))
    optimizer = make_optimizer(model, cfg)
    train_loader = make_train_loader(train_set, seed, cfg["batch_size"])
    total_params, trainable_params = count_params(model)

    history, best_f1, best_epoch, best_state, wait = [], -1.0, 0, None, 0
    run_start = time.perf_counter()
    for epoch in range(1, cfg["max_epochs"] + 1):
        t0 = time.perf_counter()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()     # GPU 작업이 끝날 때까지 기다린 뒤 시간 측정
        train_sec = time.perf_counter() - t0
        val = evaluate(model, val_loader)
        epoch_sec = time.perf_counter() - t0

        improved = val["macro_f1"] > best_f1          # 동점은 갱신으로 보지 않음
        if improved:
            best_f1, best_epoch, wait = val["macro_f1"], epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1

        history.append({"epoch": epoch, "train_loss": tr_loss, "train_accuracy": tr_acc,
                        "val_loss": val["loss"], "val_accuracy": val["accuracy"],
                        "val_macro_f1": val["macro_f1"], "train_sec": train_sec, "epoch_sec": epoch_sec})
        if verbose:
            print(f"[{name} | seed {seed}] epoch {epoch:3d} | "
                  f"train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
                  f"val loss {val['loss']:.3f} acc {val['accuracy']:.3f} F1 {val['macro_f1']:.3f} "
                  f"| {epoch_sec:.1f}s {'★' if improved else ''}")
        if wait >= cfg["patience"]:
            if verbose:
                print(f"→ {cfg['patience']} epoch 동안 갱신 없음: epoch {epoch}에서 중단")
            break
    total_sec = time.perf_counter() - run_start

    model.load_state_dict(best_state)                  # 최고 시점 가중치로 되돌림
    hist = pd.DataFrame(history)
    best_row = hist.loc[hist["epoch"] == best_epoch].iloc[0]
    summary = {
        "experiment": name, "seed": seed, "device": GPU_NAME,
        "base_experiment": cfg["base_experiment"],
        "changed_from_baseline": json.dumps(cfg["changed_from_baseline"], ensure_ascii=False, default=str),
        "pretrained": cfg["pretrained"], "train_scope": cfg["train_scope"],
        "lr": json.dumps(cfg["lr"]), "max_epochs": cfg["max_epochs"],
        **{k: cfg[k] for k in BASELINE_COMMON},
        "train_transform": repr(getattr(train_set, "transform", "확인 불가")),   # 실제로 쓴 전처리
        "val_transform": repr(getattr(val_set, "transform", "확인 불가")),
        "total_params": total_params, "trainable_params": trainable_params,
        "best_epoch": best_epoch, "stopped_epoch": int(hist["epoch"].max()),
        "early_stopped": wait >= cfg["patience"],
        "total_time_sec": round(total_sec, 1),                      # 학습+검증 포함, 이 1회 실행 시간
        "mean_epoch_sec": round(float(hist["epoch_sec"].mean()), 2),   # float(): numpy 숫자로 저장되면 가중치 파일을 안전 모드로 못 읽음
        "best_val_macro_f1": round(best_f1, 4),
        "best_val_accuracy": round(float(best_row["val_accuracy"]), 4),
        "best_val_loss": round(float(best_row["val_loss"]), 4),
        "best_train_accuracy": round(float(best_row["train_accuracy"]), 4),
    }

    if save:
        # 기록·요약·예측은 results/(Git에 올라감), 가중치는 models/(.gitignore로 제외)
        hist.to_csv(RESULT_DIR / f"history_{tag}.csv", index=False, encoding="utf-8-sig")
        with open(RESULT_DIR / f"summary_{tag}.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        torch.save({"state_dict": best_state, "summary": summary, "class_names": [str(c) for c in class_names]},
                   MODEL_DIR / f"{tag}.pt")
        # 학습이 끝난 뒤 최고 시점 모델로 Val을 한 번 더 채점해 이미지별 예측 저장 (학습에는 영향 없음)
        final = evaluate(model, val_loader)
        if abs(final["macro_f1"] - best_f1) > 1e-6:
            print(f"⚠ 저장한 모델의 Val F1({final['macro_f1']:.4f})이 학습 중 최고값({best_f1:.4f})과 다릅니다. 확인 필요")
        predictions_frame(final, val_df, class_names, name, seed).to_csv(
            RESULT_DIR / f"val_predictions_{tag}.csv", index=False, encoding="utf-8-sig")
    if verbose:
        print(f"✔ 최고 Val Macro F1 {best_f1:.4f} (epoch {best_epoch}) | 총 {total_sec / 60:.1f}분"
              f" | {'저장: ' + tag if save else '저장 안 함'}")
    return model, hist, summary


def run_or_load(cfg, seeds, train_set, val_set, val_df, class_names):
    """여러 Seed를 차례로 실행한다. 이미 저장된 결과가 있으면 다시 학습하지 않고 불러온다. (03의 10번 셀)"""
    summaries = {}
    for seed in seeds:
        tag = f"{cfg['experiment']}_seed{seed}"
        if (RESULT_DIR / f"summary_{tag}.json").exists():
            _, _, summaries[seed] = load_run(cfg["experiment"], seed, len(class_names), with_model=False)
            print(f"[{tag}] 저장된 결과를 불러왔습니다.")
            continue
        model, _, summaries[seed] = run_experiment(cfg, seed, train_set, val_set, val_df, class_names, verbose=False)
        print(f"[{tag}] Val Macro F1 {summaries[seed]['best_val_macro_f1']:.4f} "
              f"(epoch {summaries[seed]['best_epoch']}) | {summaries[seed]['total_time_sec'] / 60:.1f}분")
        del model                        # 메모리 절약: 모델은 파일로만 보관
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    return summaries


# ════════════════════════════════════════════════════════════
# 5. 저장된 결과 불러오기와 점검·그림 (03의 05·07·09번 셀)
# ════════════════════════════════════════════════════════════
def load_checkpoint(path):
    """우리가 직접 저장한 가중치 파일(.pt)을 읽는다.

    먼저 안전 모드(weights_only=True)로 읽고, 03에서 저장한 파일처럼 summary 안에 numpy 숫자가
    섞여 있어 실패하면 일반 모드로 다시 읽는다. 일반 모드는 남이 만든 파일에서는 위험할 수 있으므로,
    이 함수는 우리 프로젝트의 models/ 폴더에 직접 저장한 파일에만 쓴다.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        print(f"[안내] {Path(path).name}: 안전 모드로 읽을 수 없어 일반 모드로 읽습니다 (직접 저장한 파일이라 괜찮음).")
        return torch.load(path, map_location="cpu", weights_only=False)


def load_run(name, seed, num_classes, with_model=True):
    """저장된 실험 결과를 다시 불러온다. (커널을 재시작해도 다시 학습하지 않아도 됨)"""
    tag = f"{name}_seed{seed}"
    hist = pd.read_csv(RESULT_DIR / f"history_{tag}.csv", encoding="utf-8-sig")
    with open(RESULT_DIR / f"summary_{tag}.json", encoding="utf-8") as f:
        summary = json.load(f)
    model = None
    if with_model:
        ckpt = load_checkpoint(MODEL_DIR / f"{tag}.pt")
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(ckpt["state_dict"])
        model = model.to(DEVICE)
    return model, hist, summary


def predict_saved_run(name, seed, val_set, val_df, class_names, save=True):
    """저장된 가중치를 불러와 Val 이미지별 예측표를 만든다. 03처럼 예측을 다 저장하지 않았던 실행에 쓴다."""
    model, _, _ = load_run(name, seed, len(class_names))
    frame = predictions_frame(evaluate(model, make_eval_loader(val_set)), val_df, class_names, name, seed)
    if save:
        frame.to_csv(RESULT_DIR / f"val_predictions_{name}_seed{seed}.csv", index=False, encoding="utf-8-sig")
    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return frame


def check_frozen(model, trainable_prefixes):
    """ImageNet 원본과 비교해, 학습 대상이 아닌 값이 하나도 바뀌지 않았는지 확인한다."""
    ref = resnet18(weights=WEIGHTS).state_dict()        # 원래 ImageNet 값
    now = {k: v.cpu() for k, v in model.state_dict().items()}
    frozen_keys = [k for k in ref if not k.startswith(trainable_prefixes)]
    changed = [k for k in frozen_keys if not torch.equal(ref[k], now[k])]
    trained_keys = [k for k in ref if k.startswith(trainable_prefixes) and k in now
                    and ref[k].shape == now[k].shape]
    trained_changed = [k for k in trained_keys if not torch.equal(ref[k], now[k])]
    print(f"고정 대상 {len(frozen_keys)}개 중 바뀐 값: {len(changed)}개", "✔" if not changed else "✘ 확인 필요")
    if changed:
        print("  바뀐 항목 예시:", changed[:5])
    if trained_keys:
        print(f"학습 대상 {len(trained_keys)}개 중 바뀐 값: {len(trained_changed)}개 (학습이 실제로 일어났는지)")


def plot_history(hist, summary, axes):
    """한 실험의 학습 곡선(손실, 점수)을 두 칸에 그린다."""
    best = summary["best_epoch"]
    axes[0].plot(hist["epoch"], hist["train_loss"], label="train loss")
    axes[0].plot(hist["epoch"], hist["val_loss"], label="val loss")
    axes[1].plot(hist["epoch"], hist["train_accuracy"], label="train acc", alpha=0.6)
    axes[1].plot(hist["epoch"], hist["val_accuracy"], label="val acc", alpha=0.6)
    axes[1].plot(hist["epoch"], hist["val_macro_f1"], label="val macro F1", linewidth=2)
    for ax in axes:
        ax.axvline(best, color="gray", linestyle="--", linewidth=1)   # 최고 시점 표시
        ax.set_xlabel("epoch")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    axes[0].set_title(f"{summary['experiment']} (seed {summary['seed']}) - loss", fontsize=10)
    axes[1].set_title(f"best val macro F1 {summary['best_val_macro_f1']:.3f} @ epoch {best}", fontsize=10)
    axes[1].set_ylim(0, 1.02)


def comparison_table(summaries):
    """여러 실행의 요약을 비교표 한 장으로 만든다. (03 결과 파일도 그대로 읽을 수 있음)"""
    rows = []
    for s in summaries:
        rows.append({
            "방식": s["experiment"], "seed": s["seed"],
            "출발 기준선": s.get("base_experiment", s["experiment"]),       # 03 결과에는 이 항목이 없음
            "바뀐 조건": s.get("changed_from_baseline", "{}"),
            "학습 범위": s["train_scope"], "학습률": s["lr"],
            "전체 파라미터": f"{s['total_params']:,}", "학습 가능 파라미터": f"{s['trainable_params']:,}",
            "최대 epoch": s["max_epochs"], "최고 epoch": s["best_epoch"], "중단 epoch": s["stopped_epoch"],
            "총 시간(분)": round(s["total_time_sec"] / 60, 2), "epoch당(초)": s["mean_epoch_sec"],
            "Val Macro F1": s["best_val_macro_f1"], "Val Accuracy": s["best_val_accuracy"],
            "Train Acc(최고 시점)": s["best_train_accuracy"],
        })
    return pd.DataFrame(rows)
