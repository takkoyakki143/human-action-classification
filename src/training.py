import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score, f1_score

from .preprocessing import NUM_CLASSES


# ==================================================
# 1. Train 함수
# ==================================================

def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    freeze_backbone_bn=False
):
    model.train()

    # Feature Extraction 시 백본 BatchNorm 통계 고정
    if freeze_backbone_bn:
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in dataloader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        preds = logits.argmax(dim=1)

        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return {
        "loss": running_loss / len(all_labels),

        "accuracy": accuracy_score(
            all_labels, all_preds
        ),

        "f1": f1_score(
            all_labels,
            all_preds,
            labels=list(range(NUM_CLASSES)),
            average="macro",
            zero_division=0
        )
    }


# ==================================================
# 2. Validation 함수
# ==================================================

def validate_one_epoch(
    model,
    dataloader,
    criterion,
    device
):
    model.eval()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)

            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    return {
        "loss": running_loss / len(all_labels),

        "accuracy": accuracy_score(
            all_labels, all_preds
        ),

        "f1": f1_score(
            all_labels,
            all_preds,
            labels=list(range(NUM_CLASSES)),
            average="macro",
            zero_division=0
        )
    }


# ==================================================
# 3. 전체 Epoch 학습 및 Checkpoint 저장
# ==================================================

def fit(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs,
    checkpoint_path,
    freeze_backbone_bn=False
):
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    history = []

    best_val_f1 = -float("inf")
    best_epoch = 0

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    total_start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):

        epoch_start_time = time.perf_counter()

        # Train
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            freeze_backbone_bn=freeze_backbone_bn
        )

        # Validation
        val_metrics = validate_one_epoch(
            model,
            val_loader,
            criterion,
            device
        )

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        epoch_time = (
            time.perf_counter() - epoch_start_time
        )

        # Epoch별 결과 기록
        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_f1": train_metrics["f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
            "epoch_time_sec": epoch_time
        })

        # 최적 Checkpoint 저장
        is_best = val_metrics["f1"] > best_val_f1

        if is_best:

            best_val_f1 = val_metrics["f1"]
            best_epoch = epoch

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_f1": best_val_f1,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"]
            }, checkpoint_path)

        # 학습 진행 상황 출력
        print(
            f"Epoch [{epoch:02d}/{epochs}] | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Train Acc: {train_metrics['accuracy']:.4f} | "
            f"Train F1: {train_metrics['f1']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f} | "
            f"Time: {epoch_time:.2f}s"
            + (" | Best Model Saved" if is_best else "")
        )

    total_training_time = (
        time.perf_counter() - total_start_time
    )

    history_df = pd.DataFrame(history)

    print("\n===== Training Completed =====")
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Validation Macro F1: {best_val_f1:.4f}")
    print(f"Total Training Time: {total_training_time:.2f}s")
    print(f"Total Training Time: {total_training_time / 60:.2f}min")
    print(f"Checkpoint Saved: {checkpoint_path}")

    return history_df, total_training_time


# ==================================================
# 4. 학습 기록 및 실험 요약 저장
# ==================================================

def save_training_results(
    history_df,
    total_training_time,
    experiment_config,
    model,
    device,
    result_dir
):
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    experiment_name = experiment_config["experiment_name"]

    # Epoch별 학습 기록 저장
    history_path = (
        result_dir / f"{experiment_name}_history.csv"
    )

    history_df.to_csv(history_path, index=False)

    # 최적 Epoch 추출
    best_result = history_df.loc[
        history_df["val_f1"].idxmax()
    ]

    # 파라미터 수 계산
    total_params = sum(
        p.numel() for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    # 실험 설정 및 주요 결과 요약
    summary = {
        **experiment_config,

        "total_params": total_params,
        "trainable_params": trainable_params,

        "best_epoch": int(best_result["epoch"]),
        "best_train_loss": best_result["train_loss"],
        "best_train_accuracy": best_result["train_accuracy"],
        "best_train_f1": best_result["train_f1"],

        "best_val_loss": best_result["val_loss"],
        "best_val_accuracy": best_result["val_accuracy"],
        "best_val_f1": best_result["val_f1"],

        "mean_epoch_time_sec": (
            history_df["epoch_time_sec"].mean()
        ),

        "total_training_time_sec": total_training_time,

        "device": str(device),

        "gpu_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else "None"
        ),

        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda
    }

    summary_df = pd.DataFrame([summary])

    summary_path = (
        result_dir / f"{experiment_name}_summary.csv"
    )

    summary_df.to_csv(summary_path, index=False)

    print("History Saved:", history_path)
    print("Summary Saved:", summary_path)

    return summary_df