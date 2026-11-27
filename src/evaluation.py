from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from .preprocessing import NUM_CLASSES


# 클래스 표시명 (target_id 순서)
CLASS_NAMES = [
    "Climbing",
    "Jumping",
    "Riding Bike",
    "Riding Horse",
    "Rowing Boat",
    "Running",
    "Shooting Arrow",
    "Throwing Frisbee"
]


# ==================================================
# 1. 최적 Checkpoint 불러오기
# ==================================================

def load_best_model(
    model,
    checkpoint_path,
    device
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    print("===== Best Checkpoint Loaded =====")
    print("Best Epoch:", checkpoint["epoch"])
    print(f"Validation Loss: {checkpoint['val_loss']:.4f}")
    print(f"Validation Accuracy: {checkpoint['val_accuracy']:.4f}")
    print(f"Validation Macro F1: {checkpoint['val_f1']:.4f}")

    return model, checkpoint


# ==================================================
# 2. 모델 평가 및 예측 결과 수집
# ==================================================

def evaluate_model(
    model,
    dataloader,
    criterion,
    device
):
    model.eval()

    running_loss = 0.0

    all_labels = []
    all_preds = []
    all_probs = []
    all_confidences = []

    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)

            loss = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)

            # 클래스별 예측 확률
            probs = torch.softmax(logits, dim=1)

            # 예측 클래스 및 Confidence
            confidences, preds = probs.max(dim=1)

            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
            all_confidences.extend(
                confidences.cpu().tolist()
            )

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)
    y_confidence = np.array(all_confidences)

    class_ids = list(range(NUM_CLASSES))

    # 전체 평가 지표
    metrics = {
        "loss": running_loss / len(y_true),

        "accuracy": accuracy_score(
            y_true, y_pred
        ),

        "macro_precision": precision_score(
            y_true,
            y_pred,
            labels=class_ids,
            average="macro",
            zero_division=0
        ),

        "macro_recall": recall_score(
            y_true,
            y_pred,
            labels=class_ids,
            average="macro",
            zero_division=0
        ),

        "macro_f1": f1_score(
            y_true,
            y_pred,
            labels=class_ids,
            average="macro",
            zero_division=0
        )
    }

    return {
        "metrics": metrics,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_probs": y_probs,
        "y_confidence": y_confidence
    }


# ==================================================
# 3. 클래스별 성능 및 혼동 유형 분석
# ==================================================

def analyze_class_performance(
    y_true,
    y_pred,
    class_names=CLASS_NAMES
):
    class_ids = list(range(NUM_CLASSES))

    # Classification Report
    report = classification_report(
        y_true,
        y_pred,
        labels=class_ids,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).T

    class_report_df = report_df.loc[
        class_names,
        ["precision", "recall", "f1-score", "support"]
    ].copy()

    # Confusion Matrix
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=class_ids
    )

    cm_df = pd.DataFrame(
        cm,
        index=class_names,
        columns=class_names
    )

    # 주요 혼동 유형
    confusion_records = []

    for true_id in class_ids:
        for pred_id in class_ids:

            if true_id == pred_id:
                continue

            if cm[true_id, pred_id] > 0:

                confusion_records.append({
                    "true_class": class_names[true_id],
                    "predicted_class": class_names[pred_id],
                    "count": cm[true_id, pred_id]
                })

    confusion_pairs_df = pd.DataFrame(
        confusion_records,
        columns=[
            "true_class",
            "predicted_class",
            "count"
        ]
    )

    confusion_pairs_df = (
        confusion_pairs_df
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "report_df": report_df,
        "class_report_df": class_report_df,
        "cm_df": cm_df,
        "confusion_pairs_df": confusion_pairs_df
    }


# ==================================================
# 4. 이미지별 예측 결과 저장
# ==================================================

def save_predictions(
    evaluation,
    metadata_df,
    dataloader,
    experiment_name,
    result_dir
):
    # 순차적으로 평가한 DataLoader만 허용
    if not isinstance(
        dataloader.sampler,
        torch.utils.data.SequentialSampler
    ):
        raise ValueError(
            "Prediction saving requires a sequential DataLoader."
        )

    if dataloader.drop_last:
        raise ValueError(
            "Prediction saving requires drop_last=False."
        )

    predictions_df = (
        metadata_df.copy().reset_index(drop=True)
    )

    y_true = evaluation["y_true"]
    y_pred = evaluation["y_pred"]
    y_probs = evaluation["y_probs"]
    y_confidence = evaluation["y_confidence"]

    # 이미지 수 일치 여부 확인
    if len(predictions_df) != len(y_true):
        raise ValueError(
            "Metadata and prediction lengths do not match."
        )

    # 정답 라벨 순서 확인
    if not np.array_equal(
        predictions_df["target_id"].to_numpy(),
        y_true
    ):
        raise ValueError(
            "Metadata and prediction labels do not match."
        )

    # 예측 결과 추가
    predictions_df["experiment_name"] = experiment_name

    predictions_df["y_true"] = y_true
    predictions_df["y_pred"] = y_pred

    predictions_df["predicted_class"] = [
        CLASS_NAMES[i] for i in y_pred
    ]

    predictions_df["confidence"] = y_confidence

    predictions_df["is_correct"] = (
        y_true == y_pred
    )

    # 클래스별 예측 확률 저장
    for class_id in range(NUM_CLASSES):

        predictions_df[f"prob_{class_id}"] = (
            y_probs[:, class_id]
        )

    # CSV 저장
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    prediction_path = (
        result_dir / f"{experiment_name}_predictions.csv"
    )

    predictions_df.to_csv(
        prediction_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("Predictions Saved:", prediction_path)

    return predictions_df


# ==================================================
# 5. 평가 결과 저장
# ==================================================

def save_evaluation_results(
    evaluation,
    class_analysis,
    experiment_name,
    epoch,
    result_dir,
    split="validation"
):
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    # 전체 평가 지표
    evaluation_df = pd.DataFrame([{
        "experiment_name": experiment_name,
        "split": split,
        "epoch": epoch,
        **evaluation["metrics"]
    }])

    evaluation_df.to_csv(
        result_dir / f"{experiment_name}_evaluation.csv",
        index=False
    )

    # 클래스별 평가 결과
    class_analysis["class_report_df"].to_csv(
        result_dir / f"{experiment_name}_class_report.csv",
        index_label="class_name"
    )

    # Confusion Matrix
    class_analysis["cm_df"].to_csv(
        result_dir / f"{experiment_name}_confusion_matrix.csv",
        index_label="true_class"
    )

    # 주요 혼동 유형
    class_analysis["confusion_pairs_df"].to_csv(
        result_dir / f"{experiment_name}_confusion_pairs.csv",
        index=False
    )

    print("Evaluation Results Saved")

    return evaluation_df