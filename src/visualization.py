from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from .evaluation import CLASS_NAMES


# ==================================================
# 1. 학습 곡선 시각화
# ==================================================

def plot_learning_curves(
    history_df,
    experiment_name,
    figure_dir
):
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    best_epoch = history_df.loc[
        history_df["val_f1"].idxmax(),
        "epoch"
    ]

    fig, axes = plt.subplots(
        1, 3,
        figsize=(18, 5)
    )

    metric_settings = [
        ("loss", "Loss"),
        ("accuracy", "Accuracy"),
        ("f1", "Macro F1")
    ]

    for ax, (metric, title) in zip(
        axes, metric_settings
    ):

        ax.plot(
            history_df["epoch"],
            history_df[f"train_{metric}"],
            label="Train"
        )

        ax.plot(
            history_df["epoch"],
            history_df[f"val_{metric}"],
            label="Validation"
        )

        ax.axvline(
            x=best_epoch,
            color="gray",
            linestyle="--",
            label=f"Best Epoch ({best_epoch})"
        )

        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)

        if metric != "loss":
            ax.set_ylim(0, 1.05)
        else:
            ax.set_ylim(bottom=0)

        ax.grid(alpha=0.3)
        ax.legend()

    fig.suptitle(
        f"{experiment_name} - Learning Curves",
        fontsize=14
    )

    fig.tight_layout()

    save_path = (
        figure_dir /
        f"{experiment_name}_learning_curves.png"
    )

    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)

    print("Learning Curves Saved:", save_path)


# ==================================================
# 2. 클래스별 성능 시각화
# ==================================================

def plot_class_metrics(
    class_report_df,
    macro_f1,
    experiment_name,
    figure_dir
):
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    ax = class_report_df[
        ["precision", "recall", "f1-score"]
    ].plot(
        kind="bar",
        figsize=(13, 6),
        width=0.8
    )

    ax.set_title(
        f"{experiment_name} - Class-wise Performance"
    )

    ax.set_xlabel("Action Class")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)

    ax.axhline(
        y=macro_f1,
        color="gray",
        linestyle="--",
        label=f"Macro F1 ({macro_f1:.4f})"
    )

    plt.xticks(rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    fig = ax.get_figure()
    fig.tight_layout()

    save_path = (
        figure_dir /
        f"{experiment_name}_class_metrics.png"
    )

    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)

    print("Class Metrics Saved:", save_path)


# ==================================================
# 3. Confusion Matrix 시각화
# ==================================================

def plot_confusion_matrix(
    cm_df,
    experiment_name,
    figure_dir,
    normalize=False,
    vmax=20
):
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    # 정규화 여부 설정
    if normalize:

        plot_df = cm_df.div(
            cm_df.sum(axis=1).replace(0, 1),
            axis=0
        )

        fmt = ".1%"
        color_max = 1
        suffix = "confusion_matrix_normalized"

    else:

        plot_df = cm_df
        fmt = "d"
        color_max = vmax
        suffix = "confusion_matrix"

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    sns.heatmap(
        plot_df,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        vmin=0,
        vmax=color_max,
        square=True,
        ax=ax
    )

    ax.set_title(
        f"{experiment_name} - Confusion Matrix"
    )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    fig.tight_layout()

    save_path = (
        figure_dir /
        f"{experiment_name}_{suffix}.png"
    )

    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)

    print("Confusion Matrix Saved:", save_path)
    