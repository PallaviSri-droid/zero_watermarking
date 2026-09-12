from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def save_summary_bars(df: pd.DataFrame, out_dir: str = "figures") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("mean_nc", "NC (higher is better)", False),
        ("mean_intra_hd", "Intra-image Hamming distance (lower is better)", True),
        ("mean_inter_hd", "Inter-image Hamming distance (higher is better)", False),
        ("collision_gap", "Collision gap (higher is better)", False),
        ("auc", "ROC-AUC (higher is better)", False),
        ("eer", "EER (lower is better)", True),
    ]
    for column, title, ascending in metrics:
        figure, axis = plt.subplots(figsize=(9, 5))
        data = df.sort_values(column, ascending=ascending)
        sns.barplot(data=data, x=column, y="method", ax=axis)
        axis.set_title(title)
        axis.set_ylabel("")
        figure.tight_layout()
        figure.savefig(out / f"{column}.png", dpi=220)
        plt.close(figure)


def plot_pareto(df: pd.DataFrame, out_dir: str = "figures") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.scatter(df["mean_intra_hd"], df["mean_inter_hd"], s=70)
    for _, row in df.iterrows():
        axis.annotate(
            row["method"],
            (row["mean_intra_hd"], row["mean_inter_hd"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    axis.set_xlabel("Mean intra-image HD ↓")
    axis.set_ylabel("Mean inter-image HD ↑")
    axis.set_title("Robustness–discriminability plane")
    figure.tight_layout()
    figure.savefig(out / "pareto_robustness_discrimination.png", dpi=220)
    plt.close(figure)


def heatmap_attack(details_df: pd.DataFrame, out_dir: str = "figures") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pivot = details_df.pivot_table(index="method", columns="attack", values="hd", aggfunc="mean")
    figure, axis = plt.subplots(figsize=(10, 5))
    sns.heatmap(pivot, annot=True, fmt=".3f", ax=axis, cbar_kws={"label": "Hamming distance"})
    axis.set_title("Attack robustness heatmap")
    figure.tight_layout()
    figure.savefig(out / "attack_heatmap.png", dpi=220)
    plt.close(figure)
