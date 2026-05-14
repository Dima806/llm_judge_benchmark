from __future__ import annotations

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def kappa_heatmap(
    kappa_matrix: pd.DataFrame,
    title: str = "Cohen's Kappa Heatmap",
) -> matplotlib.figure.Figure:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(kappa_matrix, annot=True, fmt=".2f", vmin=-1, vmax=1, center=0, ax=ax)
    ax.set_title(title)
    return fig


def bias_heatmap(
    bias_matrix: pd.DataFrame,
    title: str = "Systematic Bias (Judge \u2212 Human)",
) -> matplotlib.figure.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(bias_matrix, annot=True, fmt="+.3f", center=0, ax=ax)
    ax.set_title(title)
    return fig


def scatter_matrix(
    scores: dict[str, np.ndarray],
    metric: str,
    title: str | None = None,
) -> matplotlib.figure.Figure:
    df = pd.DataFrame(scores)
    n = len(df.columns)
    fig, axes = plt.subplots(n, n, figsize=(10, 10))
    for i, col_i in enumerate(df.columns):
        for j, col_j in enumerate(df.columns):
            ax = axes[i][j]  # type: ignore[index]
            if i == j:
                ax.hist(df[col_i], bins=10)
                ax.set_xlabel(col_i)
            else:
                ax.scatter(df[col_j], df[col_i], alpha=0.5, s=20)
                ax.set_xlabel(col_j)
                ax.set_ylabel(col_i)
    if title:
        fig.suptitle(f"{title} — {metric}")
    fig.tight_layout()
    return fig


def calibration_curve(
    raw_scores: np.ndarray,
    corrected_scores: np.ndarray,
    human_scores: np.ndarray,
    label: str = "Model",
) -> matplotlib.figure.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax0, ax1 = axes[0], axes[1]  # type: ignore[index]

    ax0.scatter(raw_scores, human_scores, alpha=0.6, label="Raw vs Human")
    ax0.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax0.set_xlabel("Raw Score")
    ax0.set_ylabel("Human Score")
    ax0.set_title(f"{label} \u2014 Before Calibration")
    ax0.legend()

    ax1.scatter(corrected_scores, human_scores, alpha=0.6, label="Corrected vs Human")
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax1.set_xlabel("Corrected Score")
    ax1.set_ylabel("Human Score")
    ax1.set_title(f"{label} \u2014 After Calibration")
    ax1.legend()

    fig.tight_layout()
    return fig
