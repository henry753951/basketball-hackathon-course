from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_angle_series(
    df: pd.DataFrame, columns: list[str], output_path: str | Path | None = None
):
    plt.figure(figsize=(10, 4))
    x = df["frame"] if "frame" in df.columns else range(len(df))
    for col in columns:
        if col in df.columns:
            plt.plot(x, df[col], label=col)
    plt.xlabel("frame")
    plt.ylabel("angle / value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.show()


def plot_ball_path(df: pd.DataFrame, output_path: str | Path | None = None):
    plt.figure(figsize=(7, 5))
    if {"x", "y"}.issubset(df.columns):
        plt.plot(df["x"], df["y"], marker="o")
        plt.gca().invert_yaxis()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Ball path")
    plt.grid(True, alpha=0.3)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.show()
