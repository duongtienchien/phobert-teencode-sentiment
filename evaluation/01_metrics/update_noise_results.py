"""
evaluation/01_metrics/update_noise_results.py
----------------------------------------------
Nhanh feat/evaluation (Ni 3 - Son)
Chay script nay sau khi Ni 2 (Hieu) export prediction files
cho cac tap: test_noise_10, test_noise_30, test_noise_50, test_khong_dau

Ni 2 can export cac file sau vao thu muc model_outputs/:
  baseline_noise10_predictions.csv
  baseline_noise30_predictions.csv
  baseline_noise50_predictions.csv
  baseline_khongdau_predictions.csv
  proposed_noise10_predictions.csv
  proposed_noise30_predictions.csv
  proposed_noise50_predictions.csv
  proposed_khongdau_predictions.csv

Moi file can co cac cot: true_label, predicted_label, confidence
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score

matplotlib.rcParams["font.family"] = ["DejaVu Sans"]

BASE = r"d:\Ky 7\Hoc may nang cao\phobert-teencode-sentiment"
PRED = fr"{BASE}\model_outputs"
OUT  = fr"{BASE}\evaluation\01_metrics"
os.makedirs(OUT, exist_ok=True)

# Ten file du doan cho tung tap nhieu (Ni 2 phai export)
NOISE_PRED_FILES = {
    "test_noise_10" : {
        "Baseline": fr"{PRED}\baseline_noise10_predictions.csv",
        "Proposed": fr"{PRED}\proposed_noise10_predictions.csv",
    },
    "test_noise_30" : {
        "Baseline": fr"{PRED}\baseline_noise30_predictions.csv",
        "Proposed": fr"{PRED}\proposed_noise30_predictions.csv",
    },
    "test_noise_50" : {
        "Baseline": fr"{PRED}\baseline_noise50_predictions.csv",
        "Proposed": fr"{PRED}\proposed_noise50_predictions.csv",
    },
    "test_khong_dau": {
        "Baseline": fr"{PRED}\baseline_khongdau_predictions.csv",
        "Proposed": fr"{PRED}\proposed_khongdau_predictions.csv",
    },
}

# Ket qua da co (test goc va teencode)
EXISTING = {
    ("Baseline", "test")          : (0.8108, 0.8367),
    ("Proposed", "test")          : (0.8291, 0.8510),
    ("Baseline", "test_teencode") : (0.8533, 0.9200),
    ("Proposed", "test_teencode") : (0.8947, 0.9400),
}


def load_results():
    """Doc ket qua tu cac file prediction (neu co), bo qua neu PENDING."""
    results = {}

    # Ket qua da co
    for (model, set_name), (macro_f1, acc) in EXISTING.items():
        results[(model, set_name)] = {"macro_f1": macro_f1, "accuracy": acc}

    # Doc ket qua noise
    for set_name, model_files in NOISE_PRED_FILES.items():
        for model, path in model_files.items():
            if os.path.exists(path):
                df = pd.read_csv(path)
                y_true = df["true_label"].values
                y_pred = df["predicted_label"].values
                macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
                acc      = accuracy_score(y_true, y_pred)
                results[(model, set_name)] = {"macro_f1": macro_f1, "accuracy": acc}
                print(f"  [OK] {model} / {set_name}: Macro-F1={macro_f1:.4f}")
            else:
                results[(model, set_name)] = None
                print(f"  [PENDING] {model} / {set_name}: {os.path.basename(path)}")

    return results


def plot_noise_curve(results):
    """Ve bieu do F1 theo muc nhieu (cap nhat theo ket qua hien co)."""
    noise_map = {
        "test"          : 0,
        "test_noise_10" : 10,
        "test_noise_30" : 30,
        "test_noise_50" : 50,
    }

    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = {"Baseline": "#3498db", "Proposed": "#e67e22"}

    for model in ["Baseline", "Proposed"]:
        x_pts, y_pts = [], []
        x_pend = []
        for set_name, pct in noise_map.items():
            val = results.get((model, set_name))
            if val is not None:
                x_pts.append(pct)
                y_pts.append(val["macro_f1"])
            else:
                x_pend.append(pct)

        ax.plot(x_pts, y_pts, "o-", color=colors[model],
                linewidth=2.5, markersize=9, label=model, zorder=3)
        for x, y in zip(x_pts, y_pts):
            ax.annotate(f"{y:.4f}", (x, y),
                        textcoords="offset points", xytext=(0, 11),
                        ha="center", fontsize=9.5, fontweight="bold",
                        color=colors[model])
        # Diem PENDING
        if x_pend:
            ax.scatter(x_pend,
                       [min(y_pts) - 0.03] * len(x_pend),
                       marker="x", s=120, color=colors[model],
                       linewidths=2, zorder=4, alpha=0.5)

    # Vung PENDING
    pending_sets = [pct for s, pct in noise_map.items()
                    if results.get(("Baseline", s)) is None]
    if pending_sets:
        x_min = min(pending_sets) - 3
        ax.axvspan(x_min, 53, alpha=0.07, color="gray")
        ax.text(np.mean(pending_sets), 0.56,
                "PENDING\n(can Ni 2 export\nprediction files)",
                ha="center", va="bottom", fontsize=9, color="gray",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                          edgecolor="gray", alpha=0.9))

    ax.set_xticks([0, 10, 30, 50])
    ax.set_xticklabels(
        ["0%\n(test goc)", "10%\n(noise_10)", "30%\n(noise_30)", "50%\n(noise_50)"],
        fontsize=10,
    )
    ax.set_ylim(0.5, 1.05)
    ax.set_xlabel("Muc do nhieu Teencode (%)", fontsize=12)
    ax.set_ylabel("Macro F1-score", fontsize=12)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=11, title="Mo hinh", title_fontsize=10)
    ax.set_title(
        "Duong cong Macro F1 theo muc do nhieu Teencode\n"
        "Baseline vs Proposed — PhoBERT Sentiment",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()

    out = fr"{OUT}\f1_noise_curve.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  -> Cap nhat bieu do: {out}")


def update_metrics_csv(results):
    """Cap nhat file metrics_all_sets.csv voi ket qua moi."""
    rows = []
    all_sets = [
        ("test", "test.csv (goc)"),
        ("test_teencode", "test_teencode.csv"),
        ("test_noise_10", "test_noise_10.csv"),
        ("test_noise_30", "test_noise_30.csv"),
        ("test_noise_50", "test_noise_50.csv"),
        ("test_khong_dau", "test_khong_dau.csv"),
    ]
    for set_name, set_label in all_sets:
        for model in ["Baseline", "Proposed"]:
            val = results.get((model, set_name))
            rows.append({
                "model":       model,
                "test_set":    set_name,
                "macro_f1":    round(val["macro_f1"], 4) if val else None,
                "accuracy":    round(val["accuracy"], 4) if val else None,
                "status":      "OK" if val else "PENDING",
            })
    df = pd.DataFrame(rows)
    out = fr"{OUT}\metrics_noise_summary.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"  -> Cap nhat CSV: {out}")


if __name__ == "__main__":
    print("=" * 65)
    print("  CAP NHAT KET QUA NHIEU (update_noise_results.py)")
    print("=" * 65)
    results = load_results()
    update_metrics_csv(results)
    plot_noise_curve(results)
    print("\nXong! Chay lai script nay sau khi Ni 2 export prediction files.")
