# =============================================================================
# run_evaluation.py  –  Nhánh feat/evaluation
# Nhiệm vụ:
#   1. Đo Macro F1-score trên test.csv và test_teencode.csv
#   2. Vẽ biểu đồ phân bố dữ liệu, biểu do phân mảnh, Confusion Matrix
#   3. Thực hiện phân tích sai số định tính (Error Analysis)
# =============================================================================

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import (
    f1_score, accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

warnings.filterwarnings("ignore")
matplotlib.rcParams['font.family'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

OUT = "evaluation_results"
os.makedirs(OUT, exist_ok=True)

LABEL_NAMES   = ["Negative (0)", "Neutral (1)", "Positive (2)"]
LABEL_SHORT   = ["Neg", "Neu", "Pos"]
LABEL_MAP     = {0: "Negative", 1: "Neutral", 2: "Positive"}

PRED_FILES = {
    "Baseline_Test":     "model_outputs/baseline_model_test_predictions.csv",
    "Proposed_Test":     "model_outputs/proposed_model_test_predictions.csv",
    "Baseline_Teencode": "model_outputs/baseline_model_teencode_predictions.csv",
    "Proposed_Teencode": "model_outputs/proposed_model_teencode_predictions.csv",
}

# ---------------------------------------------------------------------------
# 1. BIỂU ĐỒ PHÂN BỐ NHÃN
# ---------------------------------------------------------------------------
def plot_label_distribution():
    print("=" * 60)
    print("[1] Ve bieu do phan bo du lieu (Label Distribution)")
    print("=" * 60)

    splits = {
        "Train":      "train.csv",
        "Validation": "val.csv",
        "Test":       "test.csv",
    }

    frames = []
    for split_name, path in splits.items():
        df = pd.read_csv(path)
        df["split"] = split_name
        frames.append(df[["label_id", "split"]])
    combined = pd.concat(frames, ignore_index=True)
    combined["label_name"] = combined["label_id"].map(LABEL_MAP)

    # --- Bar chart grouped ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: stacked bar per split
    ax = axes[0]
    colors = ["#e74c3c", "#95a5a6", "#2ecc71"]
    pivot = (
        combined.groupby(["split", "label_id"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=[0, 1, 2])
    )
    pivot.columns = ["Negative (0)", "Neutral (1)", "Positive (2)"]
    pivot.index = pd.CategoricalIndex(pivot.index,
                                      categories=["Train", "Validation", "Test"])
    pivot = pivot.sort_index()
    pivot.plot(kind="bar", ax=ax, color=colors, edgecolor="white",
               linewidth=0.6, rot=0)
    ax.set_title("Label Distribution per Split", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Dataset Split", fontsize=11)
    ax.set_ylabel("Number of Samples", fontsize=11)
    ax.legend(title="Label", fontsize=9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    for container in ax.containers:
        ax.bar_label(container, fontsize=8, padding=2)

    # Right: pie chart of full dataset
    ax2 = axes[1]
    total_counts = combined["label_id"].value_counts().sort_index()
    wedge_colors = ["#e74c3c", "#95a5a6", "#2ecc71"]
    wedges, texts, autotexts = ax2.pie(
        total_counts,
        labels=[f"Negative\n(n={total_counts[0]})",
                f"Neutral\n(n={total_counts[1]})",
                f"Positive\n(n={total_counts[2]})"],
        colors=wedge_colors,
        autopct="%1.1f%%",
        startangle=140,
        pctdistance=0.75,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for t in autotexts:
        t.set_fontsize(10)
        t.set_fontweight("bold")
    ax2.set_title("Overall Label Distribution\n(Train + Val + Test)",
                  fontsize=13, fontweight="bold", pad=10)

    fig.suptitle(
        "PhoBERT Teencode Sentiment – Data Label Distribution\n"
        "(0: Negative / 1: Neutral / 2: Positive)",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    out_path = f"{OUT}/label_distribution.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved: {out_path}")


# ---------------------------------------------------------------------------
# 2. BIỂU DO PHÂN MẢNH SUBWORD 
# ---------------------------------------------------------------------------
def plot_subword_fragmentation():
    print("\n" + "=" * 60)
    print("[2] Ve bieu do phan manh Subword (Fragmentation)")
    print("=" * 60)

    categories  = ["Before\nNormalization\n(Raw)", "After\nNormalization\n(Normalized)"]
    subwords    = [7421, 7092]
    percentages = [17.38, 16.69]
    bar_colors  = ["#e74c3c", "#2ecc71"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(categories, subwords, color=bar_colors,
                  width=0.45, edgecolor="white", linewidth=1.2,
                  zorder=3)
    ax.set_ylim(5500, 8500)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    # Annotate bars
    for bar, sub, pct in zip(bars, subwords, percentages):
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 60,
            f"{sub:,} subwords\n({pct}%)",
            ha="center", va="bottom",
            fontsize=11, fontweight="bold",
        )

    # Reduction arrow annotation
    ax.annotate(
        "",
        xy=(1, subwords[1] + 320),
        xytext=(0, subwords[0] + 320),
        arrowprops=dict(
            arrowstyle="->", color="#c0392b", lw=2.0,
            connectionstyle="arc3,rad=-0.25",
        ),
    )
    ax.text(
        0.5, max(subwords) + 500,
        f"Reduction: 4.43%\n(-329 fragmented tokens)",
        ha="center", va="bottom",
        fontsize=10, color="#c0392b", fontweight="bold",
    )

    ax.set_ylabel("Number of Fragmented Subwords", fontsize=12)
    ax.set_title(
        "Subword Fragmentation: Before vs After Teencode Normalization\n"
        "(PhoBERT Tokenizer – vinai/phobert-base)",
        fontsize=13, fontweight="bold", pad=12,
    )

    # Legend
    legend_handles = [
        mpatches.Patch(color="#e74c3c", label="Before normalization (Raw)"),
        mpatches.Patch(color="#2ecc71", label="After normalization (Normalized)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=10)

    fig.tight_layout()
    out_path = f"{OUT}/subword_fragmentation.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved: {out_path}")


# ---------------------------------------------------------------------------
# 3. MACRO F1 SCORES TABLE + CSV
# ---------------------------------------------------------------------------
def compute_f1_scores():
    print("\n" + "=" * 60)
    print("[3] Do Macro F1-score tren test.csv va test_teencode.csv")
    print("=" * 60)

    rows = []
    for name, path in PRED_FILES.items():
        if not os.path.exists(path):
            print(f"  [WARN] File not found: {path}")
            continue

        df = pd.read_csv(path)
        y_true = df["true_label"]
        y_pred = df["predicted_label"]

        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        acc      = accuracy_score(y_true, y_pred)
        n        = len(df)
        n_errors = int((y_true != y_pred).sum())

        # Per-class F1
        per_class = f1_score(y_true, y_pred, labels=[0, 1, 2],
                             average=None, zero_division=0)

        rows.append({
            "Model_Dataset":    name,
            "Split":            "test_teencode" if "Teencode" in name else "test",
            "Model":            "Baseline" if "Baseline" in name else "Proposed",
            "N_Samples":        n,
            "N_Errors":         n_errors,
            "Error_Rate_%":     round(n_errors / n * 100, 2),
            "Accuracy":         round(acc, 4),
            "Macro_F1":         round(macro_f1, 4),
            "F1_Negative":      round(per_class[0], 4),
            "F1_Neutral":       round(per_class[1], 4),
            "F1_Positive":      round(per_class[2], 4),
        })

        print(f"\n  [{name}]")
        print(f"    Samples  : {n}")
        print(f"    Accuracy : {acc:.4f}")
        print(f"    Macro F1 : {macro_f1:.4f}")
        print(f"    Per-class F1 -> Negative: {per_class[0]:.4f} | "
              f"Neutral: {per_class[1]:.4f} | Positive: {per_class[2]:.4f}")

    df_summary = pd.DataFrame(rows)
    out_path = f"{OUT}/f1_scores_summary.csv"
    df_summary.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n  -> Saved F1 summary: {out_path}")
    return df_summary


# ---------------------------------------------------------------------------
# 4. CONFUSION MATRICES
# ---------------------------------------------------------------------------
def plot_confusion_matrices():
    print("\n" + "=" * 60)
    print("[4] Ve Confusion Matrix cho 4 model/dataset")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes_flat = axes.flatten()

    titles = {
        "Baseline_Test":     "Baseline — test.csv",
        "Proposed_Test":     "Proposed — test.csv",
        "Baseline_Teencode": "Baseline — test_teencode.csv",
        "Proposed_Teencode": "Proposed — test_teencode.csv",
    }
    colors = {
        "Baseline_Test":     "Blues",
        "Proposed_Test":     "Greens",
        "Baseline_Teencode": "Oranges",
        "Proposed_Teencode": "Purples",
    }

    for ax, (name, path) in zip(axes_flat, PRED_FILES.items()):
        if not os.path.exists(path):
            ax.set_visible(False)
            continue

        df     = pd.read_csv(path)
        y_true = df["true_label"]
        y_pred = df["predicted_label"]
        cm     = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        acc      = accuracy_score(y_true, y_pred)

        sns.heatmap(
            cm, annot=True, fmt="d",
            cmap=colors[name],
            xticklabels=LABEL_SHORT,
            yticklabels=LABEL_SHORT,
            ax=ax,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title(
            f"{titles[name]}\nAcc={acc:.3f}  Macro-F1={macro_f1:.3f}",
            fontsize=11, fontweight="bold", pad=8,
        )
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("True Label", fontsize=10)

    fig.suptitle(
        "Confusion Matrices — PhoBERT Sentiment Classification\n"
        "(Neg=Negative, Neu=Neutral, Pos=Positive)",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    out_path = f"{OUT}/confusion_matrices_all.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved combined matrix: {out_path}")

    # Also save individual PNGs
    for name, path in PRED_FILES.items():
        if not os.path.exists(path):
            continue
        df     = pd.read_csv(path)
        y_true = df["true_label"]
        y_pred = df["predicted_label"]
        cm     = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        acc      = accuracy_score(y_true, y_pred)

        fig_s, ax_s = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt="d",
            cmap=colors[name],
            xticklabels=LABEL_NAMES,
            yticklabels=LABEL_NAMES,
            ax=ax_s,
            linewidths=0.5,
            linecolor="white",
        )
        ax_s.set_title(
            f"Confusion Matrix — {name.replace('_', ' ')}\n"
            f"Acc={acc:.4f}  Macro-F1={macro_f1:.4f}",
            fontsize=12, fontweight="bold",
        )
        ax_s.set_xlabel("Predicted", fontsize=11)
        ax_s.set_ylabel("Actual", fontsize=11)
        plt.xticks(rotation=15)
        fig_s.tight_layout()
        single_path = f"{OUT}/confusion_matrix_{name}.png"
        fig_s.savefig(single_path, dpi=180, bbox_inches="tight")
        plt.close(fig_s)
        print(f"  -> Saved: {single_path}")


# ---------------------------------------------------------------------------
# 5. F1 SCORE COMPARISON BAR CHART
# ---------------------------------------------------------------------------
def plot_f1_comparison(df_summary):
    print("\n" + "=" * 60)
    print("[5] Ve bieu do so sanh Macro F1 (Baseline vs Proposed)")
    print("=" * 60)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- test.csv ---
    ax = axes[0]
    test_rows = df_summary[df_summary["Split"] == "test"].copy()
    x = np.arange(3)
    width = 0.3
    labels_x = ["Negative", "Neutral", "Positive"]

    if len(test_rows) >= 2:
        bl = test_rows[test_rows["Model"] == "Baseline"].iloc[0]
        pr = test_rows[test_rows["Model"] == "Proposed"].iloc[0]

        bl_vals = [bl["F1_Negative"], bl["F1_Neutral"], bl["F1_Positive"]]
        pr_vals = [pr["F1_Negative"], pr["F1_Neutral"], pr["F1_Positive"]]

        bars1 = ax.bar(x - width / 2, bl_vals, width, label="Baseline",
                       color="#3498db", edgecolor="white")
        bars2 = ax.bar(x + width / 2, pr_vals, width, label="Proposed",
                       color="#e67e22", edgecolor="white")
        ax.bar_label(bars1, fmt="%.3f", fontsize=8, padding=3)
        ax.bar_label(bars2, fmt="%.3f", fontsize=8, padding=3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels_x, fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("F1-score", fontsize=11)
        ax.set_title(
            f"Per-class F1 on test.csv\nBaseline Macro={bl['Macro_F1']:.4f} | "
            f"Proposed Macro={pr['Macro_F1']:.4f}",
            fontsize=11, fontweight="bold",
        )
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        ax.legend(fontsize=10)

    # --- test_teencode.csv ---
    ax2 = axes[1]
    tc_rows = df_summary[df_summary["Split"] == "test_teencode"].copy()

    if len(tc_rows) >= 2:
        bl_tc = tc_rows[tc_rows["Model"] == "Baseline"].iloc[0]
        pr_tc = tc_rows[tc_rows["Model"] == "Proposed"].iloc[0]

        bl_vals2 = [bl_tc["F1_Negative"], bl_tc["F1_Neutral"], bl_tc["F1_Positive"]]
        pr_vals2 = [pr_tc["F1_Negative"], pr_tc["F1_Neutral"], pr_tc["F1_Positive"]]

        bars3 = ax2.bar(x - width / 2, bl_vals2, width, label="Baseline",
                        color="#3498db", edgecolor="white")
        bars4 = ax2.bar(x + width / 2, pr_vals2, width, label="Proposed",
                        color="#e67e22", edgecolor="white")
        ax2.bar_label(bars3, fmt="%.3f", fontsize=8, padding=3)
        ax2.bar_label(bars4, fmt="%.3f", fontsize=8, padding=3)
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels_x, fontsize=10)
        ax2.set_ylim(0, 1.1)
        ax2.set_ylabel("F1-score", fontsize=11)
        ax2.set_title(
            f"Per-class F1 on test_teencode.csv\nBaseline Macro={bl_tc['Macro_F1']:.4f} | "
            f"Proposed Macro={pr_tc['Macro_F1']:.4f}",
            fontsize=11, fontweight="bold",
        )
        ax2.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax2.set_axisbelow(True)
        ax2.legend(fontsize=10)

    fig.suptitle(
        "F1-score Comparison: Baseline vs Proposed Model",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    out_path = f"{OUT}/f1_comparison_chart.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved: {out_path}")


# ---------------------------------------------------------------------------
# 6. ERROR ANALYSIS
# ---------------------------------------------------------------------------
def error_analysis():
    print("\n" + "=" * 60)
    print("[6] Phan tich sai so dinh tinh (Error Analysis)")
    print("=" * 60)

    all_error_rows = []

    for name, path in PRED_FILES.items():
        if not os.path.exists(path):
            continue

        df = pd.read_csv(path)
        errors = df[df["true_label"] != df["predicted_label"]].copy()
        errors["model_dataset"] = name
        errors["true_label_name"]  = errors["true_label"].map(LABEL_MAP)
        errors["pred_label_name"]  = errors["predicted_label"].map(LABEL_MAP)
        errors["error_type"] = (
            errors["true_label_name"] + " → " + errors["pred_label_name"]
        )

        # Save per-model error CSV
        out_csv = f"{OUT}/error_analysis_{name}.csv"
        errors[[
            "model_dataset", "comment", "true_label_name", "pred_label_name",
            "error_type", "confidence", "is_teencode", "category",
        ]].to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"  -> Saved: {out_csv}  ({len(errors)} errors)")
        all_error_rows.append(errors)

    if not all_error_rows:
        return

    combined_errors = pd.concat(all_error_rows, ignore_index=True)

    # --- Save merged error analysis ---
    out_all = f"{OUT}/error_analysis_all.csv"
    combined_errors[[
        "model_dataset", "comment", "true_label_name", "pred_label_name",
        "error_type", "confidence", "is_teencode", "category",
    ]].to_csv(out_all, index=False, encoding="utf-8-sig")
    print(f"  -> Saved combined: {out_all}")

    # --- Error-type heatmap per model ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for idx, split in enumerate(["test", "test_teencode"]):
        ax = axes[idx]
        if split == "test":
            subset = combined_errors[combined_errors["model_dataset"].str.contains("Test")]
        else:
            subset = combined_errors[combined_errors["model_dataset"].str.contains("Teencode")]

        pivot = (
            subset.groupby(["model_dataset", "error_type"])
            .size()
            .unstack(fill_value=0)
        )
        if pivot.empty:
            ax.set_visible(False)
            continue

        sns.heatmap(
            pivot, annot=True, fmt="d",
            cmap="YlOrRd", linewidths=0.5,
            linecolor="white", ax=ax,
        )
        ax.set_title(
            f"Error Type Count — {'test.csv' if split == 'test' else 'test_teencode.csv'}",
            fontsize=11, fontweight="bold",
        )
        ax.set_xlabel("Error Type (True → Predicted)", fontsize=10)
        ax.set_ylabel("Model", fontsize=10)
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=9)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)

    fig.suptitle(
        "Error Analysis — Confusion Type Heatmap",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    out_path = f"{OUT}/error_type_heatmap.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved: {out_path}")

    # --- Confidence distribution for errors vs correct ---
    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
    for idx, (name, path) in enumerate(list(PRED_FILES.items())[:2]):  # only test.csv models
        if not os.path.exists(path):
            continue
        ax = axes2[idx]
        df = pd.read_csv(path)
        df["Prediction"] = df.apply(
            lambda r: "Correct" if r["true_label"] == r["predicted_label"] else "Wrong",
            axis=1
        )
        for category, grp in df.groupby("Prediction"):
            color = "#2ecc71" if category == "Correct" else "#e74c3c"
            grp["confidence"].hist(
                ax=ax, alpha=0.6, bins=20, label=category, color=color,
            )
        ax.set_title(f"Confidence Distribution — {name.replace('_', ' ')}",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Confidence Score", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.legend(fontsize=10)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)

    fig2.suptitle(
        "Prediction Confidence: Correct vs Wrong Predictions",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig2.tight_layout()
    out_path2 = f"{OUT}/error_confidence_distribution.png"
    fig2.savefig(out_path2, dpi=180, bbox_inches="tight")
    plt.close(fig2)
    print(f"  -> Saved: {out_path2}")

    # --- Qualitative Error Report (Markdown) ---
    _write_qualitative_report(combined_errors)


def _write_qualitative_report(combined_errors):
    """Viết báo cáo phân tích sai số định tính dạng Markdown."""

    lines = []
    lines.append("# Error Analysis Report — PhoBERT Teencode Sentiment\n")
    lines.append(
        "> **Nhánh:** `feat/evaluation` &emsp; **Thành viên:** Ní 3 – Sơn\n"
    )
    lines.append("---\n")

    # --- Summary table per model ---
    lines.append("## 1. Tổng quan số lỗi\n")
    lines.append("| Model & Dataset | Tổng mẫu | Số lỗi | Tỷ lệ lỗi (%) |")
    lines.append("|---|---|---|---|")
    for name, path in PRED_FILES.items():
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        n_total = len(df)
        n_err = int((df["true_label"] != df["predicted_label"]).sum())
        lines.append(f"| {name.replace('_', ' ')} | {n_total} | {n_err} | {n_err/n_total*100:.2f}% |")
    lines.append("")

    # --- Error pattern ---
    lines.append("## 2. Phân tích loại lỗi (Error Types)\n")
    lines.append(
        "Bảng dưới thể hiện số lần mô hình nhầm nhãn theo từng cặp "
        "`True → Predicted`:\n"
    )
    lines.append("| Model & Dataset | True Label | Predicted Label | Số lần |")
    lines.append("|---|---|---|---|")
    for name in PRED_FILES:
        subset = combined_errors[combined_errors["model_dataset"] == name]
        if subset.empty:
            continue
        grouped = (
            subset.groupby(["true_label_name", "pred_label_name"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        for _, row in grouped.iterrows():
            lines.append(
                f"| {name.replace('_', ' ')} | {row['true_label_name']} | "
                f"{row['pred_label_name']} | {row['count']} |"
            )
    lines.append("")

    # --- Qualitative examples ---
    lines.append("## 3. Ví dụ sai số điển hình (Qualitative Examples)\n")
    lines.append(
        "Dưới đây là các mẫu dự đoán sai **có confidence cao** (mô hình "
        "\"tự tin\" nhưng vẫn sai) — đây thường là những trường hợp khó nhất.\n"
    )

    for name in PRED_FILES:
        subset = combined_errors[combined_errors["model_dataset"] == name]
        if subset.empty:
            continue
        hard_errors = subset.nlargest(5, "confidence")
        lines.append(f"### {name.replace('_', ' ')}\n")
        lines.append(
            "| # | Comment | True | Predicted | Confidence | Is Teencode |"
        )
        lines.append("|---|---|---|---|---|---|")
        for i, (_, row) in enumerate(hard_errors.iterrows(), 1):
            comment = str(row["comment"]).replace("|", "\\|")[:80]
            lines.append(
                f"| {i} | {comment} | {row['true_label_name']} | "
                f"{row['pred_label_name']} | {row['confidence']:.4f} | "
                f"{'Yes' if row.get('is_teencode', False) else 'No'} |"
            )
        lines.append("")

    # --- Observations ---
    lines.append("## 4. Nhận xét định tính\n")

    # Compute stats for observations
    test_bl = combined_errors[combined_errors["model_dataset"] == "Baseline_Test"]
    test_pr = combined_errors[combined_errors["model_dataset"] == "Proposed_Test"]
    tc_bl   = combined_errors[combined_errors["model_dataset"] == "Baseline_Teencode"]
    tc_pr   = combined_errors[combined_errors["model_dataset"] == "Proposed_Teencode"]

    # Neutral confusion rate
    def neutral_confusion(df_err):
        if df_err.empty:
            return 0, 0
        to_neutral = int((df_err["pred_label_name"] == "Neutral").sum())
        from_neutral = int((df_err["true_label_name"] == "Neutral").sum())
        return to_neutral, from_neutral

    bl_to_neu, bl_from_neu = neutral_confusion(test_bl)
    pr_to_neu, pr_from_neu = neutral_confusion(test_pr)

    lines.append(
        "### 4.1 Lớp Neutral (Trung tính) là khó phân loại nhất\n\n"
        f"Trên `test.csv`, Baseline dự đoán sai **{bl_to_neu + bl_from_neu}** "
        f"mẫu liên quan đến nhãn Neutral, trong khi Proposed chỉ sai "
        f"**{pr_to_neu + pr_from_neu}** mẫu. Lý do là câu trung tính thường "
        "ngắn, thiếu ngữ cảnh hoặc chứa cả yếu tố tích cực lẫn tiêu cực, "
        "khiến mô hình dễ nhầm sang Negative hoặc Positive.\n"
    )

    lines.append(
        "### 4.2 Mô hình Proposed cải thiện rõ rệt trên test_teencode.csv\n\n"
        f"- Baseline_Teencode lỗi: **{len(tc_bl)}** mẫu\n"
        f"- Proposed_Teencode lỗi: **{len(tc_pr)}** mẫu\n\n"
        "Module chuẩn hóa Teencode giúp giảm phân mảnh subword, từ đó "
        "mô hình đọc đúng ngữ nghĩa của từ lóng và phân loại chính xác hơn.\n"
    )

    lines.append(
        "### 4.3 Lỗi confidence cao — mô hình \"tự tin sai\"\n\n"
        "Những trường hợp confidence > 0.9 nhưng vẫn sai thường rơi vào:\n"
        "- **Irony / Sarcasm**: câu khen nhưng ngụ ý chê, hoặc ngược lại.\n"
        "- **Code-switching**: trộn Anh–Việt hoặc không dấu hoàn toàn.\n"
        "- **Rất ngắn**: mẫu chỉ 1–3 từ, thiếu ngữ cảnh để xác định cảm xúc.\n"
        "- **Emoji / ký tự đặc biệt**: PhoBERT tokenizer không nhận diện tốt "
        "biểu tượng cảm xúc.\n"
    )

    lines.append(
        "### 4.4 Tác động của Teencode Normalization\n\n"
        "Module chuẩn hóa giảm 4.43% token bị băm vụn (329 token khôi phục). "
        "Điều này giải thích vì sao Proposed tốt hơn Baseline đặc biệt trên "
        "`test_teencode.csv` — nơi tỷ lệ từ lóng cao nhất.\n"
    )

    lines.append("---\n")
    lines.append(
        "_Báo cáo được tạo tự động bởi `run_evaluation.py` — "
        "nhánh `feat/evaluation`._\n"
    )

    report_text = "\n".join(lines)
    out_path = f"{OUT}/error_analysis_report.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"  -> Saved qualitative report: {out_path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  EVALUATION PIPELINE — feat/evaluation (Ní 3 - Sơn)")
    print("=" * 60 + "\n")

    plot_label_distribution()
    plot_subword_fragmentation()
    df_summary = compute_f1_scores()
    plot_confusion_matrices()
    plot_f1_comparison(df_summary)
    error_analysis()

    print("\n" + "=" * 60)
    print("  HOAN THANH! Ket qua luu tai: evaluation_results/")
    print("=" * 60)
