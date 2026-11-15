"""
Chay script nay sau khi lay code cua Ni 2 ve.
Cap nhat toan bo bieu do va CSV voi so lieu noise sets thuc te.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report

matplotlib.rcParams["font.family"] = ["DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

BASE    = r"d:\Ky 7\Hoc may nang cao\phobert-teencode-sentiment"
PRED    = fr"{BASE}\model_outputs"
OUT_M   = fr"{BASE}\evaluation\01_metrics"
OUT_E   = fr"{BASE}\evaluation\02_error_analysis"
os.makedirs(OUT_M, exist_ok=True)
os.makedirs(OUT_E, exist_ok=True)

LABEL_NAMES = ["Negative", "Neutral", "Positive"]
LABEL_SHORT = ["Neg", "Neu", "Pos"]
LABEL_MAP   = {0: "Negative", 1: "Neutral", 2: "Positive"}

# Ten file prediction cua Ni 2 (moi)
PRED_FILES = {
    ("Baseline", "test")          : fr"{PRED}\baseline_model_test_predictions.csv",
    ("Baseline", "test_teencode") : fr"{PRED}\baseline_model_teencode_predictions.csv",
    ("Baseline", "noise_10")      : fr"{PRED}\baseline_model_noise_10_predictions.csv",
    ("Baseline", "noise_30")      : fr"{PRED}\baseline_model_noise_30_predictions.csv",
    ("Baseline", "noise_50")      : fr"{PRED}\baseline_model_noise_50_predictions.csv",
    ("Baseline", "khong_dau")     : fr"{PRED}\baseline_model_khong_dau_predictions.csv",
    ("Proposed", "test")          : fr"{PRED}\proposed_model_test_predictions.csv",
    ("Proposed", "test_teencode") : fr"{PRED}\proposed_model_teencode_predictions.csv",
    ("Proposed", "noise_10")      : fr"{PRED}\proposed_model_noise_10_predictions.csv",
    ("Proposed", "noise_30")      : fr"{PRED}\proposed_model_noise_30_predictions.csv",
    ("Proposed", "noise_50")      : fr"{PRED}\proposed_model_noise_50_predictions.csv",
    ("Proposed", "khong_dau")     : fr"{PRED}\proposed_model_khong_dau_predictions.csv",
}

SET_ORDER = ["test", "noise_10", "noise_30", "noise_50", "khong_dau", "test_teencode"]
SET_LABEL = {
    "test"         : "test\n(0% noise)",
    "noise_10"     : "noise_10\n(~3% bién đổi)",
    "noise_30"     : "noise_30\n(~9% bién đổi)",
    "noise_50"     : "noise_50\n(~15% bién đổi)",
    "khong_dau"    : "khong_dau\n(không dấu)",
    "test_teencode": "test_teencode\n(teencode thực)",
}

# ===========================================================================
# 1. TINH LAI TOAN BO METRICS
# ===========================================================================
def compute_all_metrics():
    print("=" * 65)
    print(" [1] TINH LAI TOAN BO METRICS (6 TAP, 2 MODEL)")
    print("=" * 65)
    rows = []
    for (model, set_name), path in PRED_FILES.items():
        if not os.path.exists(path):
            print(f"  [SKIP] {model} / {set_name}: file not found")
            continue
        df     = pd.read_csv(path)
        y_true = df["true_label"].values
        y_pred = df["predicted_label"].values

        mac_f1  = f1_score(y_true, y_pred, average="macro",    zero_division=0)
        wei_f1  = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        acc     = accuracy_score(y_true, y_pred)
        per_f1  = f1_score(y_true, y_pred, labels=[0,1,2], average=None, zero_division=0)
        n_err   = int((y_true != y_pred).sum())

        rows.append({
            "Model": model, "Test_Set": set_name,
            "N_Samples": len(df), "N_Errors": n_err,
            "Error_Rate_%": round(n_err/len(df)*100, 2),
            "Accuracy": round(acc, 4),
            "Macro_F1": round(mac_f1, 4),
            "Weighted_F1": round(wei_f1, 4),
            "F1_Negative": round(per_f1[0], 4),
            "F1_Neutral":  round(per_f1[1], 4),
            "F1_Positive": round(per_f1[2], 4),
        })
        print(f"  [{model:8s}] {set_name:15s}: "
              f"Macro-F1={mac_f1:.4f} | Weighted-F1={wei_f1:.4f} | Acc={acc:.4f}")

    df_metrics = pd.DataFrame(rows)
    out = fr"{OUT_M}\metrics_all_sets.csv"
    df_metrics.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n  -> Saved: {out}")
    return df_metrics

# ===========================================================================
# 2. BIEU DO F1 THEO MUC NHIEU (day du so lieu)
# ===========================================================================
def plot_noise_curve(df_metrics):
    print("\n" + "=" * 65)
    print(" [2] BIEU DO F1 THEO MUC NHIEU (day du so lieu)")
    print("=" * 65)

    noise_sets = ["test", "noise_10", "noise_30", "noise_50"]
    noise_pct  = [0, 10, 30, 50]
    colors     = {"Baseline": "#3498db", "Proposed": "#e67e22"}

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    # --- (a) Duong cong Macro F1 ---
    ax = axes[0]
    for model in ["Baseline", "Proposed"]:
        sub = df_metrics[df_metrics["Model"] == model]
        y = []
        for s in noise_sets:
            row = sub[sub["Test_Set"] == s]
            y.append(row["Macro_F1"].values[0] if len(row) else None)

        x_valid = [x for x, v in zip(noise_pct, y) if v is not None]
        y_valid = [v for v in y if v is not None]
        ax.plot(x_valid, y_valid, "o-", color=colors[model],
                linewidth=2.5, markersize=9, label=model, zorder=3)
        for x, yv in zip(x_valid, y_valid):
            ax.annotate(f"{yv:.4f}", (x, yv),
                        textcoords="offset points", xytext=(0, 11),
                        ha="center", fontsize=9, fontweight="bold",
                        color=colors[model])

    ax.set_xticks(noise_pct)
    ax.set_xticklabels(["0%\n(test goc)", "10%\n(noise_10)",
                         "30%\n(noise_30)", "50%\n(noise_50)"], fontsize=10)
    ax.set_ylim(0.6, 1.0)
    ax.set_xlabel("Muc do nhieu Teencode (%)", fontsize=12)
    ax.set_ylabel("Macro F1-score", fontsize=12)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=11, title="Mo hinh")
    ax.set_title("Duong cong Macro F1 theo muc do nhieu\n(0% → 10% → 30% → 50%)",
                 fontsize=12, fontweight="bold")

    # --- (b) Thanh grouped: tat ca 6 tap ---
    ax2 = axes[1]
    x    = np.arange(len(SET_ORDER))
    width = 0.35
    bl = df_metrics[df_metrics["Model"] == "Baseline"].set_index("Test_Set")
    pr = df_metrics[df_metrics["Model"] == "Proposed"].set_index("Test_Set")

    bl_vals = [bl.loc[s, "Macro_F1"] if s in bl.index else 0 for s in SET_ORDER]
    pr_vals = [pr.loc[s, "Macro_F1"] if s in pr.index else 0 for s in SET_ORDER]

    bars1 = ax2.bar(x - width/2, bl_vals, width, label="Baseline",
                    color="#3498db", edgecolor="white")
    bars2 = ax2.bar(x + width/2, pr_vals, width, label="Proposed",
                    color="#e67e22", edgecolor="white")
    ax2.bar_label(bars1, fmt="%.3f", fontsize=7.5, padding=2)
    ax2.bar_label(bars2, fmt="%.3f", fontsize=7.5, padding=2)
    ax2.set_xticks(x)
    ax2.set_xticklabels([SET_LABEL[s] for s in SET_ORDER], fontsize=8)
    ax2.set_ylim(0.5, 1.05)
    ax2.set_ylabel("Macro F1-score", fontsize=12)
    ax2.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax2.set_axisbelow(True)
    ax2.legend(fontsize=11)
    ax2.set_title("Macro F1 tren 6 tap test\n(Baseline vs Proposed)",
                  fontsize=12, fontweight="bold")

    fig.suptitle("Ket qua Macro F1 — PhoBERT Sentiment Classification",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = fr"{OUT_M}\f1_noise_curve.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved: {out}")

# ===========================================================================
# 3. BANG KET QUA CHINH (tong hop dep)
# ===========================================================================
def plot_main_results_table(df_metrics):
    print("\n" + "=" * 65)
    print(" [3] BANG KET QUA CHINH")
    print("=" * 65)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis("off")

    bl = df_metrics[df_metrics["Model"] == "Baseline"].set_index("Test_Set")
    pr = df_metrics[df_metrics["Model"] == "Proposed"].set_index("Test_Set")

    table_data = []
    headers = ["Test Set", "Baseline\nAcc", "Baseline\nMacro-F1",
               "Proposed\nAcc", "Proposed\nMacro-F1", "Delta\nMacro-F1"]
    set_display = {
        "test"         : "test.csv (goc)",
        "noise_10"     : "test_noise_10",
        "noise_30"     : "test_noise_30",
        "noise_50"     : "test_noise_50",
        "khong_dau"    : "test_khong_dau",
        "test_teencode": "test_teencode",
    }
    for s in SET_ORDER:
        bl_row = bl.loc[s] if s in bl.index else None
        pr_row = pr.loc[s] if s in pr.index else None
        bl_acc  = f"{bl_row['Accuracy']:.4f}"  if bl_row is not None else "-"
        bl_f1   = f"{bl_row['Macro_F1']:.4f}"  if bl_row is not None else "-"
        pr_acc  = f"{pr_row['Accuracy']:.4f}"  if pr_row is not None else "-"
        pr_f1   = f"{pr_row['Macro_F1']:.4f}"  if pr_row is not None else "-"
        delta   = f"+{pr_row['Macro_F1'] - bl_row['Macro_F1']:.4f}" \
                  if (bl_row is not None and pr_row is not None) else "-"
        table_data.append([set_display.get(s, s), bl_acc, bl_f1, pr_acc, pr_f1, delta])

    tbl = ax.table(cellText=table_data, colLabels=headers,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.6)

    # Mau header
    for j in range(len(headers)):
        tbl[(0, j)].set_facecolor("#2c3e50")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")
    # Mau dong xen ke + highlight delta
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            tbl[(i, j)].set_facecolor("#f8f9fa" if i % 2 == 0 else "white")
        # Delta column
        delta_val = table_data[i-1][5]
        if delta_val.startswith("+"):
            tbl[(i, 5)].set_facecolor("#d5f5e3")
            tbl[(i, 5)].set_text_props(color="#1e8449", fontweight="bold")

    ax.set_title("Bang ket qua chinh — Macro F1 tren 6 tap test\n"
                 "(Baseline vs Proposed — PhoBERT Teencode Sentiment)",
                 fontsize=13, fontweight="bold", pad=20)
    fig.tight_layout()
    out = fr"{OUT_M}\main_results_table.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved: {out}")

# ===========================================================================
# 4. CONFUSION MATRICES (6 TAP × 2 MODEL = 12, hien thi 2×3 cho moi model)
# ===========================================================================
def plot_confusion_matrices_all(df_metrics):
    print("\n" + "=" * 65)
    print(" [4] CONFUSION MATRICES — 6 TAP TEST")
    print("=" * 65)

    cmaps = {"Baseline": "Blues", "Proposed": "Greens"}
    sub_sets = ["test", "test_teencode", "noise_10", "noise_30", "noise_50", "khong_dau"]

    for model in ["Baseline", "Proposed"]:
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        for ax, set_name in zip(axes.flatten(), sub_sets):
            key = (model, set_name)
            if key not in PRED_FILES or not os.path.exists(PRED_FILES[key]):
                ax.set_visible(False)
                continue
            df     = pd.read_csv(PRED_FILES[key])
            y_true = df["true_label"].values
            y_pred = df["predicted_label"].values
            cm     = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
            mac_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
            acc    = accuracy_score(y_true, y_pred)

            sns.heatmap(cm, annot=True, fmt="d", cmap=cmaps[model],
                        xticklabels=LABEL_SHORT, yticklabels=LABEL_SHORT,
                        ax=ax, linewidths=0.5, linecolor="white")
            ax.set_title(f"{SET_LABEL[set_name]}\nAcc={acc:.3f}  F1={mac_f1:.3f}",
                         fontsize=10, fontweight="bold")
            ax.set_xlabel("Predicted", fontsize=9)
            ax.set_ylabel("Actual", fontsize=9)

        fig.suptitle(f"Confusion Matrices — Model: {model}\n"
                     f"(Neg=Negative, Neu=Neutral, Pos=Positive)",
                     fontsize=14, fontweight="bold", y=1.01)
        fig.tight_layout()
        out = fr"{OUT_M}\confusion_matrices_{model}.png"
        fig.savefig(out, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> Saved: {out}")

# ===========================================================================
# 5. CAP NHAT ERROR ANALYSIS (noise sets moi)
# ===========================================================================
def update_error_analysis():
    print("\n" + "=" * 65)
    print(" [5] CAP NHAT ERROR ANALYSIS (them noise sets)")
    print("=" * 65)

    all_errors = []
    summary_rows = []

    for (model, set_name), path in PRED_FILES.items():
        if not os.path.exists(path):
            continue
        df     = pd.read_csv(path)
        errors = df[df["true_label"] != df["predicted_label"]].copy()
        errors["model"]          = model
        errors["test_set"]       = set_name
        errors["true_name"]      = errors["true_label"].map(LABEL_MAP)
        errors["pred_name"]      = errors["predicted_label"].map(LABEL_MAP)
        errors["error_type"]     = errors["true_name"] + " -> " + errors["pred_name"]
        errors["replacements"]   = errors.get("replacements", "[]") if "replacements" in errors.columns else "[]"
        all_errors.append(errors)

        n_total = len(df)
        n_err   = len(errors)
        summary_rows.append({
            "model": model, "test_set": set_name,
            "n_total": n_total, "n_errors": n_err,
            "error_rate_%": round(n_err/n_total*100, 2),
        })
        print(f"  [{model:8s}] {set_name:15s}: {n_err}/{n_total} errors ({n_err/n_total*100:.1f}%)")

    combined = pd.concat(all_errors, ignore_index=True)
    cols = [c for c in ["model","test_set","comment","true_name","pred_name",
                         "error_type","confidence","replacements","is_teencode","category"]
            if c in combined.columns]
    combined[cols].to_csv(fr"{OUT_E}\error_detail_all.csv",
                          index=False, encoding="utf-8-sig")

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(fr"{OUT_E}\error_summary.csv",
                      index=False, encoding="utf-8-sig")

    # Danh sach cau sai GUI NI 1 (tu test + test_teencode)
    for_ni1 = combined[combined["test_set"].isin(["test","test_teencode"])].copy()
    for_ni1_cols = [c for c in ["model","test_set","comment","true_name","pred_name",
                                  "error_type","confidence","replacements","is_teencode"]
                    if c in for_ni1.columns]
    for_ni1[for_ni1_cols].sort_values(["test_set","model"]).to_csv(
        fr"{OUT_E}\error_list_for_ni1.csv", index=False, encoding="utf-8-sig")

    print(f"\n  -> Saved: error_detail_all.csv ({len(combined)} total errors)")
    print(f"  -> Saved: error_summary.csv")
    print(f"  -> Saved: error_list_for_ni1.csv ({len(for_ni1)} errors from test+teencode)")

    # Bieu do so sanh error rate
    fig, ax = plt.subplots(figsize=(12, 5))
    pivot = df_summary.pivot(index="test_set", columns="model",
                             values="error_rate_%").reindex(SET_ORDER).fillna(0)
    pivot.plot(kind="bar", ax=ax, color=["#3498db", "#e67e22"],
               edgecolor="white", rot=20, width=0.65)
    ax.set_title("Ti le loi (%) theo tap test — Baseline vs Proposed",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Error Rate (%)")
    ax.set_xlabel("")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", fontsize=8.5, padding=2)
    ax.legend(title="Model", fontsize=10)
    fig.tight_layout()
    fig.savefig(fr"{OUT_E}\error_rate_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved: error_rate_comparison.png")

# ===========================================================================
# MAIN
# ===========================================================================
if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  CAP NHAT SAU KHI LAY CODE NI 2 — feat/evaluation")
    print("=" * 65 + "\n")

    df_metrics = compute_all_metrics()
    plot_noise_curve(df_metrics)
    plot_main_results_table(df_metrics)
    plot_confusion_matrices_all(df_metrics)
    update_error_analysis()

    print("\n" + "=" * 65)
    print("  HOAN THANH! Tat ca ket qua da duoc cap nhat.")
    print("=" * 65)
