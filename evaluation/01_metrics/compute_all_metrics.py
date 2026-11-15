"""
evaluation/01_metrics/compute_all_metrics.py
--------------------------------------------
Nhanh feat/evaluation 
Nhiem vu:
  - Tinh Macro F1, Weighted F1, Precision, Recall, Confusion Matrix
    tren 6 tap test (test, test_teencode, test_noise_10/30/50, test_khong_dau)
  - Bao cao trung binh +- do lech chuan qua 3 seed (da gop trong 1 file du doan)
  - Ghep theo cot 'id' giua file du doan va file test
"""

import os, ast
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)

matplotlib.rcParams["font.family"] = ["DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

BASE    = r"d:\Ky 7\Hoc may nang cao\phobert-teencode-sentiment"
PRED    = fr"{BASE}\model_outputs"
DATA_V2 = fr"{BASE}\data\v2"
OUT     = fr"{BASE}\evaluation\01_metrics"
os.makedirs(OUT, exist_ok=True)

LABEL_NAMES = ["Negative", "Neutral", "Positive"]

# ---------------------------------------------------------------------------
# Mapping: ten model -> file du doan (da co san tu Ni 2)
# ---------------------------------------------------------------------------
PRED_FILES = {
    "Baseline": fr"{PRED}\baseline_model_test_predictions.csv",
    "Proposed": fr"{PRED}\proposed_model_test_predictions.csv",
}

# Cac tap test trong data/v2 (6 tap)
TEST_SETS = {
    "test"          : fr"{DATA_V2}\test.csv",
    "test_teencode" : fr"{DATA_V2}\test_teencode.csv",
    "test_noise_10" : fr"{DATA_V2}\test_noise_10.csv",
    "test_noise_30" : fr"{DATA_V2}\test_noise_30.csv",
    "test_noise_50" : fr"{DATA_V2}\test_noise_50.csv",
    "test_khong_dau": fr"{DATA_V2}\test_khong_dau.csv",
}


def get_metrics(y_true, y_pred):
    mac_f1  = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    wei_f1  = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    mac_pre = precision_score(y_true, y_pred, average="macro",    zero_division=0)
    mac_rec = recall_score   (y_true, y_pred, average="macro",    zero_division=0)
    acc     = accuracy_score (y_true, y_pred)
    per_f1  = f1_score(y_true, y_pred, labels=[0,1,2], average=None, zero_division=0)
    return dict(
        accuracy=acc, macro_f1=mac_f1, weighted_f1=wei_f1,
        macro_precision=mac_pre, macro_recall=mac_rec,
        f1_neg=per_f1[0], f1_neu=per_f1[1], f1_pos=per_f1[2],
    )


# ---------------------------------------------------------------------------
# 1. Doc prediction files, ghep theo id
# ---------------------------------------------------------------------------
def load_preds_by_id(pred_path, test_path):
    """
    Ghep file du doan voi file test theo cot 'id'.
    Prediction file hien tai chua co 'id' — dung index khop
    (vi Ni 2 xuat theo thu tu test set, index = 0..N-1).
    """
    df_pred = pd.read_csv(pred_path)
    df_test = pd.read_csv(test_path)

    # Neu pred co id -> merge theo id
    if "id" in df_pred.columns and "id" in df_test.columns:
        df = df_test[["id","label_id"]].merge(
            df_pred[["id","predicted_label","confidence"]],
            on="id", how="inner"
        )
        y_true = df["label_id"].values
        y_pred = df["predicted_label"].values
    else:
        # Fallback: Khop theo thu tu (chi ap dung voi test.csv day du)
        # Loc theo id co trong test set
        if "id" in df_test.columns:
            # Dung comment de ghep
            df_pred2 = df_pred.copy().reset_index(drop=True)
            df_test2 = df_test.copy().reset_index(drop=True)
            n = min(len(df_pred2), len(df_test2))
            y_true = df_test2["label_id"].values[:n]
            y_pred = df_pred2["predicted_label"].values[:n]
        else:
            y_true = df_pred["true_label"].values
            y_pred = df_pred["predicted_label"].values

    return y_true, y_pred, df_pred


def compute_metrics_all_sets():
    print("=" * 65)
    print(" [1] TINH METRICS TREN 6 TAP TEST")
    print("=" * 65)

    # --- Dung file du doan hien co (tu test.csv goc 490 mau) ---
    # Cac file du doan cua Ni 2 duoc xuat tu test.csv root (490 mau)
    # data/v2/test.csv co 448 mau (phien ban v2)
    # => Su dung prediction files hien co, mapping theo comment
    pred_bl = pd.read_csv(fr"{PRED}\baseline_model_test_predictions.csv")
    pred_pr = pd.read_csv(fr"{PRED}\proposed_model_test_predictions.csv")
    pred_bl_tc = pd.read_csv(fr"{PRED}\baseline_model_teencode_predictions.csv")
    pred_pr_tc = pd.read_csv(fr"{PRED}\proposed_model_teencode_predictions.csv")

    # --- Build result rows ---
    rows = []

    # Test + teencode (da co du lieu chinh xac)
    for model_name, pred_test, pred_tc in [
        ("Baseline", pred_bl, pred_bl_tc),
        ("Proposed", pred_pr, pred_pr_tc),
    ]:
        for set_name, df_pred in [("test", pred_test), ("test_teencode", pred_tc)]:
            y_true = df_pred["true_label"].values
            y_pred = df_pred["predicted_label"].values
            m = get_metrics(y_true, y_pred)
            rows.append({"model": model_name, "test_set": set_name,
                         "n_samples": len(df_pred), **m})
            print(f"  [{model_name}] {set_name}: "
                  f"Macro-F1={m['macro_f1']:.4f} | "
                  f"Weighted-F1={m['weighted_f1']:.4f} | "
                  f"Acc={m['accuracy']:.4f}")

    # Noise sets + khong_dau: dung data/v2, ghep theo comment voi pred hien co
    # NOTE: Day la uoc tinh tu prediction file test.csv hien co
    # (chua co pred file rieng cho noise — se can Ni 2 export them)
    # => Ta simulate bang cach filter subset tuong ung
    for noise_set in ["test_noise_10","test_noise_30","test_noise_50","test_khong_dau"]:
        df_v2 = pd.read_csv(fr"{DATA_V2}\{noise_set}.csv")
        # Noise set la bien the cua test set goc, dung source_id de ghep
        if "source_id" in df_v2.columns:
            # Lay predicted_label tu pred_bl dua tren comment khop
            # (do noise set chinh la cac ban bien the)
            for model_name, df_pred in [("Baseline", pred_bl), ("Proposed", pred_pr)]:
                # Map comment goc -> predicted
                pred_map = dict(zip(df_pred["comment"].str.lower().str.strip(),
                                    df_pred["predicted_label"]))
                y_true_list, y_pred_list = [], []
                for _, row in df_v2.iterrows():
                    # Tim comment goc trong pred
                    src_id = row.get("source_id", row["id"])
                    # Dung label_id cua tap noise
                    y_true_list.append(row["label_id"])
                    # Du doan: lay tu pred theo index (approximation)
                    # vi model chua chay tren noise set
                    y_pred_list.append(row["label_id"])  # placeholder

                # NOTE: placeholder - can Ni 2 chay model tren noise sets
                # Hien tai ghi ro la PENDING
                rows.append({
                    "model": model_name, "test_set": noise_set,
                    "n_samples": len(df_v2),
                    "accuracy": None, "macro_f1": None, "weighted_f1": None,
                    "macro_precision": None, "macro_recall": None,
                    "f1_neg": None, "f1_neu": None, "f1_pos": None,
                    "note": "PENDING - can Ni 2 export prediction file cho noise sets"
                })
            break  # chi can ghi 1 lan
        # Neu khong co source_id
        for model_name in ["Baseline", "Proposed"]:
            rows.append({
                "model": model_name, "test_set": noise_set,
                "n_samples": len(df_v2),
                "accuracy": None, "macro_f1": None, "weighted_f1": None,
                "macro_precision": None, "macro_recall": None,
                "f1_neg": None, "f1_neu": None, "f1_pos": None,
                "note": "PENDING - can Ni 2 export prediction file cho noise sets"
            })

    df_metrics = pd.DataFrame(rows)
    out = fr"{OUT}\metrics_all_sets.csv"
    df_metrics.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n  -> Saved: {out}")
    return df_metrics, pred_bl, pred_pr, pred_bl_tc, pred_pr_tc


# ---------------------------------------------------------------------------
# 2. BAO CAO TRUNG BINH ± DO LECH CHUAN QUA 3 SEED
# ---------------------------------------------------------------------------
def compute_seed_stats():
    """
    Du lieu hien tai: Ni 2 chi xuat 1 seed (seed=42).
    Tao bao cao mau chuan de Ni 2 co the dien vao sau.
    Ta simulate 3 seed tu 1 file hien co (bootstrap) de demo format.
    """
    print("\n" + "=" * 65)
    print(" [2] BAO CAO TRUNG BINH +- DO LECH CHUAN (3 SEED)")
    print("=" * 65)

    pred_bl = pd.read_csv(fr"{PRED}\baseline_model_test_predictions.csv")
    pred_pr = pd.read_csv(fr"{PRED}\proposed_model_test_predictions.csv")

    # Simulate 3 seed bang bootstrap (chi de demo format; thay bang ket qua thuc te)
    np.random.seed(0)
    rows = []
    for model_name, df_pred in [("Baseline", pred_bl), ("Proposed", pred_pr)]:
        y_true = df_pred["true_label"].values
        y_pred = df_pred["predicted_label"].values

        seed_metrics = []
        for seed in [42, 123, 456]:  # 3 seed
            # Hien tai chi co seed=42, seed khac se la PENDING
            if seed == 42:
                m = get_metrics(y_true, y_pred)
            else:
                # Placeholder: se duoc thay bang ket qua thuc tu Ni 2
                m = {k: None for k in
                     ["accuracy","macro_f1","weighted_f1",
                      "macro_precision","macro_recall",
                      "f1_neg","f1_neu","f1_pos"]}
            m["seed"] = seed
            m["model"] = model_name
            seed_metrics.append(m)
            rows.append(m)

        # Tinh mean & std chi khi co du 3 ket qua thuc
        valid = [m for m in seed_metrics if m["macro_f1"] is not None]
        if len(valid) >= 1:
            for metric in ["accuracy","macro_f1","weighted_f1",
                           "macro_precision","macro_recall"]:
                vals = [m[metric] for m in valid if m[metric] is not None]
                mean = np.mean(vals)
                std  = np.std(vals) if len(vals) > 1 else 0.0
                print(f"  [{model_name}] {metric}: "
                      f"{mean:.4f} ± {std:.4f}  (n={len(vals)} seeds available)")

    df_seeds = pd.DataFrame(rows)
    out_seeds = fr"{OUT}\seed_results.csv"
    df_seeds.to_csv(out_seeds, index=False, encoding="utf-8-sig")
    print(f"\n  -> Saved seed template: {out_seeds}")
    print("  NOTE: Seed 123 va 456 con PENDING — Ni 2 se bong vao sau")
    return df_seeds


# ---------------------------------------------------------------------------
# 3. CONFUSION MATRICES (6 tap, 2 model)
# ---------------------------------------------------------------------------
def plot_confusion_matrices(pred_bl, pred_pr, pred_bl_tc, pred_pr_tc):
    print("\n" + "=" * 65)
    print(" [3] VE CONFUSION MATRICES")
    print("=" * 65)

    datasets = {
        "Baseline | test.csv"        : (pred_bl["true_label"],  pred_bl["predicted_label"],  "Blues"),
        "Proposed | test.csv"        : (pred_pr["true_label"],  pred_pr["predicted_label"],  "Greens"),
        "Baseline | test_teencode"   : (pred_bl_tc["true_label"], pred_bl_tc["predicted_label"], "Oranges"),
        "Proposed | test_teencode"   : (pred_pr_tc["true_label"], pred_pr_tc["predicted_label"], "Purples"),
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    for ax, (title, (yt, yp, cmap)) in zip(axes.flatten(), datasets.items()):
        cm = confusion_matrix(yt, yp, labels=[0, 1, 2])
        mac_f1 = f1_score(yt, yp, average="macro", zero_division=0)
        acc    = accuracy_score(yt, yp)
        sns.heatmap(cm, annot=True, fmt="d", cmap=cmap,
                    xticklabels=["Neg","Neu","Pos"],
                    yticklabels=["Neg","Neu","Pos"],
                    ax=ax, linewidths=0.5, linecolor="white")
        ax.set_title(f"{title}\nAcc={acc:.3f}  Macro-F1={mac_f1:.3f}",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Actual", fontsize=10)

    fig.suptitle("Confusion Matrices — PhoBERT Sentiment Classification",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = fr"{OUT}\confusion_matrices.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved: {out}")


# ---------------------------------------------------------------------------
# 4. BIEU DO F1 THEO MUC NHIEU (0% -> 10% -> 30% -> 50%)
# ---------------------------------------------------------------------------
def plot_noise_f1_curve():
    """
    Hien tai chi co ket qua cho test (0% noise) va test_teencode.
    Ve khung bieu do chuan; dien so lieu PENDING tu Ni 2.
    """
    print("\n" + "=" * 65)
    print(" [4] BIEU DO F1 THEO MUC NHIEU")
    print("=" * 65)

    pred_bl = pd.read_csv(fr"{PRED}\baseline_model_test_predictions.csv")
    pred_pr = pd.read_csv(fr"{PRED}\proposed_model_test_predictions.csv")

    bl_f1_0 = f1_score(pred_bl["true_label"], pred_bl["predicted_label"],
                       average="macro", zero_division=0)
    pr_f1_0 = f1_score(pred_pr["true_label"], pred_pr["predicted_label"],
                       average="macro", zero_division=0)

    # Ket qua thuc tu noise sets (PENDING - can Ni 2)
    # Dien placeholder None, duoc ve dut doan
    noise_pct = [0, 10, 30, 50]
    bl_scores = [bl_f1_0, None, None, None]  # PENDING
    pr_scores = [pr_f1_0, None, None, None]  # PENDING

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"Baseline": "#3498db", "Proposed": "#e67e22"}

    for model_name, scores in [("Baseline", bl_scores), ("Proposed", pr_scores)]:
        x_valid = [x for x, v in zip(noise_pct, scores) if v is not None]
        y_valid = [v for v in scores if v is not None]
        ax.plot(x_valid, y_valid, "o-", color=colors[model_name],
                linewidth=2.5, markersize=8, label=model_name)
        for x, y in zip(x_valid, y_valid):
            ax.annotate(f"{y:.4f}", (x, y),
                        textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=9, fontweight="bold",
                        color=colors[model_name])

    # Danh dau diem PENDING
    pending_x = [10, 30, 50]
    for px in pending_x:
        ax.axvline(px, color="gray", linestyle=":", alpha=0.4)
    ax.text(30, (bl_f1_0 + pr_f1_0)/2,
            "⚠ PENDING\n(can Ni 2 export\npredictions cho\nnoise sets)",
            ha="center", va="center", fontsize=9, color="gray",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                      edgecolor="gray", alpha=0.8))

    ax.set_xticks(noise_pct)
    ax.set_xticklabels(["0%\n(test goc)", "10%\n(noise_10)", "30%\n(noise_30)", "50%\n(noise_50)"])
    ax.set_xlabel("Muc do nhieu Teencode (%)", fontsize=12)
    ax.set_ylabel("Macro F1-score", fontsize=12)
    ax.set_ylim(0.5, 1.05)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=11, title="Mo hinh")
    ax.set_title("Duong cong F1 theo muc do nhieu Teencode\n(0% → 10% → 30% → 50%)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = fr"{OUT}\f1_noise_curve.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved (khung): {out}")
    print("  NOTE: So lieu noise 10/30/50 con PENDING — can Ni 2 export")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df_metrics, pred_bl, pred_pr, pred_bl_tc, pred_pr_tc = compute_metrics_all_sets()
    compute_seed_stats()
    plot_confusion_matrices(pred_bl, pred_pr, pred_bl_tc, pred_pr_tc)
    plot_noise_f1_curve()
    print("\n" + "=" * 65)
    print("  DONE! Ket qua tai: evaluation/01_metrics/")
    print("=" * 65)
