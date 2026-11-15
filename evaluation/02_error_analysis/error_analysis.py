"""
evaluation/02_error_analysis/error_analysis.py
-----------------------------------------------
Nhanh feat/evaluation (Ni 3 - Son)
Nhiem vu:
  - Dung cot 'replacements' de xem cau sai co teencode bi sot khong
  - Phan loai loi: mia mai, phu dinh kep, ngu canh thieu
  - Xuat danh sach cau sai gui Ni 1
"""

import os, ast, re
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.rcParams["font.family"] = ["DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

BASE    = r"d:\Ky 7\Hoc may nang cao\phobert-teencode-sentiment"
PRED    = fr"{BASE}\model_outputs"
DATA_V2 = fr"{BASE}\data\v2"
OUT     = fr"{BASE}\evaluation\02_error_analysis"
os.makedirs(OUT, exist_ok=True)

LABEL_MAP  = {0: "Negative", 1: "Neutral", 2: "Positive"}

# ===========================================================================
# UTILS
# ===========================================================================
IRONY_PATTERNS = [
    # Dau hieu mia mai / sarcasm bang emoji va emoticon
    r'(=\)|\)\)|:v|:3|:>|haha|hehe|hihi|he+|🤣|😂|😏|😒|🙄|😑|😐)',
    # Khen qua muc / sarcasm khen
    r'\b(hay\s+(đấy|nhỉ|lắm|thật|ghê|vậy)|đỉnh\s+(thật|vậy|ghê|quá)|giỏi\s+(thật|ghê|lắm)|'
    r'thông\s+minh\s+(quá|thật|vậy)|tuyệt\s+(vời\s+quá|thật)|xịn\s+(ghê|thật|lắm))\b',
    # Phu nhan sau do khen / boc phet
    r'\b(nói\s+thì|nghe\s+thì|nhìn\s+thì).*(hay|tốt|đẹp|ổn)',
    r'\b(ai\s+ngờ|không\s+ngờ|bất\s+ngờ)\b',
    r'\b(chưa\s+bao\s+giờ\s+thấy).*(hay|tốt|giỏi|đỉnh)',
    r'\b(mình\s+không\s+hiểu\s+sao\s+có\s+người)\b',
    # Irony "Dạ / Ừ" + tu khen che
    r'^(dạ|ừ|uh|oke|okay)[,.]?\s*(em|anh|mình|tôi)?\s*(nghèo|dốt|ngu|tệ|lười)',
    # Cau hoi tu man / sarcasm question
    r'\?(.*)(được\s+gì|làm\s+được|hiểu\s+gì|biết\s+gì)',
    # :)) hay :D cuoi khay
    r':\)+|:\s*D\b',
    # Spill the tea / slang sarcasm
    r'\b(spill|tea|cap\s+xong|ngồi\s+đơ|plot\s+twist)\b',
]

NEG_DOUBLE_PATTERNS = [
    r'\b(không\s+phải\s+không|chẳng\s+phải\s+chẳng|không\s+ai\s+không)\b',
    r'\b(không\s+hề\s+không|không\s+bao\s+giờ\s+không)\b',
    r'\b(ko\s+phai\s+ko|kg\s+phai\s+kg)\b',
    r'\b(ai\s+chê\s+thì\s+thôi|chê\s+thì\s+thôi)\b',
    r'\b(không\s+phải\s+là\s+không|chẳng\s+phải\s+là)\b',
    r'\b(không\s+sao[,.]?\s+chỉ\s+là)\b',
]

CONTEXT_MISSING_PATTERNS = [
    # Rat ngan: 1-3 tu don
    r'^[a-zA-ZÀ-ỹ\s]{1,15}$',
    r'^(okay|ok|oke|fine|sure|yep|no|yes|có|ko|k|r|um|uh|ờ|ừ|đúng|sai)\.?$',
    # Tieng Anh thuan khong co context tieng Viet
    r'^[a-zA-Z\s]+$',
    # Cau chi co 1-2 tu
    r'^\S+(\s\S+)?$',
]

def classify_error(row):
    """Phan loai loai loi tu comment goc."""
    text = str(row.get("comment", "")).lower()
    categories = []

    # 1. Mia mai / sarcasm
    if any(re.search(p, text, re.IGNORECASE) for p in IRONY_PATTERNS):
        categories.append("mia_mai_sarcasm")

    # 2. Phu dinh kep
    if any(re.search(p, text, re.IGNORECASE) for p in NEG_DOUBLE_PATTERNS):
        categories.append("phu_dinh_kep")

    # 3. Ngu canh thieu (cau qua ngan)
    if any(re.search(p, text, re.IGNORECASE) for p in CONTEXT_MISSING_PATTERNS):
        categories.append("ngu_canh_thieu")

    # 4. Kiem tra replacements (teencode bi sot)
    reps = row.get("replacements", "[]")
    if isinstance(reps, str):
        try:
            reps_list = ast.literal_eval(reps)
        except Exception:
            reps_list = []
    else:
        reps_list = []

    # Neu da duoc normalize nhung van sai -> co the la teencode phuc tap
    has_rep = len(reps_list) > 0

    if not categories:
        if has_rep:
            categories.append("teencode_phuc_tap")
        else:
            categories.append("other")

    return "|".join(categories), has_rep, reps_list


def has_missed_teencode(replacements_list):
    """Kiem tra xem cau co teencode bi sot trong normalizer khong."""
    # Neu replacements rong ma comment co dau hieu teencode
    return len(replacements_list) == 0  # sot het


# ===========================================================================
# 1. LOAD VA GHEP DU LIEU
# ===========================================================================
def load_and_merge():
    """
    Ghep prediction files voi data/v2/test_teencode.csv
    de lay cot 'replacements'.
    """
    pred_bl = pd.read_csv(fr"{PRED}\baseline_model_test_predictions.csv")
    pred_pr = pd.read_csv(fr"{PRED}\proposed_model_test_predictions.csv")
    pred_bl_tc = pd.read_csv(fr"{PRED}\baseline_model_teencode_predictions.csv")
    pred_pr_tc = pd.read_csv(fr"{PRED}\proposed_model_teencode_predictions.csv")

    # data/v2 co cot 'replacements'
    test_v2    = pd.read_csv(fr"{DATA_V2}\test.csv")
    test_tc_v2 = pd.read_csv(fr"{DATA_V2}\test_teencode.csv")

    # --- Ghep test.csv theo comment ---
    # Map: comment_lower -> replacements
    rep_map_full = dict(zip(
        test_v2["comment"].str.lower().str.strip(),
        test_v2["replacements"]
    ))
    rep_map_tc = dict(zip(
        test_tc_v2["comment"].str.lower().str.strip(),
        test_tc_v2["replacements"]
    ))

    def add_replacements(df, rep_map):
        df = df.copy()
        df["replacements"] = df["comment"].str.lower().str.strip().map(rep_map).fillna("[]")
        return df

    pred_bl    = add_replacements(pred_bl, {**rep_map_full, **rep_map_tc})
    pred_pr    = add_replacements(pred_pr, {**rep_map_full, **rep_map_tc})
    pred_bl_tc = add_replacements(pred_bl_tc, {**rep_map_tc, **rep_map_full})
    pred_pr_tc = add_replacements(pred_pr_tc, {**rep_map_tc, **rep_map_full})

    return pred_bl, pred_pr, pred_bl_tc, pred_pr_tc


# ===========================================================================
# 2. PHAN TICH SAI SO CHI TIET
# ===========================================================================
def analyze_errors(df, model_name, set_name):
    errors = df[df["true_label"] != df["predicted_label"]].copy()
    errors["true_name"]  = errors["true_label"].map(LABEL_MAP)
    errors["pred_name"]  = errors["predicted_label"].map(LABEL_MAP)
    errors["error_type"] = errors["true_name"] + " -> " + errors["pred_name"]

    # Phan loai loi
    errors[["error_category", "has_replacement", "replacements_list"]] = \
        errors.apply(lambda r: pd.Series(classify_error(r)), axis=1)

    # Kiem tra teencode bi sot: replacements rong nhung is_teencode=True
    errors["teencode_missed"] = (
        errors.get("is_teencode", pd.Series([False]*len(errors))) == True
    ) & (errors["has_replacement"] == False)

    return errors


def run_error_analysis():
    print("=" * 65)
    print(" ERROR ANALYSIS — DUNG COT REPLACEMENTS")
    print("=" * 65)

    pred_bl, pred_pr, pred_bl_tc, pred_pr_tc = load_and_merge()

    datasets = {
        ("Baseline", "test")        : pred_bl,
        ("Proposed", "test")        : pred_pr,
        ("Baseline", "test_teencode"): pred_bl_tc,
        ("Proposed", "test_teencode"): pred_pr_tc,
    }

    all_errors = []
    summary_rows = []

    for (model, set_name), df_pred in datasets.items():
        errors = analyze_errors(df_pred, model, set_name)
        errors["model"]    = model
        errors["test_set"] = set_name
        all_errors.append(errors)

        n_total  = len(df_pred)
        n_err    = len(errors)
        n_missed = int(errors["teencode_missed"].sum())

        # Phan loai
        cat_counts = {}
        for cats in errors["error_category"]:
            for c in cats.split("|"):
                cat_counts[c] = cat_counts.get(c, 0) + 1

        summary_rows.append({
            "model": model, "test_set": set_name,
            "n_total": n_total, "n_errors": n_err,
            "error_rate_%": round(n_err/n_total*100, 2),
            "n_teencode_missed": n_missed,
            **{f"n_{k}": v for k, v in cat_counts.items()}
        })

        print(f"\n  [{model}] {set_name}: {n_err}/{n_total} errors "
              f"({n_err/n_total*100:.1f}%)")
        print(f"    Teencode bi sot trong loi: {n_missed}")
        for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
            print(f"    - {cat}: {cnt}")

    combined = pd.concat(all_errors, ignore_index=True)

    # --- Save chi tiet ---
    cols_save = ["model","test_set","comment","true_name","pred_name",
                 "error_type","error_category","confidence",
                 "has_replacement","replacements","teencode_missed",
                 "category","is_teencode"]
    cols_save = [c for c in cols_save if c in combined.columns]
    out_detail = fr"{OUT}\error_detail_all.csv"
    combined[cols_save].to_csv(out_detail, index=False, encoding="utf-8-sig")
    print(f"\n  -> Saved detail: {out_detail}")

    # --- Save summary ---
    df_summary = pd.DataFrame(summary_rows)
    out_summary = fr"{OUT}\error_summary.csv"
    df_summary.to_csv(out_summary, index=False, encoding="utf-8-sig")
    print(f"  -> Saved summary: {out_summary}")

    # --- Danh sach cau sai GUI NI 1 ---
    for_ni1_cols = ["model","test_set","comment","true_name","pred_name",
                    "error_category","confidence","replacements","is_teencode"]
    for_ni1_cols = [c for c in for_ni1_cols if c in combined.columns]
    out_ni1 = fr"{OUT}\error_list_for_ni1.csv"
    combined[for_ni1_cols].sort_values(
        ["test_set","model","error_category"]
    ).to_csv(out_ni1, index=False, encoding="utf-8-sig")
    print(f"  -> Saved list for Ni 1: {out_ni1}")

    return combined, df_summary


# ===========================================================================
# 3. BIEU DO PHAN TICH SAI SO
# ===========================================================================
def plot_error_charts(combined, df_summary):
    print("\n  Ve bieu do error analysis...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- (a) Error rate per model & set ---
    ax = axes[0]
    pivot = df_summary.pivot(index="test_set", columns="model",
                             values="error_rate_%").fillna(0)
    pivot.plot(kind="bar", ax=ax, color=["#3498db","#e67e22"],
               edgecolor="white", rot=20)
    ax.set_title("Error Rate (%) by Model & Test Set",
                 fontsize=11, fontweight="bold")
    ax.set_ylabel("Error Rate (%)")
    ax.set_xlabel("")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(title="Model", fontsize=9)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", fontsize=8, padding=2)

    # --- (b) Error category breakdown ---
    ax2 = axes[1]
    # Dem tong error category tren tat ca
    cat_all = {}
    for cats in combined["error_category"]:
        for c in str(cats).split("|"):
            cat_all[c] = cat_all.get(c, 0) + 1
    df_cat = pd.DataFrame(list(cat_all.items()),
                          columns=["category","count"]).sort_values("count", ascending=True)
    colors_cat = plt.cm.Set2(np.linspace(0, 1, len(df_cat)))
    df_cat.plot.barh(x="category", y="count", ax=ax2,
                     color=colors_cat, legend=False)
    ax2.set_title("Error Category Distribution\n(All Models & Sets)",
                  fontsize=11, fontweight="bold")
    ax2.set_xlabel("Count")
    ax2.set_ylabel("")
    ax2.xaxis.grid(True, linestyle="--", alpha=0.5)
    ax2.set_axisbelow(True)
    for i, v in enumerate(df_cat["count"]):
        ax2.text(v + 0.3, i, str(v), va="center", fontsize=9)

    # --- (c) Confusion: True -> Predicted pie ---
    ax3 = axes[2]
    err_type_counts = combined["error_type"].value_counts()
    wedge_colors = plt.cm.Paired(np.linspace(0, 1, len(err_type_counts)))
    ax3.pie(err_type_counts.values,
            labels=err_type_counts.index,
            colors=wedge_colors,
            autopct="%1.1f%%",
            startangle=90,
            pctdistance=0.75,
            wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    ax3.set_title("Error Types Distribution\n(True → Predicted)",
                  fontsize=11, fontweight="bold")

    fig.suptitle("Error Analysis — PhoBERT Teencode Sentiment",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = fr"{OUT}\error_analysis_charts.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved: {out}")

    # --- Confidence distribution: Correct vs Wrong ---
    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
    for idx, (model, df_pred) in enumerate([
        ("Baseline", pd.read_csv(fr"{PRED}\baseline_model_test_predictions.csv")),
        ("Proposed", pd.read_csv(fr"{PRED}\proposed_model_test_predictions.csv")),
    ]):
        ax = axes2[idx]
        correct = df_pred[df_pred["true_label"] == df_pred["predicted_label"]]["confidence"]
        wrong   = df_pred[df_pred["true_label"] != df_pred["predicted_label"]]["confidence"]
        ax.hist(correct, bins=20, alpha=0.65, color="#2ecc71", label=f"Correct (n={len(correct)})")
        ax.hist(wrong,   bins=20, alpha=0.65, color="#e74c3c", label=f"Wrong   (n={len(wrong)})")
        ax.set_title(f"Confidence Distribution — {model}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Confidence Score")
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
    fig2.suptitle("Prediction Confidence: Correct vs Wrong",
                  fontsize=13, fontweight="bold", y=1.01)
    fig2.tight_layout()
    out2 = fr"{OUT}\confidence_distribution.png"
    fig2.savefig(out2, dpi=180, bbox_inches="tight")
    plt.close(fig2)
    print(f"  -> Saved: {out2}")


# ===========================================================================
# MAIN
# ===========================================================================
if __name__ == "__main__":
    combined, df_summary = run_error_analysis()
    plot_error_charts(combined, df_summary)
    print("\n" + "=" * 65)
    print("  DONE! Ket qua tai: evaluation/02_error_analysis/")
    print("=" * 65)
