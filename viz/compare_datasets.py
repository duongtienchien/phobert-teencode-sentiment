"""
Kiểm tra một bộ dữ liệu mới có hợp với đề tài hay không, bằng cách đo các chỉ số
"độ nhiễu" giống hệt bộ hiện tại (vne_dataset.csv) rồi đặt cạnh nhau.

    python viz/compare_datasets.py path/to/train.json path/to/valid.json path/to/test.json

Mặc định đọc định dạng UIT-VSMEC (json, cột Sentence / Emotion) và quy 7 cảm xúc về 3 nhãn.
Kết quả: viz/outputs/compare/  (bảng chỉ số + biểu đồ)
"""
import json
import os
import re
import sys

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eda_charts as ec  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "viz", "outputs", "compare")

# UIT-VSMEC: 7 cảm xúc -> 3 nhãn. Surprise mơ hồ (vui hoặc sợ) nên bỏ.
VSMEC_MAP = {"Enjoyment": 2, "Other": 1, "Sadness": 0, "Anger": 0, "Disgust": 0, "Fear": 0}
VNE_MAP = {"negative": 0, "toxic": 0, "neutral": 1, "positive": 2}

EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF☀-➿]")
EMOTICON_RE = re.compile(r"(?::\)+|:\(+|=\)+|:v\b|:d\b|<3|\^\^|:p\b|:3\b|=\(+)", re.I)
# Mục từ điển ánh xạ về chính nó (vd "thằng") không tính là teencode.
# Viết tắt không có nguyên âm: dc, k, j, vl, dm, cx, mk, bn ...
ABBR_RE = re.compile(r"^[bcdfghjklmnpqrstvwxzđ]{1,4}$")
DATASET_COLORS = {"Bộ hiện tại (vne)": "#898781", "UIT-VSMEC": "#1baf7a"}


def load_vsmec(paths):
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            rows += json.load(f)
    df = pd.DataFrame(rows).rename(columns={"Sentence": "text", "Emotion": "raw_label"})
    df["label_id"] = df["raw_label"].map(VSMEC_MAP)
    return df


def load_vne():
    df = pd.read_csv(os.path.join(ROOT, "vne_dataset.csv")).rename(columns={"comment": "text", "label": "raw_label"})
    df["label_id"] = df["raw_label"].map(VNE_MAP)
    return df


def tokens(text):
    return [t.strip(ec.PUNCT) for t in ec.REPEAT_RE.sub(r"\1", str(text).lower()).split()]


def metrics(df, tdict):
    toks = df["text"].map(tokens)
    has = lambda f: toks.map(lambda ws: any(f(w) for w in ws if w)).mean() * 100  # noqa: E731
    style = df["text"].astype(str).map(ec.writing_style)
    kept = df["label_id"].notna()
    lab = df.loc[kept, "label_id"].map(ec.LABEL_NAMES)
    nodau_by_label = style[kept].eq("Không dấu").groupby(lab).mean() * 100
    norm = df["text"].astype(str).str.lower().str.strip()
    return {
        "Số câu dùng được (3 nhãn)": int(kept.sum()),
        "Câu có teencode trong từ điển (%)": has(lambda w: w in tdict and tdict[w] != w),
        "Câu có viết tắt không nguyên âm (%)": has(lambda w: bool(ABBR_RE.match(w))),
        "Câu có emoji hoặc emoticon (%)": (df["text"].astype(str).str.contains(EMOJI_RE)
                                           | df["text"].astype(str).str.contains(EMOTICON_RE)).mean() * 100,
        "Câu viết không dấu (%)": style.eq("Không dấu").mean() * 100,
        "Câu tiếng Anh / khác (%)": style.eq("Khác").mean() * 100,
        "Câu trùng lặp (%)": norm.duplicated().mean() * 100,
        "Độ dài trung bình (tiếng)": df["text"].astype(str).str.split().str.len().mean(),
        "Không dấu ở nhãn Tiêu cực (%)": nodau_by_label.get("Tiêu cực", 0),
        "Không dấu ở nhãn Tích cực (%)": nodau_by_label.get("Tích cực", 0),
        **{f"Nhãn {n} (%)": (lab == n).mean() * 100 for n in ec.LABEL_NAMES.values()},
    }


def fig_compare(table):
    panels = ["Câu có teencode trong từ điển (%)", "Câu có viết tắt không nguyên âm (%)",
              "Câu có emoji hoặc emoticon (%)", "Câu viết không dấu (%)",
              "Câu tiếng Anh / khác (%)", "Câu trùng lặp (%)"]
    fig = make_subplots(rows=2, cols=3, subplot_titles=[p.replace(" (%)", "") for p in panels],
                        vertical_spacing=0.2, horizontal_spacing=0.08)
    for i, p in enumerate(panels):
        r, c = i // 3 + 1, i % 3 + 1
        for name, color in DATASET_COLORS.items():
            v = table.loc[p, name]
            fig.add_bar(row=r, col=c, x=[name.split(" (")[0]], y=[v], name=name, showlegend=(i == 0),
                        marker=dict(color=color, cornerradius=4), text=[f"{v:.1f}%"],
                        textposition="outside", textfont=dict(color=ec.INK_2), cliponaxis=False,
                        hovertemplate=name + "<br>" + p + ": %{y:.1f}%<extra></extra>")
        fig.update_yaxes(row=r, col=c, ticksuffix="%", range=[0, max(table.loc[p].max() * 1.3, 1)])
    fig.update_annotations(font=dict(size=13, color=ec.INK))
    fig.update_xaxes(showticklabels=False)
    t1 = table.loc[panels[0]]
    fig = ec.style(fig, "So sánh độ nhiễu: bộ hiện tại và UIT-VSMEC",
                   f"Teencode theo từ điển của nhóm: {t1.iloc[0]:.1f}% so với {t1.iloc[1]:.1f}% số câu",
                   height=620)
    fig.update_layout(legend=dict(yanchor="top", y=-0.04, xanchor="center", x=0.5), margin=dict(b=70))
    return fig


def fig_labels(table):
    fig = go.Figure()
    names = list(DATASET_COLORS)
    for lab in ec.LABEL_NAMES.values():
        vals = [table.loc[f"Nhãn {lab} (%)", n] for n in names]
        fig.add_bar(y=[n.split(" (")[0] for n in names], x=vals, name=lab, orientation="h",
                    marker=dict(color=ec.LABEL_COLORS[lab], line=dict(color=ec.SURFACE, width=2)),
                    text=[f"{v:.0f}%" for v in vals], textposition="inside", insidetextanchor="middle",
                    textfont=dict(color="white"), hovertemplate="%{y} · " + lab + ": %{x:.1f}%<extra></extra>")
    fig.update_layout(barmode="stack", legend_traceorder="normal")
    fig.update_xaxes(range=[0, 100], ticksuffix="%", title="Tỷ lệ câu (%)")
    fig.update_yaxes(autorange="reversed")
    return ec.style(fig, "Phân bố nhãn sau khi quy về 3 lớp",
                    "UIT-VSMEC: Enjoyment → Tích cực; Sadness, Anger, Disgust, Fear → Tiêu cực; Other → Trung tính; bỏ Surprise",
                    height=320)


def main(paths):
    os.makedirs(OUT, exist_ok=True)
    tdict = ec.load_dict(ROOT)
    table = pd.DataFrame({"Bộ hiện tại (vne)": metrics(load_vne(), tdict),
                          "UIT-VSMEC": metrics(load_vsmec(paths), tdict)})
    table.round(2).to_csv(os.path.join(OUT, "bang_so_sanh.csv"), encoding="utf-8-sig")
    print(table.round(2).to_string())
    for fig, name, w in [(fig_compare(table), "13_so_sanh_do_nhieu", 1100), (fig_labels(table), "14_so_sanh_nhan", 1000)]:
        fig.write_html(os.path.join(OUT, f"{name}.html"), include_plotlyjs="cdn")
        fig.write_image(os.path.join(OUT, f"{name}.png"), width=w, scale=2)
        print("->", name)


if __name__ == "__main__":
    main(sys.argv[1:])
