"""
So sánh lượng teencode trước và sau khi tự sinh teencode (dữ liệu v2).

    python viz/teencode_before_after.py      # chạy sau preprocess_v2.py

Cách đếm (giống nhau cho mọi file): một từ là teencode nếu
    - có trong teencode_dict.json (trừ mục ánh xạ về chính nó), hoặc
    - là biến thể trong teencode_generator.py, hoặc
    - có ký tự lặp >= 3 lần (ngonnn).
Câu bỏ dấu KHÔNG tính là teencode ở đây.
"""
import json
import os
import re
import sys

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import eda_charts as ec  # noqa: E402
from teencode_generator import TeencodeGenerator, nfc  # noqa: E402

DATA = os.path.join(ROOT, "data", "v2")
OUT = os.path.join(ROOT, "viz", "outputs", "teencode_before_after")
LONG_RE = re.compile(r"(\w)\1{2,}")
# Từ tiếng Việt bình thường / chữ cái đơn mơ hồ: không tính là teencode
AMBIGUOUS = {"quê", "chất", "đỉnh", "thiệt", "tui", "gi", "a", "e", "m", "t", "b", "c", "v", "n"}
# Từ lóng mượn tiếng Anh: đếm riêng
EN_SLANG = {"flex", "crush", "toxic", "cringe", "share", "comment", "cmt", "follow", "fl", "unfollow", "post",
            "makeup", "bff", "date", "bye", "bai", "hello", "hi", "weekend", "cute", "trend", "fan", "sale",
            "perfect", "cool", "thanks", "sorry", "fake news", "gato"}
SETS = [  # (nhóm, tên hiển thị, file)
    ("Train", "Train gốc", "train.csv"),
    ("Train", "Train tăng cường", "train_aug.csv"),
    ("Test", "Test gốc", "test.csv"),
    ("Test", "Nhiễu 10%", "test_noise_10.csv"),
    ("Test", "Nhiễu 30%", "test_noise_30.csv"),
    ("Test", "Nhiễu 50%", "test_noise_50.csv"),
]


def build_lexicon():
    with open(os.path.join(ROOT, "teencode_dict.json"), encoding="utf-8") as f:
        td = {nfc(k): nfc(v) for k, v in json.load(f).items()}
    lex = {k for k, v in td.items() if k != v}
    gen = TeencodeGenerator(os.path.join(ROOT, "teencode_dict.json"))
    multi = set()
    for vs in gen.table.values():
        for v, _, _ in vs:
            (multi if " " in v else lex).add(v)
    lex -= AMBIGUOUS | EN_SLANG
    multi -= EN_SLANG
    return lex, multi


def count(texts, lex, multi):
    per_sent, types, en = [], set(), []
    for t in texts:
        t = nfc(str(t)).lower()
        n = 0
        en.append(any(w.strip(ec.PUNCT) in EN_SLANG for w in t.split()) or "fake news" in t)
        for w in t.split():
            w = w.strip(ec.PUNCT)
            if w in lex or LONG_RE.search(w):
                n += 1
                types.add(LONG_RE.sub(r"\1\1\1", w))
        for m in multi:
            if m in t:
                n += t.count(m)
                types.add(m)
        per_sent.append(n)
    s = pd.Series(per_sent)
    return {"so_cau": len(s), "cau_co_teencode": int((s > 0).sum()),
            "ty_le_cau_co_teencode_pct": round(100 * (s > 0).mean(), 2),
            "tong_luot_teencode": int(s.sum()), "teencode_tb_moi_cau": round(s.mean(), 3),
            "so_dang_teencode_khac_nhau": len(types),
            "ty_le_cau_co_tu_long_tieng_anh_pct": round(100 * pd.Series(en).mean(), 2)}


def chart(table):
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1,
                        subplot_titles=["Câu có ít nhất 1 teencode (%)", "Số dạng teencode khác nhau"])
    colors = {"Train gốc": "#898781", "Train tăng cường": "#6250d6", "Test gốc": "#898781",
              "Nhiễu 10%": "#c9c1f5", "Nhiễu 30%": "#9a8ce6", "Nhiễu 50%": "#6250d6"}
    for c, (col, fmt) in enumerate([("ty_le_cau_co_teencode_pct", "{:.1f}%"), ("so_dang_teencode_khac_nhau", "{:,.0f}")], 1):
        fig.add_bar(row=1, col=c, x=table.index, y=table[col], showlegend=False,
                    marker=dict(color=[colors[i] for i in table.index], cornerradius=4),
                    text=[fmt.format(v).replace(",", ".") for v in table[col]], textposition="outside",
                    textfont=dict(color=ec.INK_2), cliponaxis=False,
                    hovertemplate="%{x}: %{text}<extra></extra>")
    fig.update_yaxes(row=1, col=1, ticksuffix="%", range=[0, table["ty_le_cau_co_teencode_pct"].max() * 1.2])
    fig.update_yaxes(row=1, col=2, range=[0, table["so_dang_teencode_khac_nhau"].max() * 1.2])
    fig.update_annotations(font=dict(size=13, color=ec.INK))
    a, b = table.loc["Train gốc", "ty_le_cau_co_teencode_pct"], table.loc["Train tăng cường", "ty_le_cau_co_teencode_pct"]
    return ec.style(fig, f"Tự sinh teencode nâng tỷ lệ câu có teencode trong train từ {a:.0f}% lên {b:.0f}%",
                    "Màu xám: dữ liệu gốc; màu tím: dữ liệu sau khi sinh teencode (càng đậm càng nhiều nhiễu)",
                    height=480)


def main():
    os.makedirs(OUT, exist_ok=True)
    lex, multi = build_lexicon()
    rows = {}
    for _, name, f in SETS:
        d = pd.read_csv(os.path.join(DATA, f))
        if f == "train_aug.csv":
            d = d[d["augment"] != "khong_dau"]
        rows[name] = count(d["comment"], lex, multi)
    table = pd.DataFrame(rows).T
    table.to_csv(os.path.join(OUT, "bang_truoc_sau.csv"), encoding="utf-8-sig")
    print(table.to_string())
    fig = chart(table)
    fig.write_html(os.path.join(OUT, "15_teencode_truoc_sau.html"), include_plotlyjs="cdn")
    try:
        fig.write_image(os.path.join(OUT, "15_teencode_truoc_sau.png"), width=1100, scale=2)
    except Exception as e:  # thiếu Chrome cho kaleido -> vẫn có file HTML
        print(f"(bỏ qua PNG: {str(e).strip().splitlines()[0]} -> chạy: plotly_get_chrome)")


if __name__ == "__main__":
    main()
