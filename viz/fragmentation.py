"""
Ní 1 - Trực quan hóa độ phân mảnh từ vựng của PhoBERT tokenizer, trước và sau chuẩn hóa.

Mọi số liệu được TÍNH TRỰC TIẾP từ dữ liệu (không nhập tay), trên đúng hai cột mà
mô hình nhận vào: text_raw_segmented (Baseline) và text_normalized_segmented (Đề xuất).

Chạy từ thư mục gốc của repo (cần mạng để tải tokenizer lần đầu):
    pip install transformers plotly kaleido pandas
    python viz/fragmentation.py
Hoặc chỉ định tokenizer đã tải sẵn:
    python viz/fragmentation.py --tokenizer path/to/phobert-base

Định nghĩa (một "từ" = một token sau khi tách từ bằng PyVi, vd "sản_phẩm"):
    - Từ bị phân mảnh (OOV): từ không có sẵn trong từ vựng PhoBERT, bị tách thành >= 2 subword.
    - Subword / từ: số subword trung bình PhoBERT sinh ra cho mỗi từ (1.0 = không phân mảnh).
    - Tỷ lệ token "@@": cách đo cũ trong measure_fragmentation.py, giữ lại để đối chiếu.
"""
import argparse
import json
import os
import sys

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eda_charts as ec  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "viz", "outputs", "fragmentation")
VERSIONS = {"Gốc": "text_raw_segmented", "Chuẩn hóa": "text_normalized_segmented"}
SUBSETS = ["Toàn bộ dữ liệu", "Câu có teencode"]


class SubwordCounter:
    """Bộ nhớ đệm kết quả tách subword theo từng từ (tránh gọi tokenizer lặp lại)."""

    def __init__(self, tokenizer):
        self.tok, self.cache = tokenizer, {}

    def pieces(self, word):
        if word not in self.cache:
            self.cache[word] = self.tok.tokenize(word) or [word]
        return self.cache[word]


def sentence_stats(text, sc):
    words = str(text).split()
    n_words = len(words) or 1
    pieces = [sc.pieces(w) for w in words]
    n_sub = sum(len(p) for p in pieces)
    return {
        "n_words": len(words),
        "n_oov": sum(len(p) > 1 for p in pieces),
        "n_subwords": n_sub,
        "n_at": sum(t.endswith("@@") for p in pieces for t in p),
        "sub_per_word": n_sub / n_words,
    }


def compute(df, sc):
    rows = []
    for vname, col in VERSIONS.items():
        st = pd.DataFrame([sentence_stats(t, sc) for t in df[col]], index=df.index)
        st["version"], st["is_teencode"] = vname, df["is_teencode"].values
        rows.append(st)
    per_sent = pd.concat(rows)

    summary = []
    for subset in SUBSETS:
        d = per_sent if subset == SUBSETS[0] else per_sent[per_sent["is_teencode"]]
        for vname in VERSIONS:
            s = d[d["version"] == vname]
            summary.append({
                "tap": subset, "phien_ban": vname, "so_cau": len(s),
                "so_tu": int(s["n_words"].sum()), "so_subword": int(s["n_subwords"].sum()),
                "ty_le_tu_oov_pct": 100 * s["n_oov"].sum() / s["n_words"].sum(),
                "subword_moi_tu": s["n_subwords"].sum() / s["n_words"].sum(),
                "ty_le_token_@@_pct": 100 * s["n_at"].sum() / s["n_subwords"].sum(),
            })
    return per_sent, pd.DataFrame(summary)


def fig_overview(summary):
    """Hai chỉ số phân mảnh, trước - sau chuẩn hóa, trên toàn bộ dữ liệu và trên câu có teencode."""
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12,
                        subplot_titles=["Từ bị tách thành nhiều subword (%)", "Số subword trung bình mỗi từ"])
    for c, (metric, fmt) in enumerate([("ty_le_tu_oov_pct", "{:.1f}%"), ("subword_moi_tu", "{:.3f}")], start=1):
        for vname, color in ec.VERSION_COLORS.items():
            s = summary[summary["phien_ban"] == vname].set_index("tap").reindex(SUBSETS)
            fig.add_bar(row=1, col=c, x=SUBSETS, y=s[metric], name=vname, showlegend=(c == 1),
                        legendgroup=vname, marker=dict(color=color, cornerradius=4),
                        text=[fmt.format(v) for v in s[metric]], textposition="outside",
                        textfont=dict(color=ec.INK_2),
                        hovertemplate="%{x} · " + vname + ": %{text}<extra></extra>")
    top = summary.set_index(["tap", "phien_ban"])
    fig.update_yaxes(row=1, col=1, ticksuffix="%", range=[0, summary["ty_le_tu_oov_pct"].max() * 1.25])
    fig.update_yaxes(row=1, col=2, range=[1, summary["subword_moi_tu"].max() * 1.04])
    fig.update_annotations(font=dict(size=13, color=ec.INK))
    fig.update_layout(barmode="group")
    raw = top.loc[(SUBSETS[1], "Gốc"), "ty_le_tu_oov_pct"]
    norm = top.loc[(SUBSETS[1], "Chuẩn hóa"), "ty_le_tu_oov_pct"]
    verb = "giảm" if norm < raw else "tăng"
    fig = ec.style(fig, f"Trên câu có teencode, tỷ lệ từ bị tách {verb} từ {raw:.1f}% còn {norm:.1f}%",
                   "Đo bằng tokenizer PhoBERT trên đúng dữ liệu đầu vào của hai mô hình", height=500)
    fig.update_layout(legend=dict(yanchor="top", y=-0.12, xanchor="center", x=0.5), margin=dict(b=80))
    return fig


def fig_teencode_pieces(hits, tdict, sc, n=15):
    """Mỗi teencode bị PhoBERT tách thành mấy subword, và từ chuẩn của nó thì sao."""
    # "Mảnh thừa" = số subword - số từ (0 = từ nằm nguyên vẹn trong từ vựng PhoBERT),
    # để cụm nhiều từ như "số tài khoản" không bị tính là phân mảnh.
    rows = []
    for key, cnt in hits.most_common(n):
        # eda_charts.teencode_hits trả về Counter{(teencode, từ chuẩn): số lần}
        k, target = key if isinstance(key, tuple) else (key, tdict[key])
        raw_p = sc.pieces(k)
        norm_p = [p for w in target.split() for p in sc.pieces(w)]
        rows.append((f"{k}  →  {target}", len(raw_p) - 1, len(norm_p) - len(target.split()),
                     " ".join(raw_p), " ".join(norm_p), cnt))
    rows = rows[::-1]
    fig = go.Figure()
    for lab, r, nm, *_ in rows:
        fig.add_scatter(x=[r, nm], y=[lab, lab], mode="lines", line=dict(color=ec.AXIS, width=2),
                        showlegend=False, hoverinfo="skip")
    # Gốc: vòng rỗng lớn, Chuẩn hóa: chấm đặc nhỏ -> vẫn thấy cả hai khi trùng giá trị
    for vname, idx, pidx, marker in [
        ("Gốc (teencode)", 1, 3, dict(size=16, color="rgba(0,0,0,0)", line=dict(color=ec.VERSION_COLORS["Gốc"], width=3))),
        ("Chuẩn hóa (từ chuẩn)", 2, 4, dict(size=9, color=ec.VERSION_COLORS["Chuẩn hóa"])),
    ]:
        fig.add_scatter(x=[r[idx] for r in rows], y=[r[0] for r in rows], mode="markers", name=vname,
                        marker=marker, customdata=[[r[pidx], r[5]] for r in rows],
                        hovertemplate="%{y}<br>" + vname + ": %{x} mảnh thừa<br>[%{customdata[0]}]"
                                      "<br>xuất hiện %{customdata[1]} lần<extra></extra>")
    xmax = max(max(r[1], r[2]) for r in rows) if rows else 1
    fig.update_xaxes(title="Số mảnh bị tách thừa (0 = giữ nguyên vẹn)", dtick=1, range=[-0.2, xmax + 0.3])
    intact = sum(1 for r in rows if r[1] == 0)
    return ec.style(fig, "Mỗi teencode bị PhoBERT tách vụn ra sao, trước và sau chuẩn hóa",
                    f"{n} teencode phổ biến nhất · {intact} dạng vốn có sẵn trong từ vựng PhoBERT (chuẩn hóa chỉ gộp nghĩa)",
                    height=580)


def fig_per_sentence(per_sent):
    """Phân bố số subword / từ của từng câu có teencode, trước và sau."""
    d = per_sent[per_sent["is_teencode"]]
    fig = go.Figure()
    for vname, color in ec.VERSION_COLORS.items():
        fig.add_box(x=d.loc[d["version"] == vname, "sub_per_word"], name=vname, orientation="h",
                    marker_color=color, line=dict(width=2), boxmean=True,
                    hovertemplate=vname + ": %{x:.2f} subword/từ<extra></extra>")
    fig.update_xaxes(title="Số subword trung bình mỗi từ, tính trong từng câu")
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(showlegend=False)
    return ec.style(fig, "Phân bố mức phân mảnh của từng câu có teencode",
                    "Mỗi điểm là một câu; giá trị 1.0 nghĩa là không từ nào bị tách", height=360)


def load_tokenizer(name):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(name)


def main(tokenizer=None, tokenizer_name="vinai/phobert-base"):
    os.makedirs(OUT, exist_ok=True)
    print("1. Đọc dữ liệu...")
    df = ec.load_splits(os.path.join(ROOT, "data", "v2"))
    tdict = ec.load_dict(ROOT)
    hits = ec.teencode_hits(df, tdict)

    print(f"2. Tải tokenizer {tokenizer_name}...")
    sc = SubwordCounter(tokenizer or load_tokenizer(tokenizer_name))

    print("3. Đo phân mảnh...")
    per_sent, summary = compute(df, sc)
    summary.to_csv(os.path.join(OUT, "phan_manh_tong_hop.csv"), index=False, encoding="utf-8-sig")
    print(summary.round(3).to_string(index=False))

    print("4. Vẽ biểu đồ...")
    for fig, name, w in [(fig_overview(summary), "10_phan_manh_tong_quan", 1100),
                         (fig_teencode_pieces(hits, tdict, sc), "11_teencode_bi_tach", 1000),
                         (fig_per_sentence(per_sent), "12_phan_manh_tung_cau", 1000)]:
        fig.write_html(os.path.join(OUT, f"{name}.html"), include_plotlyjs="cdn")
        try:
            fig.write_image(os.path.join(OUT, f"{name}.png"), width=w, scale=2)
        except Exception as e:
            print(f"   (bỏ qua PNG {name}: {e})")
        print(f"-> {name}")

    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary.round(4).to_dict(orient="records"), f, ensure_ascii=False, indent=2)
    print(f"\nXong! Kết quả trong {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="vinai/phobert-base",
                    help="tên trên HuggingFace hoặc thư mục tokenizer đã tải")
    main(tokenizer_name=ap.parse_args().tokenizer)
