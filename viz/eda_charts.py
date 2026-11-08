"""
Ní 1 - Khám phá dữ liệu & Trực quan hóa tiền xử lý.

Mỗi hàm `fig_*` nhận DataFrame và trả về một plotly Figure, nên Dashboard
Streamlit (Ní 3) có thể gọi lại trực tiếp:  st.plotly_chart(fig_label_by_split(df))
"""
import json
import re
from collections import Counter

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ----------------------------------------------------------------------------
# Quy ước màu & giao diện (dùng chung cho mọi biểu đồ của nhóm)
# ----------------------------------------------------------------------------
LABEL_NAMES = {0: "Tiêu cực", 1: "Trung tính", 2: "Tích cực"}
# Nhãn có thứ tự cực tính -> thang phân kỳ đỏ / xám / xanh
LABEL_COLORS = {"Tiêu cực": "#e34948", "Trung tính": "#898781", "Tích cực": "#2a78d6"}
# So sánh dữ liệu gốc và dữ liệu đã chuẩn hóa
VERSION_COLORS = {"Gốc": "#eb6834", "Chuẩn hóa": "#6250d6"}
SPLIT_ORDER = ["Train", "Validation", "Test"]

INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
SURFACE, GRID, AXIS = "#fcfcfb", "#e1e0d9", "#c3c2b7"
FONT = "system-ui, -apple-system, 'Segoe UI', 'DejaVu Sans', sans-serif"


def style(fig, title, subtitle=None, height=460):
    """Áp giao diện chung: nền sáng, lưới mảnh, chữ trung tính."""
    text = f"<b>{title}</b>"
    if subtitle:
        text += f"<br><span style='font-size:13px;color:{INK_2}'>{subtitle}</span>"
    fig.update_layout(
        title=dict(text=text, x=0, xanchor="left", font=dict(size=18, color=INK)),
        font=dict(family=FONT, size=13, color=INK_2),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        height=height, margin=dict(l=20, r=30, t=95, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1,
                    title_text="", font=dict(color=INK_2)),
        hoverlabel=dict(bgcolor="white", font=dict(family=FONT, color=INK)),
        bargap=0.3,
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False,
                     tickfont=dict(color=MUTED), title_font=dict(color=INK_2))
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False,
                     tickfont=dict(color=MUTED), title_font=dict(color=INK_2))
    return fig


# ----------------------------------------------------------------------------
# Tiện ích dữ liệu
# ----------------------------------------------------------------------------
REPEAT_RE = re.compile(r"(\w)\1{2,}")  # giống preprocess.py: "ngonnn" -> "ngon"

# Hư từ phổ biến, bỏ khi đếm "top từ" để thấy từ mang nghĩa.
# Giữ lại "không" vì đó chính là đích chuẩn hóa của ko/k/hok.
STOPWORDS = set("""
là và của có cho thì mà này đó với các những một được đã đang sẽ cũng như để khi
ở trong ra vào lại nữa thế vậy rồi đi lên xuống nào gì ai bị nên hay hoặc nhưng
tôi tao mày bạn mình nó họ chúng_ta chúng_tôi em anh chị ông bà cái con người
rất quá lắm thật luôn chỉ còn vẫn đều từ theo về tại vì nếu thôi à ạ ơi nhé nha
nhỉ hả hở ha haha hihi kkk ta thấy làm biết muốn phải
""".split())


def load_splits(root="."):
    """Đọc train/val/test, gắn tên tập và tên nhãn."""
    parts = []
    for name, file in [("Train", "train.csv"), ("Validation", "val.csv"), ("Test", "test.csv")]:
        d = pd.read_csv(f"{root}/{file}")
        d["split"] = name
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    df["label_name"] = df["label_id"].map(LABEL_NAMES)
    df["comment"] = df["comment"].astype(str)
    df["normalized_text"] = df["normalized_text"].astype(str)
    df["len_raw"] = df["comment"].str.split().str.len()
    df["len_norm"] = df["normalized_text"].str.split().str.len()
    if "replacements" in df.columns:  # dữ liệu v2: module ghi lại những gì đã thay
        df["changed"] = df["replacements"].fillna("[]") != "[]"
    else:
        df["changed"] = df["comment"].str.lower() != df["normalized_text"]
    df["category_clean"] = (df["category"].astype(str).str.lower()
                            .str.replace("\\", "", regex=False).str.strip())
    df["writing_style"] = df["comment"].apply(writing_style)
    return df


def load_dict(root="."):
    with open(f"{root}/teencode_dict.json", encoding="utf-8") as f:
        return json.load(f)


def teencode_hits(df, tdict):
    """
    Đếm teencode mà module THỰC SỰ thay, dạng Counter{(cũ, mới): số lần}.
    Dữ liệu v2: đọc cột `replacements` (chỉ tính từ là chữ, bỏ emoji / emoticon / số tiền).
    Dữ liệu v1: tách theo dấu cách và tra từ điển như preprocess.py cũ.
    """
    c = Counter()
    if "replacements" in df.columns:
        for r in df["replacements"].fillna("[]"):
            for pair in json.loads(r):
                old, new = pair.split("→", 1)
                if re.fullmatch(r"[^\W\d_]+", old):
                    c[(old.lower(), new)] += 1
        return c
    for t in df["comment"]:
        for w in REPEAT_RE.sub(r"\1", t.lower()).split():
            if w in tdict:
                c[(w, tdict[w])] += 1
    return c


PUNCT = ".,!?;:\"'()[]{}…“”‘’*~"


def teencode_missed(df, tdict):
    """
    Teencode có trong từ điển nhưng module v1 bỏ sót vì dính dấu câu (vd "ko," hoặc "fb?"),
    và module hiện tại (cột normalized_text) còn sót bao nhiêu trong số đó.
    Trả về (Counter v1 bỏ sót, Counter hiện tại vẫn sót, ví dụ).
    """
    v1, still, examples = Counter(), Counter(), {}
    for t, norm in zip(df["comment"], df["normalized_text"]):
        norm_tokens = set(re.findall(r"\w+", str(norm).lower()))
        for tok in REPEAT_RE.sub(r"\1", t.lower()).split():
            core = tok.strip(PUNCT)
            if tok not in tdict and core != tok and core in tdict:
                v1[core] += 1
                examples.setdefault(core, tok)
                if core in norm_tokens and tdict[core] != core:
                    still[core] += 1
    return v1, still, examples


# Kiểu viết: có dấu / tiếng Việt không dấu / khác (tiếng Anh, quá ngắn)
VN_DIACRITIC = re.compile(r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]", re.I)
# Một âm tiết tiếng Việt viết không dấu: phụ âm đầu + nguyên âm + phụ âm cuối
VN_SYLLABLE = re.compile(r"^(ngh|ng|nh|ch|gh|gi|kh|ph|qu|th|tr|[bcdghklmnprstvx])?[aeiouy]{1,3}(ng|nh|ch|[cmnpt])?$")
STYLE_ORDER = ["Có dấu", "Không dấu", "Khác"]
STYLE_COLORS = {"Có dấu": "#d3d1c9", "Không dấu": "#eb6834", "Khác": "#6250d6"}


def writing_style(text):
    """Phân loại gần đúng (heuristic): >= 80% từ có dạng âm tiết tiếng Việt -> không dấu."""
    if VN_DIACRITIC.search(text):
        return "Có dấu"
    words = re.findall(r"[a-z]+", text.lower())
    if len(words) >= 2 and sum(bool(VN_SYLLABLE.match(w)) for w in words) / len(words) >= 0.8:
        return "Không dấu"
    return "Khác"


def top_words(texts, n=10):
    c = Counter()
    for t in texts:
        for w in str(t).lower().split():
            w = w.strip(".,!?;:\"'()[]…-")
            if len(w) > 1 and w not in STOPWORDS and not w.isdigit():
                c[w] += 1
    return c.most_common(n)


# ----------------------------------------------------------------------------
# 1. EDA đa chiều
# ----------------------------------------------------------------------------
def fig_label_by_split(df):
    """Tỷ lệ nhãn trên từng tập - kiểm chứng chia phân tầng."""
    ct = pd.crosstab(df["split"], df["label_name"]).reindex(SPLIT_ORDER)
    pct = ct.div(ct.sum(axis=1), axis=0) * 100
    fig = go.Figure()
    for lab in LABEL_NAMES.values():
        fig.add_bar(
            y=[f"{s} ({ct.loc[s].sum():,})".replace(",", ".") for s in SPLIT_ORDER],
            x=pct[lab], name=lab, orientation="h",
            marker=dict(color=LABEL_COLORS[lab], line=dict(color=SURFACE, width=2)),
            text=[f"{v:.0f}%" for v in pct[lab]], textposition="inside",
            insidetextanchor="middle", textfont=dict(color="white"),
            customdata=ct[lab], hovertemplate="%{y}<br>" + lab + ": %{customdata} câu (%{x:.1f}%)<extra></extra>",
        )
    fig.update_layout(barmode="stack", legend_traceorder="normal")
    fig.update_xaxes(title="Tỷ lệ bình luận (%)", range=[0, 100], ticksuffix="%")
    fig.update_yaxes(autorange="reversed")
    return style(fig, "Ba tập dữ liệu có cùng tỷ lệ nhãn",
                 "Chia phân tầng 80/10/10 giữ nguyên tỷ lệ Tiêu cực, Trung tính, Tích cực ở mọi tập", height=340)


def fig_length_by_label(df):
    """Độ dài bình luận (số tiếng) theo nhãn."""
    fig = go.Figure()
    for lab in LABEL_NAMES.values():
        s = df.loc[df["label_name"] == lab, "len_raw"]
        fig.add_box(y=s, name=lab, marker_color=LABEL_COLORS[lab], line=dict(width=2),
                    boxmean=True, hovertemplate=lab + "<br>%{y} tiếng<extra></extra>")
    fig.update_yaxes(title="Số tiếng trong một bình luận")
    fig.update_layout(showlegend=False)
    return style(fig, "Bình luận ở cả ba nhãn đều ngắn, độ dài gần như nhau",
                 "Phân bố số tiếng mỗi bình luận theo nhãn (đường nét đứt: trung bình)")


def fig_teencode_rate(df):
    """Tỷ lệ bình luận chứa teencode theo nhãn."""
    r = df.groupby("label_name")["is_teencode"].agg(["mean", "sum", "count"]).reindex(LABEL_NAMES.values())
    overall = df["is_teencode"].mean() * 100
    fig = go.Figure(go.Bar(
        x=list(r.index), y=r["mean"] * 100,
        marker=dict(color=[LABEL_COLORS[l] for l in r.index], cornerradius=4),
        text=[f"{v:.1f}%" for v in r["mean"] * 100], textposition="outside", textfont=dict(color=INK_2),
        customdata=r[["sum", "count"]].values,
        hovertemplate="%{x}: %{customdata[0]} / %{customdata[1]} câu có teencode<extra></extra>",
    ))
    fig.add_hline(y=overall, line=dict(color=MUTED, width=1, dash="dot"),
                  annotation_text=f"Toàn bộ dữ liệu: {overall:.1f}%",
                  annotation_position="top right", annotation_font_color=INK_2)
    fig.update_yaxes(title="Bình luận chứa teencode (%)", ticksuffix="%",
                     range=[0, max(r["mean"].max() * 100, overall) * 1.3])
    return style(fig, "Tỷ lệ bình luận chứa teencode theo nhãn",
                 "Một bình luận được tính nếu module chuẩn hóa ít nhất một teencode trong câu", height=400)


def fig_category_by_label(df, top_n=12):
    """Các nhóm nội dung (category) phổ biến nhất và nhãn của chúng."""
    top = df["category_clean"].value_counts().head(top_n).index
    d = df[df["category_clean"].isin(top)]
    ct = pd.crosstab(d["category_clean"], d["label_name"]).reindex(columns=list(LABEL_NAMES.values()), fill_value=0)
    ct = ct.loc[ct.sum(axis=1).sort_values().index]
    fig = go.Figure()
    for lab in LABEL_NAMES.values():
        fig.add_bar(y=ct.index, x=ct[lab], name=lab, orientation="h",
                    marker=dict(color=LABEL_COLORS[lab], line=dict(color=SURFACE, width=2)),
                    hovertemplate="%{y}<br>" + lab + ": %{x} câu<extra></extra>")
    fig.update_layout(barmode="stack", legend_traceorder="normal")
    fig.update_xaxes(title="Số bình luận")
    n_cat = df["category_clean"].nunique()
    rest = len(df) - len(d)
    return style(fig, "Mỗi nhóm nội dung gần như chỉ thuộc một nhãn",
                 f"{top_n} nhóm phổ biến nhất; {rest:,} câu còn lại rải rác trong {n_cat - top_n} nhóm nhỏ khác"
                 .replace(",", "."), height=500)


def fig_writing_style(df):
    """Kiểu viết (có dấu / không dấu / khác) theo nhãn."""
    labs = list(LABEL_NAMES.values())
    ct = pd.crosstab(df["label_name"], df["writing_style"]).reindex(index=labs, columns=STYLE_ORDER, fill_value=0)
    pct = ct.div(ct.sum(axis=1), axis=0) * 100
    fig = go.Figure()
    for s in STYLE_ORDER:
        fig.add_bar(
            y=labs, x=pct[s], name={"Khác": "Khác (tiếng Anh, quá ngắn)"}.get(s, s), orientation="h",
            marker=dict(color=STYLE_COLORS[s], line=dict(color=SURFACE, width=2)),
            text=[f"{v:.0f}%" if v >= 4 else "" for v in pct[s]], textposition="inside",
            insidetextanchor="middle", textfont=dict(color=INK if s == "Có dấu" else "white"),
            customdata=ct[s], hovertemplate="%{y} · " + s + ": %{customdata} câu (%{x:.1f}%)<extra></extra>",
        )
    fig.update_layout(barmode="stack", legend_traceorder="normal")
    fig.update_xaxes(title="Tỷ lệ bình luận (%)", range=[0, 100], ticksuffix="%")
    fig.update_yaxes(autorange="reversed")
    neg, pos = pct.loc["Tiêu cực", "Không dấu"], pct.loc["Tích cực", "Không dấu"]
    return style(fig, f"Bình luận không dấu tập trung ở nhãn Tiêu cực ({neg:.0f}% so với {pos:.1f}% ở Tích cực)",
                 "Mô hình có nguy cơ học \"viết không dấu thì là tiêu cực\"; module hiện chưa khôi phục dấu",
                 height=340)


# ----------------------------------------------------------------------------
# 2. Đối sánh trước - sau chuẩn hóa
# ----------------------------------------------------------------------------
def fig_top_teencode(hits, n=20):
    """Các teencode được module thay nhiều nhất và từ chuẩn tương ứng."""
    items = hits.most_common(n)[::-1]
    fig = go.Figure(go.Bar(
        y=[f"{old}  →  {new}" for (old, new), _ in items], x=[v for _, v in items], orientation="h",
        marker=dict(color=VERSION_COLORS["Chuẩn hóa"], cornerradius=4),
        text=[v for _, v in items], textposition="outside", textfont=dict(color=INK_2),
        hovertemplate="%{y}<br>%{x} lần thay thế<extra></extra>",
    ))
    fig.update_xaxes(title="Số lần được module thay thế")
    total = sum(hits.values())
    return style(fig, f"{n} teencode được chuẩn hóa nhiều nhất",
                 f"Module đã thay tổng cộng {total} lần, gồm {len(hits)} dạng teencode khác nhau",
                 height=620)


def word_counts(texts):
    c = Counter()
    for t in texts:
        for w in str(t).lower().split():
            w = w.strip(PUNCT)
            if w:
                c[w] += 1
    return c


def fig_word_shift(df, n=12):
    """Những từ thay đổi tần suất nhiều nhất sau chuẩn hóa (tăng / giảm)."""
    raw, norm = word_counts(df["text_raw_segmented"]), word_counts(df["text_normalized_segmented"])
    diff = {w: norm.get(w, 0) - raw.get(w, 0) for w in set(raw) | set(norm)}
    up = sorted([(w, d) for w, d in diff.items() if d > 0], key=lambda x: -x[1])[:n]
    down = sorted([(w, d) for w, d in diff.items() if d < 0], key=lambda x: x[1])[:n]
    items = sorted(up + down, key=lambda x: x[1])
    fig = go.Figure()
    for name, sel, color in [("Giảm sau chuẩn hóa (teencode bị thay)", [i for i in items if i[1] < 0], VERSION_COLORS["Gốc"]),
                             ("Tăng sau chuẩn hóa (từ chuẩn)", [i for i in items if i[1] > 0], VERSION_COLORS["Chuẩn hóa"])]:
        fig.add_bar(
            y=[w.replace("_", " ") for w, _ in sel], x=[d for _, d in sel], orientation="h", name=name,
            marker=dict(color=color, cornerradius=4),
            customdata=[(raw.get(w, 0), norm.get(w, 0)) for w, _ in sel],
            hovertemplate="%{y}<br>Gốc: %{customdata[0]} → Chuẩn hóa: %{customdata[1]} (%{x:+d})<extra></extra>",
        )
    fig.update_yaxes(categoryorder="array", categoryarray=[w.replace("_", " ") for w, _ in items])
    fig.update_xaxes(title="Chênh lệch số lần xuất hiện (chuẩn hóa − gốc)", zeroline=True,
                     zerolinecolor=AXIS, zerolinewidth=1)
    fig.update_layout(legend_traceorder="normal")
    return style(fig, "Chuẩn hóa gộp các biến thể về cùng một từ chuẩn",
                 f"{n} từ giảm và {n} từ tăng nhiều nhất trên toàn bộ dữ liệu; \"không\" nhận lại các dạng ko, k, hok",
                 height=680)


def fig_missed_teencode(missed, still, examples, tdict, n=15):
    """Teencode dính dấu câu: module v1 bỏ sót bao nhiêu, module hiện tại còn sót bao nhiêu."""
    items = missed.most_common(n)[::-1]
    mapping = dict(tdict)
    try:  # dùng các mục từ điển đã sửa ở module v2 (vd lol -> cười thay vì ánh xạ sai của v1)
        from teencode_normalizer import DICT_FIXES
        mapping.update(DICT_FIXES)
    except ImportError:
        pass
    labels = [f"{examples[k]}  ({k} → {mapping[k]})" for k, _ in items]
    series = [("Module v1 bỏ sót", [v for _, v in items], VERSION_COLORS["Gốc"])]
    if sum(still.values()) > 0:
        series.append(("Module v2 vẫn sót", [still.get(k, 0) for k, _ in items], VERSION_COLORS["Chuẩn hóa"]))
    fig = go.Figure()
    for name, vals, color in series:
        fig.add_bar(y=labels, x=vals, orientation="h", name=name, text=vals, textposition="outside",
                    textfont=dict(color=INK_2), marker=dict(color=color, cornerradius=4),
                    hovertemplate="%{y}<br>" + name + ": %{x} lần<extra></extra>")
    fig.update_layout(barmode="group", legend_traceorder="normal", bargap=0.25, showlegend=len(series) > 1)
    fig.update_xaxes(title="Số lần bị bỏ sót")
    a, b = sum(missed.values()), sum(still.values())
    return style(fig, f"Teencode dính dấu câu: v1 bỏ sót {a} lần, v2 còn sót {b}",
                 "Ví dụ \"vcl,\" không khớp \"vcl\" vì module v1 tách từ theo dấu cách; v2 tách dấu câu trước khi tra từ điển",
                 height=600)


def wordcloud_grid(df, path):
    """Lưới 2 x 3 word cloud (PNG) - Gốc / Chuẩn hóa x 3 nhãn."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud

    font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    try:
        open(font).close()
    except OSError:
        font = None  # máy khác: dùng font mặc định của wordcloud (nên chọn font có dấu tiếng Việt)

    labs = list(LABEL_NAMES.values())
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), facecolor=SURFACE)
    for r, (col, vname) in enumerate([("text_raw_segmented", "Gốc"), ("text_normalized_segmented", "Chuẩn hóa")]):
        for c, lab in enumerate(labs):
            freqs = dict(top_words(df.loc[df["label_name"] == lab, col], 120))
            color = LABEL_COLORS[lab]
            wc = WordCloud(width=600, height=380, background_color=SURFACE, font_path=font,
                           color_func=lambda *a, **k: color, prefer_horizontal=0.95,
                           ).generate_from_frequencies({w.replace("_", " "): v for w, v in freqs.items()})
            ax = axes[r, c]
            ax.imshow(wc, interpolation="bilinear")
            ax.set_title(f"{lab} · {vname}", fontsize=14, color=INK, loc="left")
            ax.axis("off")
    fig.suptitle("Word cloud theo nhãn, trước và sau chuẩn hóa Teencode",
                 fontsize=17, color=INK, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def summary_stats(df, hits, missed, still):
    raw_vocab = {w for t in df["text_raw_segmented"] for w in str(t).split()}
    norm_vocab = {w for t in df["text_normalized_segmented"] for w in str(t).split()}
    return {
        "so_binh_luan": int(len(df)),
        "ty_le_cau_chua_teencode_pct": round(df["is_teencode"].mean() * 100, 2),
        "ty_le_cau_bi_module_thay_doi_pct": round(df["changed"].mean() * 100, 2),
        "so_lan_thay_teencode": int(sum(hits.values())),
        "so_dang_teencode_duoc_thay": len(hits),
        "v1_bo_sot_teencode_dinh_dau_cau": int(sum(missed.values())),
        "v2_van_sot_teencode_dinh_dau_cau": int(sum(still.values())),
        "kich_thuoc_tu_vung_goc": len(raw_vocab),
        "kich_thuoc_tu_vung_chuan_hoa": len(norm_vocab),
        "do_dai_tb_goc": round(df["len_raw"].mean(), 2),
        "do_dai_tb_chuan_hoa": round(df["len_norm"].mean(), 2),
        "ty_le_kieu_viet_pct": {k: round(v * 100, 2) for k, v in df["writing_style"].value_counts(normalize=True).items()},
        "ty_le_khong_dau_theo_nhan_pct": {
            k: round(v * 100, 2) for k, v in
            df.groupby("label_name")["writing_style"].apply(lambda s: (s == "Không dấu").mean()).items()},
    }
