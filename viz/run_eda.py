"""
Sinh toàn bộ biểu đồ EDA và đối sánh trước - sau chuẩn hóa của Ní 1.

Chạy từ thư mục gốc của repo:
    python viz/run_eda.py

Kết quả nằm trong viz/outputs/eda/:
    *.html  biểu đồ tương tác (Plotly) - dùng cho Dashboard / mở bằng trình duyệt
    *.png   ảnh tĩnh - chèn vào báo cáo
    *.csv   số liệu gốc của từng biểu đồ
    summary.json  các chỉ số tổng quan
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import eda_charts as ec  # noqa: E402
from teencode_normalizer import DICT_FIXES  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "viz", "outputs", "eda")
os.makedirs(OUT, exist_ok=True)


def save(fig, name, width=1000):
    fig.write_html(os.path.join(OUT, f"{name}.html"), include_plotlyjs="cdn")
    try:
        fig.write_image(os.path.join(OUT, f"{name}.png"), width=width, scale=2)
    except Exception as e:  # thiếu kaleido/Chrome thì vẫn có file HTML
        print(f"   (bỏ qua PNG {name}: {e})")
    print(f"-> {name}")


def main():
    print("1. Đọc dữ liệu và từ điển...")
    df = ec.load_splits(os.path.join(ROOT, "data", "v2"))
    tdict = ec.load_dict(ROOT)
    hits = ec.teencode_hits(df, tdict)
    missed, still, examples = ec.teencode_missed(df, tdict)

    print("2. EDA đa chiều...")
    save(ec.fig_label_by_split(df), "01_phan_bo_nhan_theo_tap")
    save(ec.fig_length_by_label(df), "02_do_dai_theo_nhan", width=900)
    save(ec.fig_teencode_rate(df), "03_ty_le_teencode_theo_nhan", width=800)
    save(ec.fig_category_by_label(df), "04_nhom_noi_dung_theo_nhan")
    save(ec.fig_writing_style(df), "05_kieu_viet_theo_nhan")

    print("3. Đối sánh trước - sau chuẩn hóa...")
    save(ec.fig_top_teencode(hits), "06_top_teencode")
    save(ec.fig_word_shift(df), "07_tu_tang_giam_sau_chuan_hoa")
    ec.wordcloud_grid(df, os.path.join(OUT, "08_wordcloud_truoc_sau.png"))
    print("-> 08_wordcloud_truoc_sau")
    save(ec.fig_missed_teencode(missed, still, examples, tdict), "09_teencode_bi_sot")
    df[df["writing_style"] == "Không dấu"][["comment", "label", "label_id", "split"]].to_csv(
        os.path.join(OUT, "binh_luan_khong_dau.csv"), index=False, encoding="utf-8-sig")

    print("4. Xuất số liệu...")
    pd.DataFrame(
        [(old, new, v) for (old, new), v in hits.most_common()], columns=["teencode", "chuan_hoa", "so_lan"]
    ).to_csv(os.path.join(OUT, "teencode_duoc_thay.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [(k, examples[k], {**tdict, **DICT_FIXES}[k], v, still.get(k, 0)) for k, v in missed.most_common()],
        columns=["teencode", "vi_du_trong_cau", "chuan_hoa", "v1_bo_sot", "v2_van_sot"],
    ).to_csv(os.path.join(OUT, "teencode_bi_sot.csv"), index=False, encoding="utf-8-sig")
    stats = ec.summary_stats(df, hits, missed, still)
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    for k, v in stats.items():
        print(f"   {k}: {v}")
    print(f"\nXong! Kết quả trong {OUT}")


if __name__ == "__main__":
    main()
