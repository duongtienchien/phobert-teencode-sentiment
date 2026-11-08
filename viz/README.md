# Ní 1 – Khám phá dữ liệu & Trực quan hóa tiền xử lý

Phần việc của Ní 1 cho môn Phân Tích Trực Quan Hóa Dữ Liệu. Thư mục này dùng chung dữ liệu (`train/val/test.csv`, `teencode_dict.json`) với môn Học Máy Nâng Cao.

## Cài đặt

```bash
pip install pandas plotly kaleido wordcloud matplotlib transformers
```

## Chạy (từ thư mục gốc của repo)

```bash
python viz/run_eda.py         # EDA + đối sánh trước/sau chuẩn hóa  -> viz/outputs/eda/
python viz/fragmentation.py   # Độ phân mảnh PhoBERT (cần mạng tải tokenizer) -> viz/outputs/fragmentation/
```

Mỗi biểu đồ được xuất 2 dạng: `.html` (tương tác, mở bằng trình duyệt) và `.png` (chèn báo cáo).

## Danh sách biểu đồ

| File | Nội dung |
| --- | --- |
| 01_phan_bo_nhan_theo_tap | Tỷ lệ nhãn trên Train/Validation/Test (kiểm chứng chia phân tầng) |
| 02_do_dai_theo_nhan | Phân bố số tiếng mỗi bình luận theo nhãn |
| 03_ty_le_teencode_theo_nhan | Tỷ lệ bình luận chứa teencode theo nhãn |
| 04_nhom_noi_dung_theo_nhan | 12 nhóm nội dung (`category`) phổ biến nhất và nhãn của chúng |
| 05_kieu_viet_theo_nhan | Có dấu / không dấu / khác theo nhãn |
| 06_top_teencode | Teencode được chuẩn hóa nhiều nhất (ko → không …) |
| 07_tu_tang_giam_sau_chuan_hoa | Từ tăng/giảm tần suất nhiều nhất sau chuẩn hóa |
| 08_wordcloud_truoc_sau | Word cloud 3 nhãn × trước/sau (chỉ PNG) |
| 09_teencode_bi_sot | Teencode bị sót vì dính dấu câu (đầu vào cho từ điển v2) |
| 10_phan_manh_tong_quan | Tỷ lệ từ bị tách và subword/từ, trước/sau |
| 11_teencode_bi_tach | Mỗi teencode bị tokenizer tách ra sao, trước/sau |
| 12_phan_manh_tung_cau | Phân bố mức phân mảnh từng câu có teencode |

File số liệu đi kèm: `summary.json`, `teencode_duoc_thay.csv`, `teencode_bi_sot.csv`, `binh_luan_khong_dau.csv`, `phan_manh_tong_hop.csv`.

## Dùng lại trong Dashboard (Ní 3)

```python
import sys; sys.path.append("viz")
import eda_charts as ec
df = ec.load_splits(".")
st.plotly_chart(ec.fig_writing_style(df), use_container_width=True)
```

Màu thống nhất cho cả nhóm nằm ở đầu `eda_charts.py`: `LABEL_COLORS` (Tiêu cực đỏ, Trung tính xám, Tích cực xanh) và `VERSION_COLORS` (Gốc cam, Chuẩn hóa tím).

## Ghi chú phương pháp

- Đo phân mảnh trên đúng cột đầu vào của mô hình (`text_raw_segmented`, `text_normalized_segmented`, đã tách từ bằng PyVi), nên số liệu sẽ khác bản cũ trong `measure_fragmentation.py` (đo trên câu chưa tách từ).
- "Không dấu" được nhận diện gần đúng: câu không có ký tự dấu tiếng Việt và ≥ 80% từ có dạng âm tiết tiếng Việt.
