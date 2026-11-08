# Dữ liệu v2 – Data card

Tạo lại toàn bộ bằng một lệnh (từ thư mục gốc repo):

```bash
pip install pandas scikit-learn pyvi lingua-language-detector wordfreq
python preprocess_v2.py          # ~vài giây
python tests/test_normalizer.py  # 49/49 test module chuẩn hóa
```

Nguồn: `vne_dataset.csv` (4.896 bình luận). Seed 42. Mọi câu có `id` cố định (`vne_00001`…), câu tăng cường có `source_id` trỏ về câu gốc.

## Nhãn

`0` Tiêu cực (gồm toxic) · `1` Trung tính · `2` Tích cực

## Làm sạch

| Bước | Số câu |
| --- | --- |
| Dữ liệu gốc | 4.896 |
| Bỏ câu không có chữ | −1 |
| Bỏ câu tiếng Anh thuần | −246 |
| Sửa nhãn: duyệt tay 100 câu (8 câu) + quy tắc "thương cảm → Trung tính" (15 câu) | đổi nhãn 23 |
| Bỏ câu trùng nội dung nhưng khác nhãn | −2 |
| Bỏ câu trùng lặp | −169 |
| **Còn lại** | **4.478** |

Kiểm tra thủ công 100 câu: nhãn sai ~5% trên mẫu ngẫu nhiên. Quy tắc nhãn đã chốt: câu xót xa/tội nghiệp → Trung tính; câu động viên ("cố lên") → Tích cực. Các sửa tay lưu ở `data/manual_label_fixes.csv`.

## Các file

| File | Số câu | Dùng cho |
| --- | --- | --- |
| `train.csv` | 3.582 | Train gốc (không tăng cường) |
| `train_aug.csv` | 10.099 | **Train chính**: gốc + 6.201 bản teencode + 316 bản bỏ dấu (Tích cực/Trung tính) |
| `val.csv` | 448 | Chọn checkpoint (bản sạch) |
| `test.csv` | 448 | Test chính |
| `test_teencode.csv` | 56 | Tập con của test có teencode thật |
| `test_noise_10/30/50.csv` | 448 mỗi file | Test chèn teencode tổng hợp, ~3% / 9% / 15% số từ bị biến đổi |
| `test_khong_dau.csv` | 448 | Test bỏ hết dấu (đo đường tắt "không dấu = tiêu cực") |
| `removed_rows.csv` | 418 | Các câu đã loại + lý do |
| `cleaning_report.json` | – | Toàn bộ số liệu |

Không câu nào xuất hiện ở hai tập (train/val/test), kiểm tra tự động trong pipeline.

## Cột

| Cột | Ý nghĩa |
| --- | --- |
| `text_raw_segmented` | Đầu vào mô hình **Baseline** (chữ thường, tách từ PyVi) |
| `text_normalized_segmented` | Đầu vào mô hình **Đề xuất** (qua module chuẩn hóa v2, tách từ PyVi) |
| `label_id` | Nhãn 0/1/2 |
| `replacements` | Module đã thay những gì, vd `["ko→không", "😂→cười"]` (dùng cho error analysis) |
| `is_teencode` | Câu có ít nhất một teencode được chuẩn hóa |
| `augment`, `source_id`, `n_changes`, `changes` | Thông tin tăng cường / nhiễu |

## Module chuẩn hóa v2 (`teencode_normalizer.py`)

Tách dấu câu khỏi từ, cụm nhiều từ, tiếng cười/emoji/emoticon → từ, số tiền (50k → 50 nghìn), rút ký tự kéo dài (kể cả có dấu: quáaaa → quá), luật biến âm ngược có kiểm tra từ vựng (wá → quá, zui → vui, khôg → không), sửa lỗi từ điển v1 (`lol` → cười thay vì tục, `vd` → video).

Tỷ lệ khôi phục teencode tổng hợp (test nhiễu 50%):

| Loại | v1 | v2 |
| --- | --- | --- |
| Biến thể trong từ điển | 100% | 100% |
| Kéo dài ký tự | 98% | 98% |
| Biến thể bổ sung | 0% | 73% |
| Luật biến âm | 0% | 78% |
| Held-out (cố ý không học) | 0% | 4% |

## Độ phân mảnh từ vựng (PhoBERT tokenizer `vinai/phobert-base`)

Đo bằng `viz/fragmentation.py` trên đúng đầu vào của hai mô hình (đã tách từ), 4.478 câu. Thay cho con số 4,43% cũ (đo trên câu chưa tách từ).

| Tập | Chỉ số | Gốc | Chuẩn hóa | Giảm tương đối |
| --- | --- | --- | --- | --- |
| Toàn bộ dữ liệu | Từ bị tách thành ≥ 2 subword (OOV) | 9,26% | 7,91% | −14,5% |
| Toàn bộ dữ liệu | Subword / từ | 1,112 | 1,097 | −1,3% |
| Toàn bộ dữ liệu | Token "@@" (cách đo cũ) | 10,11% | 8,83% | −12,6% |
| Câu có teencode (621 câu) | Từ bị tách (OOV) | 20,09% | 12,42% | **−38,2%** |
| Câu có teencode (621 câu) | Subword / từ | 1,232 | 1,142 | −7,3% |

**Held-out:** ~30% biến thể và 2 luật biến âm (`ph→f`, `c→k`) chỉ xuất hiện trong tập test, không có trong train tăng cường và KHÔNG được thêm vào từ điển, để đo khả năng tổng quát hóa. Khi sửa từ điển, không thêm các biến thể này.

## Hạn chế đã biết

- Một phần dữ liệu gốc có dấu hiệu sinh theo mẫu (khối câu lặp, câu na ná nhau như "Thương anh lắm, cố lên").
- Teencode thật trong dữ liệu ít (~11% câu); phần lớn teencode trong train/test nhiễu là **tổng hợp**.
- Chưa khôi phục dấu cho câu viết không dấu; tăng cường bỏ dấu chỉ để giảm đường tắt.
- Một số viết tắt cần ngữ cảnh vẫn có thể sai (`mk` = mình/mật khẩu, `vc`).
