# Nghiên cứu ứng dụng PhoBERT kết hợp Module chuẩn hóa Teencode trong phân loại sắc thái bình luận

Dự án môn học nghiên cứu bài toán Phân loại sắc thái cảm xúc tiếng Việt (Sentiment Analysis) 3 nhãn trên dữ liệu bình luận mạng xã hội. Trọng tâm đề tài tập trung vào việc giảm thiểu hiện tượng phân mảnh từ vựng (Subword Fragmentation) của mô hình PhoBERT thông qua module tiền xử lý Teencode chuyên biệt.

---

## 1. Quy ước nhãn (Label Mapping)

Dữ liệu được chuẩn hóa và phân loại thành 3 lớp rời rạc:
* `0`: **Tiêu cực (Negative)** (bao gồm cả các bình luận toxic, chửi bới, tiêu cực).
* `1`: **Trung tính (Neutral)**.
* `2`: **Tích cực (Positive)**.

---

## 2. Cấu trúc dữ liệu & Thư mục bàn giao

Toàn bộ dữ liệu đã được xử lý phân tầng (Stratified Split 80/10/10) và lưu trữ thành các file CSV:

* `train.csv` (3.916 dòng): Dữ liệu huấn luyện.
* `val.csv` (490 dòng): Dữ liệu đánh giá kiểm định (Validation).
* `test.csv` (490 dòng): Dữ liệu kiểm thử tổng thể.
* `test_teencode.csv` (54 dòng): Tập kiểm thử độc lập chỉ chứa các câu có teencode/từ lóng.
* `teencode_dict.json`: Từ điển ánh xạ teencode chuẩn hóa (365 mục từ).

### Các trường dữ liệu chính trong file CSV:
* `label_id`: Nhãn số nguyên (`0`, `1`, `2`).
* `text_raw_segmented`: Bình luận gốc giữ nguyên teencode, đã tách từ ghép bằng `pyvi` (dành cho mô hình **Baseline**).
* `text_normalized_segmented`: Bình luận đã qua module chuẩn hóa Teencode + Regex rút gọn ký tự lặp, đã tách từ ghép bằng `pyvi` (dành cho mô hình **Đề xuất**).

---

## 3. Chỉ số thực nghiệm tiền xử lý (Ablation Metric)

Hiệu quả giảm phân mảnh từ vựng (Subword) trên PhoBERT Tokenizer (`vinai/phobert-base`):
* **Trước chuẩn hóa:** 7.421 subword phân mảnh (tỷ lệ 17.38%).
* **Sau chuẩn hóa:** 7.092 subword phân mảnh (tỷ lệ 16.69%).
* **Mức độ giảm phân mảnh:** **4.43%** trên toàn bộ kho ngữ liệu (khôi phục 329 token bị băm vụn ngữ nghĩa).

---

## 4. Hướng dẫn phân công & Quy trình phát triển (Git Flow)

* **Nhánh `main`:** Lưu trữ pipeline dữ liệu gốc, chỉ merge khi có kết quả hoàn chỉnh.
* **Nhánh `feat/modeling` (Ní 2 - Hiếu):**
  * Huấn luyện mô hình Baseline trên cột `text_raw_segmented`.
  * Huấn luyện mô hình Đề xuất trên cột `text_normalized_segmented`.
  * Xuất kết quả dự đoán và checkpoint mô hình.
* **Nhánh `feat/evaluation` (Ní 3 - Sơn):**
  * Đo Macro $F_1$-score trên `test.csv` và `test_teencode.csv`.
  * Vẽ biểu đồ phân bố dữ liệu, biểu đồ phân mảnh và Ma trận nhầm lẫn (Confusion Matrix).
  * Thực hiện phân tích sai số định tính (Error Analysis).
