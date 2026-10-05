import json
import pandas as pd

# 1. Đọc dữ liệu từ file CSV vừa tải
df = pd.read_csv("vne_dataset.csv")

print("--- THÔNG TIN TỔNG QUAN ---")
print(f"Tổng số dòng: {len(df)}")
print(f"Các cột dữ liệu: {list(df.columns)}")
print("\nPhân bố các nhãn hiện có:")
print(df["label"].value_counts())

# 2. Đọc từ điển teencode
with open("teencode_dict.json", "r", encoding="utf-8") as f:
    teencode_dict = json.load(f)

# 3. Quét kiểm tra tỷ lệ xuất hiện teencode
count_with_teencode = 0
teencode_found = set()

# Đảm bảo xử lý đúng tên cột chứa văn bản (thường là 'comment' hoặc 'text')
text_col = "comment" if "comment" in df.columns else df.columns[0]

for text in df[text_col].dropna():
    words = set(str(text).lower().split())
    matched = [w for w in words if w in teencode_dict]
    if matched:
        count_with_teencode += 1
        teencode_found.update(matched)

# 4. Xuất kết quả
total = len(df)
print("\n--- KẾT QUẢ QUÉT TEENCODE ---")
print(f"Số câu chứa ít nhất 1 từ teencode: {count_with_teencode} ({count_with_teencode / total * 100:.2f}%)")
print(f"Số từ teencode độc bản tìm thấy: {len(teencode_found)}")
print(f"Mẫu các từ bắt gặp: {list(teencode_found)[:20]}")