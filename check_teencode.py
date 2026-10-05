import json
from datasets import load_dataset

# 1. Tải tập dữ liệu từ Hugging Face
print("Đang tải dữ liệu...")
ds = load_dataset("vanhai123/vietnamese-social-comments")
train_data = ds["train"]

# 2. Đọc từ điển teencode
with open("teencode_dict.json", "r", encoding="utf-8") as f:
    teencode_dict = json.load(f)

# 3. Quét kiểm tra tỷ lệ teencode
count_with_teencode = 0
total = len(train_data)
teencode_found = set()

for row in train_data:
    words = set(str(row["comment"]).lower().split())
    matched = [w for w in words if w in teencode_dict]
    if matched:
        count_with_teencode += 1
        teencode_found.update(matched)

# 4. Xuất kết quả thống kê
print("-" * 40)
print(f"Tổng số mẫu trong tập train: {total}")
print(f"Số câu chứa teencode: {count_with_teencode} ({count_with_teencode / total * 100:.2f}%)")
print(f"Số lượng từ teencode độc bản bắt gặp: {len(teencode_found)}")
print(f"Một số teencode xuất hiện thực tế: {list(teencode_found)[:15]}")
print("-" * 40)