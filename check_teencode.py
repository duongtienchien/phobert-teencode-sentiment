import json
import pandas as pd

# 1. Doc du lieu train co san trong du an
print("Dang doc du lieu train.csv...")
train_data = pd.read_csv("train.csv", encoding="utf-8-sig")

# 2. Doc tu dien teencode
with open("teencode_dict.json", "r", encoding="utf-8") as f:
    teencode_dict = json.load(f)

# Chuan hoa cac tu trong tu dien
teencode_dict = {
    str(key).lower(): value
    for key, value in teencode_dict.items()
}

# 3. Quet kiem tra teencode
count_with_teencode = 0
total = len(train_data)
teencode_found = set()

for comment in train_data["comment"].fillna("").astype(str):
    words = set(comment.lower().split())
    matched = [word for word in words if word in teencode_dict]

    if matched:
        count_with_teencode += 1
        teencode_found.update(matched)

# 4. In ket qua
print("-" * 45)
print("KET QUA KIEM TRA TEENCODE")
print("-" * 45)
print(f"Tong so mau train: {total}")

if total > 0:
    rate = count_with_teencode / total * 100
    print(f"So cau chua teencode: {count_with_teencode} ({rate:.2f}%)")
else:
    print("Tap train khong co du lieu.")

print(f"So tu teencode khac nhau tim thay: {len(teencode_found)}")
print("Mot so tu teencode tim thay:")

for word in sorted(teencode_found)[:20]:
    print(f"- {word} -> {teencode_dict[word]}")

print("-" * 45)
print("DA KIEM TRA XONG!")