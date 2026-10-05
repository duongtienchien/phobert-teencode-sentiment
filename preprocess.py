import json
import re
import pandas as pd
from pyvi import ViTokenizer
from sklearn.model_selection import train_test_split

print("1. Đang nạp dữ liệu và từ điển...")
df = pd.read_csv("vne_dataset.csv")

with open("teencode_dict.json", "r", encoding="utf-8") as f:
    teencode_dict = json.load(f)

# Ánh xạ nhãn: toxic gộp vào negative (0), neutral (1), positive (2)
label_map = {
    "negative": 0,
    "toxic": 0,
    "neutral": 1,
    "positive": 2
}
df["label_id"] = df["label"].map(label_map)
df = df.dropna(subset=["comment", "label_id"]).copy()
df["label_id"] = df["label_id"].astype(int)

# Module chuẩn hóa Teencode + Rút gọn ký tự lặp
def clean_teencode(text):
    text = str(text).lower()
    # Rút gọn ký tự lặp >= 3 lần (ngonnn -> ngon)
    text = re.sub(r"(\w)\1{2,}", r"\1", text)
    words = text.split()
    normalized_words = [teencode_dict.get(w, w) for w in words]
    return " ".join(normalized_words)

def has_teencode(text):
    words = set(str(text).lower().split())
    return any(w in teencode_dict for w in words)

print("2. Đang chuẩn hóa teencode và tách từ tiếng Việt (PyVi)...")
df["is_teencode"] = df["comment"].apply(has_teencode)

# Cột chuẩn hóa đề xuất (PhoBERT + Tiền xử lý)
df["normalized_text"] = df["comment"].apply(clean_teencode)
df["text_normalized_segmented"] = df["normalized_text"].apply(ViTokenizer.tokenize)

# Cột gốc đối chứng (PhoBERT Baseline)
df["text_raw_segmented"] = df["comment"].astype(str).str.lower().apply(ViTokenizer.tokenize)

print("3. Đang chia tập phân tầng (Stratified Split 80 - 10 - 10)...")
train_df, temp_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["label_id"]
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, random_state=42, stratify=temp_df["label_id"]
)

# Tách riêng tập test teencode cho Ní 3 làm thực nghiệm đối chứng
test_teencode_df = test_df[test_df["is_teencode"] == True].copy()

# Xuất file CSV
train_df.to_csv("train.csv", index=False)
val_df.to_csv("val.csv", index=False)
test_df.to_csv("test.csv", index=False)
test_teencode_df.to_csv("test_teencode.csv", index=False)

print("-" * 45)
print("XỬ LÝ DỮ LIỆU THÀNH CÔNG!")
print(f"Tổng mẫu: {len(df)}")
print(f"- Train:          {len(train_df)} mẫu")
print(f"- Val:            {len(val_df)} mẫu")
print(f"- Test (Full):    {len(test_df)} mẫu")
print(f"- Test (Teencode):{len(test_teencode_df)} mẫu")
print("Đã tạo xong: train.csv, val.csv, test.csv, test_teencode.csv")
print("-" * 45)