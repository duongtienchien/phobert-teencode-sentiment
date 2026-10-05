import json
import re
import pandas as pd
from transformers import AutoTokenizer

print("1. Đang tải PhoBERT Tokenizer (vinai/phobert-base)...")
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

# Đọc dữ liệu và từ điển
print("2. Đang đọc dữ liệu...")
df = pd.read_csv("vne_dataset.csv")
with open("teencode_dict.json", "r", encoding="utf-8") as f:
    teencode_dict = json.load(f)

# Hàm chuẩn hóa
def clean_teencode(text):
    text = str(text).lower()
    text = re.sub(r"(\w)\1{2,}", r"\1", text)
    words = text.split()
    return " ".join([teencode_dict.get(w, w) for w in words])

# Hàm đếm số subword bị băm nát (chứa '@@')
def count_subwords(text_list):
    total_tokens = 0
    fragmented_tokens = 0
    
    for text in text_list:
        tokens = tokenizer.tokenize(str(text))
        total_tokens += len(tokens)
        fragmented_tokens += sum(1 for t in tokens if "@@" in t)
        
    return total_tokens, fragmented_tokens

raw_texts = df["comment"].dropna().tolist()
cleaned_texts = [clean_teencode(t) for t in raw_texts]

print("3. Đang đo lường trên tập gốc...")
tot_raw, frag_raw = count_subwords(raw_texts)

print("4. Đang đo lường trên tập sau khi qua module tiền xử lý...")
tot_clean, frag_clean = count_subwords(cleaned_texts)

rate_raw = (frag_raw / tot_raw) * 100
rate_clean = (frag_clean / tot_clean) * 100
reduction = ((frag_raw - frag_clean) / frag_raw) * 100

print("\n" + "=" * 50)
print("BÁO CÁO ĐO LƯỜNG ĐỘ PHÂN MẢNH TỪ VỰNG SUBWORD")
print("=" * 50)
print(f"TRƯỚC CHUẨN HÓA (GỐC):")
print(f"- Tổng số token:           {tot_raw}")
print(f"- Subword bị băm ('@@'):   {frag_raw} ({rate_raw:.2f}%)")
print("-" * 50)
print(f"SAU CHUẨN HÓA (ĐỀ XUẤT):")
print(f"- Tổng số token:           {tot_clean}")
print(f"- Subword bị băm ('@@'):   {frag_clean} ({rate_clean:.2f}%)")
print("-" * 50)
print(f"=> MỨC ĐỘ GIẢM PHÂN MẢNH:   {reduction:.2f}%")
print("=" * 50)