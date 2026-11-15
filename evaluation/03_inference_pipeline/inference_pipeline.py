"""
evaluation/03_inference_pipeline/inference_pipeline.py
-------------------------------------------------------
Nhanh feat/evaluation (Ni 3 - Son)
Inference pipeline: cau tho -> teencode_normalizer -> pyvi -> PhoBERT -> nhan
Chay thu nghiem de demo luong xu ly, khong can GPU (chi trich xuat logic).
"""

import os, sys, re
import pandas as pd

BASE = r"d:\Ky 7\Hoc may nang cao\phobert-teencode-sentiment"
sys.path.insert(0, BASE)
OUT  = fr"{BASE}\evaluation\03_inference_pipeline"
os.makedirs(OUT, exist_ok=True)

LABEL_MAP = {0: "Negative (Tieu cuc)", 1: "Neutral (Trung tinh)", 2: "Positive (Tich cuc)"}

# ===========================================================================
# STEP 1: Teencode Normalization (dung module cua du an)
# ===========================================================================
def step1_normalize(text: str, normalizer=None) -> tuple[str, list]:
    """
    Chuan hoa teencode qua teencode_normalizer.py.
    Tra ve (normalized_text, replacements_list).
    """
    if normalizer is None:
        try:
            from teencode_normalizer import normalize
            normalized, reps = normalize(text)
            return normalized, reps
        except ImportError:
            pass

    # Fallback: ap dung mot so quy tac co ban tu README
    replacements = []
    result = text.lower()

    BASIC_MAP = {
        r"\bko\b": "khong", r"\bk\b": "khong", r"\bkg\b": "khong",
        r"\bbt\b": "binh thuong", r"\bchac\b": "chac chan",
        r"\bmk\b": "minh", r"\btui\b": "toi", r"\bbn\b": "ban",
        r"\bdc\b": "duoc", r"\bthik\b": "thich", r"\blm\b": "lam",
        r"\bvs\b": "voi", r"\bhok\b": "khong", r"\bpk\b": "phai khong",
        r"\bfb\b": "facebook", r"\btx\b": "thuong xuyen",
    }
    for pat, repl in BASIC_MAP.items():
        m = re.search(pat, result, re.IGNORECASE)
        if m:
            original = m.group()
            result = re.sub(pat, repl, result, flags=re.IGNORECASE)
            replacements.append(f"{original}→{repl}")

    # Rut gon ky tu lap (aaaa -> a, hehe -> hehe)
    result = re.sub(r'(.)\1{2,}', r'\1\1', result)

    return result, replacements


# ===========================================================================
# STEP 2: Tach tu PyVi
# ===========================================================================
def step2_tokenize_pyvi(text: str) -> str:
    """Tach tu bang PyVi."""
    try:
        from pyvi import ViTokenizer
        return ViTokenizer.tokenize(text)
    except ImportError:
        return text  # fallback: giu nguyen


# ===========================================================================
# STEP 3: PhoBERT Inference
# ===========================================================================
def step3_phobert_inference(text_segmented: str, model=None, tokenizer=None):
    """
    Chay inference PhoBERT.
    Neu khong co model (env khong co GPU/transformers), tra ve None.
    """
    if model is None or tokenizer is None:
        return None, None  # PENDING — can GPU

    try:
        import torch
        inputs = tokenizer(
            text_segmented, return_tensors="pt",
            truncation=True, max_length=256, padding=True
        )
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]
        pred  = int(probs.argmax())
        conf  = float(probs[pred])
        return pred, conf
    except Exception as e:
        return None, None


# ===========================================================================
# FULL PIPELINE
# ===========================================================================
def run_pipeline(raw_text: str, model=None, tokenizer=None) -> dict:
    """
    Chay toan bo pipeline tu cau tho -> nhan du doan.
    Returns dict ghi lai ket qua tung buoc.
    """
    result = {
        "input_raw"       : raw_text,
        "step1_normalized": None,
        "step1_replacements": [],
        "step2_segmented" : None,
        "step3_label_id"  : None,
        "step3_label_name": None,
        "step3_confidence": None,
        "pipeline_ok"     : False,
    }

    # Step 1
    norm, reps = step1_normalize(raw_text)
    result["step1_normalized"]    = norm
    result["step1_replacements"]  = reps

    # Step 2
    segmented = step2_tokenize_pyvi(norm)
    result["step2_segmented"] = segmented

    # Step 3
    label_id, conf = step3_phobert_inference(segmented, model, tokenizer)
    result["step3_label_id"]   = label_id
    result["step3_label_name"] = LABEL_MAP.get(label_id, "PENDING (can GPU)") if label_id is not None else "PENDING (can GPU)"
    result["step3_confidence"] = conf
    result["pipeline_ok"]      = label_id is not None

    return result


# ===========================================================================
# DEMO: Chay tren cac vi du mau
# ===========================================================================
def run_demo():
    print("=" * 65)
    print(" INFERENCE PIPELINE DEMO")
    print(" (cau tho -> normalize -> pyvi -> PhoBERT -> nhan)")
    print("=" * 65)

    # Lay 10 vi du tu test set (mix cac loai)
    sample_texts = [
        "ko thik cai nay chut nao het",
        "bai nay hay that su, cam on ban nhieu lam",
        "binh thuong thoi, khong co gi dac biet",
        "may co biet cai do nhin nham ra sao ko?",
        "tui lam het roi, dc roi nhe",
        "reaction cho co, chang co tam gi het",
        "san pham ok lam, giao hang nhanh, se mua lai",
        "chat luong te qua, that vong",
        "mk thay cung duoc, khong qua tot cung khong qua xau",
        "slow burn",
    ]

    rows = []
    for text in sample_texts:
        r = run_pipeline(text)
        rows.append(r)
        print(f"\n  Input    : {r['input_raw']}")
        print(f"  Normalize: {r['step1_normalized']}")
        if r["step1_replacements"]:
            print(f"  Replace  : {r['step1_replacements']}")
        print(f"  Segment  : {r['step2_segmented']}")
        print(f"  Label    : {r['step3_label_name']}")

    df = pd.DataFrame(rows)
    out = fr"{OUT}\pipeline_demo_output.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n  -> Saved demo output: {out}")
    print("\n  NOTE: Step 3 (PhoBERT) PENDING — can chay tren may co GPU/transformers")
    print("        Ket noi voi model checkpoint cua Ni 2 de chay day du.")


if __name__ == "__main__":
    run_demo()
    print("\nPipeline schema:")
    print("  [RAW TEXT]")
    print("      |")
    print("      v")
    print("  [teencode_normalizer.py] -- trich xuat replacements")
    print("      |")
    print("      v")
    print("  [pyvi ViTokenizer] -- tach tu")
    print("      |")
    print("      v")
    print("  [PhoBERT (vinai/phobert-base)] -- classification head")
    print("      |")
    print("      v")
    print("  [LABEL: 0=Negative / 1=Neutral / 2=Positive] + confidence")
