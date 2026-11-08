"""
Pipeline dữ liệu v2 (Ní 1).

    pip install pandas scikit-learn pyvi lingua-language-detector wordfreq
    python preprocess_v2.py

Đọc vne_dataset.csv + teencode_dict.json, ghi toàn bộ kết quả vào data/v2/
(KHÔNG ghi đè train.csv / val.csv / test.csv của v1).

Bước 1 - Làm sạch & chia tập lại
    1.1  Gán id cố định theo thứ tự dòng gốc (vne_00001 ...)
    1.2  Bỏ câu rỗng / không có chữ
    1.3  Bỏ câu tiếng Anh thuần (tự động, chỉ loại khi chắc chắn)
    1.4  Bỏ câu trùng lặp (cùng nội dung sau khi bỏ dấu câu, số, khoảng trắng);
         nếu các bản trùng mang nhãn khác nhau -> bỏ cả nhóm
    1.5  Áp dụng quyết định duyệt tay trong data/manual_overrides.csv (nếu có)
    1.6  Chia phân tầng 80/10/10 (seed 42) và kiểm tra rò rỉ train/val/test = 0
Bước 4 - Dữ liệu tăng cường & tập test nhiễu (dùng teencode_generator.py)
    train_aug.csv       train gốc + 1 bản teencode cho mỗi câu (chỉ biến thể KHÔNG held-out)
                        + bản bỏ dấu cho câu Tích cực/Trung tính để cân bằng tỷ lệ "không dấu"
    test_noise_10/30/50 toàn bộ test, nhiễu teencode mức 10% / 30% / 50% (có cả biến thể held-out)
    test_khong_dau.csv  toàn bộ test, bỏ hết dấu
    Val giữ nguyên bản sạch để chọn checkpoint.

Bước 3 - Module chuẩn hóa v2: teencode_normalizer.py (unit test: python tests/test_normalizer.py)
"""
import hashlib
import json
import os
import re
import unicodedata

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
OUT = os.path.join("data", "v2")
OVERRIDES = os.path.join("data", "manual_overrides.csv")
LABEL_MAP = {"negative": 0, "toxic": 0, "neutral": 1, "positive": 2}

VN_DIACRITIC = re.compile(r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]", re.I)
# Từ tiếng Việt không dấu dễ bị tưởng là tiếng Anh -> luôn coi là tiếng Việt
VN_ALWAYS = {"dit", "da", "b", "ko", "k", "dc", "vs", "ns", "mn", "thg", "dm", "vl", "cl"}


# ---------------------------------------------------------------------------
# Tiện ích
# ---------------------------------------------------------------------------
def nfc(text):
    return unicodedata.normalize("NFC", str(text))


def strip_accents(text):
    text = text.replace("đ", "d").replace("Đ", "D")
    return unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode()


def dedup_key(text):
    """Khóa so trùng: chữ thường, chỉ giữ chữ cái, gộp khoảng trắng."""
    text = nfc(text).lower()
    text = re.sub(r"[^\w\s]|\d|_", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# 1.3 Nhận diện câu tiếng Anh thuần
# ---------------------------------------------------------------------------
class EnglishDetector:
    """
    Một câu bị coi là tiếng Anh thuần khi ĐỒNG THỜI:
      - không có ký tự dấu tiếng Việt,
      - không chứa từ nào chỉ có trong tiếng Việt (âm tiết lấy từ các câu có dấu
        trong chính bộ dữ liệu, sau khi bỏ dấu, trừ các từ cũng phổ biến trong tiếng Anh),
      - không chứa teencode trong từ điển,
      - mô hình lingua chấm >= 0.9 là tiếng Anh.
    Câu lưng chừng (0.5 <= điểm < 0.9) được GIỮ và đưa vào danh sách duyệt tay.
    """

    def __init__(self, texts, teencode_dict):
        from lingua import Language, LanguageDetectorBuilder
        from wordfreq import zipf_frequency

        self.en, self.vi = Language.ENGLISH, Language.VIETNAMESE
        self.det = LanguageDetectorBuilder.from_languages(self.en, self.vi).build()
        accented = [t for t in texts if VN_DIACRITIC.search(t)]
        syllables = {strip_accents(w) for t in accented for w in re.findall(r"[^\W\d_]+", t.lower())}
        self.vn_only = {w for w in syllables if zipf_frequency(w, "en") < 3.0} | VN_ALWAYS
        self.teencode = set(teencode_dict)

    def score(self, text):
        """Trả về (điểm tiếng Anh, có phải ứng viên không)."""
        if VN_DIACRITIC.search(text):
            return 0.0, False
        words = re.findall(r"[a-z]+", text.lower())
        if not words or any(w in self.vn_only or w in self.teencode for w in words):
            return 0.0, False
        conf = {c.language: c.value for c in self.det.compute_language_confidence_values(text)}
        return conf.get(self.en, 0.0), True


# ---------------------------------------------------------------------------
# 1.4a Sửa nhãn
# ---------------------------------------------------------------------------
LABEL_FIXES = os.path.join("data", "manual_label_fixes.csv")
# Quy tắc nhóm đã chốt (07/11): câu THƯƠNG CẢM / XÓT XA đang gán Tích cực -> Trung tính.
# Câu ĐỘNG VIÊN ("thương anh lắm, cố lên") vẫn là Tích cực. Không đụng tới câu Tiêu cực.
PITY_RE = re.compile(r"xót|tội nghiệp|tội ông|tội anh|tội chị|tội bà|tội cho|bệnh nặng|bệnh tâm thần"
                     r"|lặng lẽ mà thương|thương ông này|nhìn clip mà thương|nhìn anh mà thương", re.I)
ENCOURAGE_RE = re.compile(r"cố lên|cố gắng|vượt qua|stay strong|ủng hộ", re.I)


def fix_labels(df):
    df = df.copy()
    stats = {}
    if os.path.exists(LABEL_FIXES):
        fx = pd.read_csv(LABEL_FIXES).set_index("id")["new_label"]
        hit = df["id"].isin(fx.index)
        df.loc[hit, "label"] = df.loc[hit, "id"].map(fx)
        stats["Sửa nhãn theo duyệt tay"] = int(hit.sum())
    rule = (df["label"] == "positive") & df["comment"].str.contains(PITY_RE) & ~df["comment"].str.contains(ENCOURAGE_RE)
    df.loc[rule, "label"] = "neutral"
    stats["Sửa nhãn theo quy tắc thương cảm → Trung tính"] = int(rule.sum())
    df["label_id"] = df["label"].map(LABEL_MAP)
    return df, stats


# ---------------------------------------------------------------------------
# Module chuẩn hóa
# ---------------------------------------------------------------------------
def make_normalizer_v1(teencode_dict):
    """Module v1 (giữ lại chỉ để so sánh trong báo cáo)."""
    def normalize(text):
        text = str(text).lower()
        text = re.sub(r"(\w)\1{2,}", r"\1", text)
        return " ".join(teencode_dict.get(w, w) for w in text.split())

    def details(text):
        out = normalize(text)
        return out, [w for w in set(str(text).lower().split()) if w in teencode_dict]

    return normalize, details


def make_normalizer(train_texts):
    """Module v2 (teencode_normalizer.py). Từ vựng kiểm tra luật biến âm chỉ lấy từ TRAIN."""
    from teencode_normalizer import TeencodeNormalizer

    norm = TeencodeNormalizer("teencode_dict.json", lexicon_texts=train_texts)
    cache = {}

    def details(text):
        text = str(text)
        if text not in cache:
            cache[text] = norm.normalize(text)
        return cache[text]

    def normalize(text):
        return details(text)[0]

    return normalize, details


def is_teencode_change(old):
    """Thay đổi tính là teencode khi từ gốc là chữ (không tính emoji, emoticon, số tiền)."""
    return bool(re.fullmatch(r"[^\W\d_]+", old))


# ---------------------------------------------------------------------------
# Bước 1
# ---------------------------------------------------------------------------
def clean(df, teencode_dict, log):
    removed = []

    def drop(mask, reason):
        nonlocal df
        r = df[mask].copy()
        r["reason"] = reason
        removed.append(r)
        df = df[~mask].copy()
        log.append({"buoc": reason, "so_cau_bi_loai": int(mask.sum()), "con_lai": len(df)})

    log.append({"buoc": "Dữ liệu gốc", "so_cau_bi_loai": 0, "con_lai": len(df)})

    # 1.2 câu rỗng / không có chữ / nhãn không hợp lệ
    df["key"] = df["comment"].map(dedup_key)
    drop(df["label_id"].isna() | (df["key"] == ""), "Rỗng, không có chữ hoặc nhãn không hợp lệ")

    # 1.3 tiếng Anh thuần
    det = EnglishDetector(df["comment"].tolist(), teencode_dict)
    scored = df["comment"].map(det.score)
    df["en_score"] = [s for s, _ in scored]
    candidate = pd.Series([c for _, c in scored], index=df.index)
    review = df[candidate & (df["en_score"] >= 0.5) & (df["en_score"] < 0.9)].copy()
    drop(candidate & (df["en_score"] >= 0.9), "Tiếng Anh thuần")

    # 1.4a sửa nhãn (duyệt tay + quy tắc thương cảm), làm TRƯỚC khi so trùng
    df, fixes = fix_labels(df)
    for name, n in fixes.items():
        log.append({"buoc": name, "so_cau_bi_loai": 0, "so_cau_sua_nhan": n, "con_lai": len(df)})

    # 1.4 trùng lặp
    n_labels = df.groupby("key")["label_id"].transform("nunique")
    drop(n_labels > 1, "Trùng nội dung nhưng khác nhãn (bỏ cả nhóm)")
    drop(df.duplicated("key", keep="first"), "Trùng lặp (giữ bản đầu tiên)")

    # 1.5 quyết định duyệt tay
    if os.path.exists(OVERRIDES):
        ov = pd.read_csv(OVERRIDES)
        remove_ids = set(ov.loc[ov["action"].str.lower() == "remove", "id"])
        drop(df["id"].isin(remove_ids), "Loại theo duyệt tay")

    removed = pd.concat(removed) if removed else pd.DataFrame()
    return df, removed, review


def split(df):
    train, temp = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df["label_id"])
    val, test = train_test_split(temp, test_size=0.5, random_state=SEED, stratify=temp["label_id"])
    return train.copy(), val.copy(), test.copy()


def check_leakage(splits):
    """Không câu nào (theo khóa so trùng) được xuất hiện ở hai tập khác nhau."""
    names = list(splits)
    result = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            n = len(set(splits[a]["key"]) & set(splits[b]["key"]))
            result[f"{a} ∩ {b}"] = n
    assert all(v == 0 for v in result.values()), f"Rò rỉ dữ liệu giữa các tập: {result}"
    return result


def add_features(df, details):
    from pyvi import ViTokenizer

    res = df["comment"].map(details)
    df["normalized_text"] = res.map(lambda r: r[0])
    df["replacements"] = res.map(lambda r: json.dumps([f"{a}→{b}" for a, b in r[1]], ensure_ascii=False))
    df["is_teencode"] = res.map(lambda r: any(is_teencode_change(a) for a, _ in r[1]))
    df["text_normalized_segmented"] = df["normalized_text"].map(ViTokenizer.tokenize)
    df["text_raw_segmented"] = df["comment"].astype(str).str.lower().map(ViTokenizer.tokenize)
    return df


# ---------------------------------------------------------------------------
# Bước 4: tăng cường dữ liệu & tập test nhiễu
# ---------------------------------------------------------------------------
NOISE_LEVELS = (0.1, 0.3, 0.5)
TRAIN_NOISE_RANGE = (0.2, 0.5)
N_TEENCODE_COPIES = 2  # số bản teencode tối đa cho mỗi câu train


def _seed(row_id, salt=""):
    return int(hashlib.md5(f"{row_id}{salt}".encode()).hexdigest(), 16) % (2 ** 32)


def has_letters_no_accent(text):
    return bool(re.search(r"[a-zA-Z]", text)) and not VN_DIACRITIC.search(text)


def augment_train(train, gen):
    """Bản teencode cho mọi câu + bản bỏ dấu cho Tích cực/Trung tính."""
    import random

    rows = []
    for r in train.itertuples(index=False):
        seen = {str(r.comment).lower()}
        for k in range(N_TEENCODE_COPIES):
            # Thử tối đa 5 lần, tăng dần mức nhiễu, cho tới khi được một bản mới khác các bản trước
            base_p = random.Random(_seed(r.id, f"p{k}")).uniform(*TRAIN_NOISE_RANGE)
            for attempt in range(5):
                p = min(0.9, base_p + 0.15 * attempt)
                noisy, changes = gen.noisify(r.comment, p=p, mode="train", seed=_seed(r.id, f"{k}-{attempt}"))
                if changes and noisy not in seen:
                    seen.add(noisy)
                    rows.append({**r._asdict(), "id": f"{r.id}_tc{k + 1}", "comment": noisy,
                                 "source_id": r.id, "augment": "teencode", "n_changes": len(changes)})
                    break

    # Cân bằng "không dấu": đưa tỷ lệ ở nhãn 1, 2 lên bằng nhãn 0
    nodau = train["comment"].map(has_letters_no_accent)
    target = nodau[train["label_id"] == 0].mean()
    for lab in (1, 2):
        sub = train[train["label_id"] == lab]
        need = int(round(target * len(sub))) - int(nodau[sub.index].sum())
        pool = sub[~nodau[sub.index] & sub["comment"].str.contains(VN_DIACRITIC)]
        pick = pool.sample(n=max(0, min(need, len(pool))), random_state=SEED)
        for r in pick.itertuples(index=False):
            rows.append({**r._asdict(), "id": f"{r.id}_kd", "comment": strip_accents(r.comment),
                         "source_id": r.id, "augment": "khong_dau", "n_changes": 0})

    aug = pd.DataFrame(rows)
    base = train.assign(source_id=train["id"], augment="goc", n_changes=0)
    return pd.concat([base, aug], ignore_index=True), {
        "ty_le_khong_dau_muc_tieu_pct": round(float(target) * 100, 2),
        "so_ban_teencode": int((aug["augment"] == "teencode").sum()),
        "so_ban_khong_dau": int((aug["augment"] == "khong_dau").sum()),
    }


def noise_test(test, gen, p):
    rows, n_tokens, n_changed = [], 0, 0
    for r in test.itertuples(index=False):
        noisy, changes = gen.noisify(r.comment, p=p, mode="test", seed=_seed(r.id, p))
        n_tokens += len(str(r.comment).split())
        n_changed += len(changes)
        rows.append({**r._asdict(), "comment": noisy, "source_id": r.id, "augment": f"noise_{int(p * 100)}",
                     "n_changes": len(changes),
                     "changes": json.dumps([f"{c['goc']}→{c['moi']}" for c in changes], ensure_ascii=False)})
    return pd.DataFrame(rows), {"muc_nhieu_p": p, "ty_le_tu_bi_bien_doi_pct": round(100 * n_changed / n_tokens, 2),
                                "ty_le_cau_co_bien_doi_pct": round(100 * sum(r["n_changes"] > 0 for r in rows) / len(rows), 2)}


def recovery_rate(test, gen, normalize):
    """Module chuẩn hóa khôi phục được bao nhiêu % biến thể sinh ra, tách theo loại nhiễu."""
    stat = {}
    for r in test.itertuples(index=False):
        _, changes = gen.noisify(r.comment, p=0.5, mode="test", seed=_seed(r.id, "rec"))
        for c in changes:
            kind = "held-out (chưa từng thấy)" if c["held_out"] else (
                "luật biến âm" if c["loai"].startswith("luật") else c["loai"])
            s = stat.setdefault(kind, [0, 0])
            s[0] += normalize(c["moi"]) == c["goc"]
            s[1] += 1
    return {k: {"khoi_phuc_pct": round(100 * a / n, 2), "so_bien_the": n} for k, (a, n) in stat.items()}


COLUMNS = ["id", "comment", "label", "label_id", "category", "is_teencode", "replacements",
           "normalized_text", "text_raw_segmented", "text_normalized_segmented"]
AUG_COLUMNS = COLUMNS + ["source_id", "augment", "n_changes"]


def main():
    os.makedirs(OUT, exist_ok=True)
    print("1. Nạp dữ liệu...")
    df = pd.read_csv("vne_dataset.csv")
    df.insert(0, "id", [f"vne_{i:05d}" for i in range(1, len(df) + 1)])
    df["comment"] = df["comment"].fillna("").map(nfc).str.strip()
    df["label_id"] = df["label"].map(LABEL_MAP)
    with open("teencode_dict.json", encoding="utf-8") as f:
        teencode_dict = json.load(f)

    print("2. Làm sạch...")
    log = []
    df, removed, review = clean(df, teencode_dict, log)
    df["label_id"] = df["label_id"].astype(int)

    print("3. Chia tập phân tầng 80/10/10 và kiểm tra rò rỉ...")
    train, val, test = split(df)
    leakage = check_leakage({"train": train, "val": val, "test": test})

    print("4. Tăng cường dữ liệu và tạo tập test nhiễu...")
    from teencode_generator import TeencodeGenerator

    gen = TeencodeGenerator("teencode_dict.json")
    train_aug, aug_stats = augment_train(train, gen)
    noise_sets, noise_stats = {}, []
    for p in NOISE_LEVELS:
        d, st = noise_test(test, gen, p)
        noise_sets[f"test_noise_{int(p * 100)}"] = d
        noise_stats.append(st)
    test_kd = test.assign(comment=test["comment"].map(strip_accents), source_id=test["id"],
                          augment="khong_dau", n_changes=0)

    print("5. Chuẩn hóa và tách từ...")
    normalize, details = make_normalizer(train["comment"].tolist())
    for d in (train, val, test, train_aug, test_kd, *noise_sets.values()):
        add_features(d, details)
    test_teencode = test[test["is_teencode"]]
    normalize_v1, _ = make_normalizer_v1(teencode_dict)
    recovery = {"module_v1": recovery_rate(test, gen, normalize_v1),
                "module_v2": recovery_rate(test, gen, normalize)}

    print("6. Ghi kết quả...")
    for name, d in [("train", train), ("val", val), ("test", test), ("test_teencode", test_teencode)]:
        d[COLUMNS].to_csv(os.path.join(OUT, f"{name}.csv"), index=False, encoding="utf-8")
    train_aug[AUG_COLUMNS].to_csv(os.path.join(OUT, "train_aug.csv"), index=False, encoding="utf-8")
    test_kd[AUG_COLUMNS].to_csv(os.path.join(OUT, "test_khong_dau.csv"), index=False, encoding="utf-8")
    for name, d in noise_sets.items():
        d[AUG_COLUMNS + ["changes"]].to_csv(os.path.join(OUT, f"{name}.csv"), index=False, encoding="utf-8")
    removed[["id", "comment", "label", "reason"]].to_csv(
        os.path.join(OUT, "removed_rows.csv"), index=False, encoding="utf-8-sig")
    review[["id", "comment", "label", "en_score"]].assign(action="keep", note="").to_csv(
        os.path.join(OUT, "review_language.csv"), index=False, encoding="utf-8-sig")

    dist = lambda d: d["label_id"].value_counts().sort_index().to_dict()  # noqa: E731
    report = {
        "seed": SEED,
        "cac_buoc_lam_sach": log,
        "so_cau": {"train": len(train), "val": len(val), "test": len(test), "test_teencode": len(test_teencode)},
        "phan_bo_nhan": {"train": dist(train), "val": dist(val), "test": dist(test),
                         "test_teencode": dist(test_teencode)},
        "kiem_tra_ro_ri": leakage,
        "so_cau_can_duyet_ngon_ngu": len(review),
        "bo_sinh_teencode": gen.stats(),
        "tang_cuong_train": {**aug_stats, "tong_so_cau_train_aug": len(train_aug),
                             "phan_bo_nhan": dist(train_aug)},
        "tap_test_nhieu": noise_stats,
        "ty_le_khoi_phuc_cua_module": recovery,
    }
    with open(os.path.join(OUT, "cleaning_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("-" * 50)
    for s in log:
        if "so_cau_sua_nhan" in s:
            print(f"  {s['buoc']:<45} ~{s['so_cau_sua_nhan']:>4}  (đổi nhãn)")
        else:
            print(f"  {s['buoc']:<45} -{s['so_cau_bi_loai']:>4}  -> {s['con_lai']}")
    print(f"  Train/Val/Test/Test_teencode: {len(train)}/{len(val)}/{len(test)}/{len(test_teencode)}")
    print(f"  Rò rỉ giữa các tập: {leakage}")
    print(f"  Câu cần duyệt ngôn ngữ: {len(review)}  (data/v2/review_language.csv)")
    print(f"  Train tăng cường: {len(train_aug)} câu {aug_stats}")
    for st in noise_stats:
        print(f"  Test nhiễu p={st['muc_nhieu_p']}: {st['ty_le_tu_bi_bien_doi_pct']}% từ bị biến đổi, "
              f"{st['ty_le_cau_co_bien_doi_pct']}% câu có biến đổi")
    print(f"  Module hiện tại khôi phục: {recovery}")
    print(f"Xong! Kết quả trong {OUT}/")


if __name__ == "__main__":
    main()
