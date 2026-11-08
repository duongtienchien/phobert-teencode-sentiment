"""
Bộ sinh teencode tổng hợp (Ní 1).

Biến câu tiếng Việt chuẩn thành câu "ngôn ngữ mạng" có kiểm soát, nhãn giữ nguyên:
    1. Từ điển ngược: các mục trong teencode_dict.json  (không -> ko / k / hok ...)
    2. Biến thể bổ sung chưa có trong từ điển            (không -> khum, biết -> bít ...)
    3. Từ lóng mượn tiếng Anh                            (khoe -> flex, buồn -> suy ...)
    4. Luật biến âm cho từ bất kỳ                         (qu -> w, v -> z, gi -> j, -ng -> -g ...)
    5. Kéo dài ký tự                                      (ngon -> ngonnnn)

CHỐNG "TỰ CHẤM ĐIỂM": khoảng 30% biến thể và 2 luật biến âm được đánh dấu held-out.
    - mode="train": chỉ dùng phần KHÔNG held-out (dữ liệu tăng cường cho train)
    - mode="test" : dùng TẤT CẢ (tập test có cả teencode mô hình/module chưa từng thấy)
Module chuẩn hóa v2 KHÔNG được thêm các biến thể held-out vào từ điển.

    from teencode_generator import TeencodeGenerator
    gen = TeencodeGenerator("teencode_dict.json")
    noisy, changes = gen.noisify("Không biết vậy có được không", p=0.5, mode="test", seed=1)
"""
import hashlib
import json
import random
import re
import unicodedata

from wordfreq import zipf_frequency

# ---------------------------------------------------------------------------
# 2 + 3. Biến thể bổ sung (chuẩn -> các cách viết mạng). Viết thường, NFC.
# ---------------------------------------------------------------------------
EXTRA_VARIANTS = {
    # teencode / viết tắt tiếng Việt
    "không": ["khum", "hông", "kg", "hem"],
    "được": ["đk", "dk"],
    "biết": ["bít", "bt"],
    "vậy": ["v", "dzậy", "zị"],
    "gì": ["gi", "jì"],
    "rồi": ["rùi", "ròi"],
    "thôi": ["thui"],
    "luôn": ["lun"],
    "nhiều": ["nhìu"],
    "muốn": ["mún"],
    "buồn": ["bùn"],
    "cười": ["cừi"],
    "về": ["zề"],
    "vui": ["dzui"],
    "tôi": ["t", "tui"],
    "tao": ["t"],
    "mày": ["m", "mài"],
    "em": ["e"],
    "anh": ["a"],
    "bạn": ["b", "bợn"],
    "cũng": ["cg"],
    "đang": ["đg", "dg"],
    "trời": ["tr", "trùi"],
    "chưa": ["chx"],
    "thật": ["thiệt"],
    "người": ["ngừi"],
    "thương": ["thưn"],
    "hôm nay": ["hum nay", "hn"],
    "bây giờ": ["bh", "bjo"],
    "bao giờ": ["bh"],
    "dễ thương": ["dth", "dthw", "dễ thw"],
    "bình thường": ["bthg"],
    "không có gì": ["kcj", "kcgi"],
    "cảm ơn": ["thanks", "cmơn", "camon"],
    "xin lỗi": ["sorry", "sỏi"],
    "nhắn tin": ["nt"],
    "như thế nào": ["ntnao"],
    "làm gì": ["lm j", "lm gì"],
    "với": ["vớj"],
    "nhưng": ["nhg"],
    "trước": ["trứk"],
    "quá": ["qa"],
    # từ lóng mượn tiếng Anh / từ lóng Gen Z
    "khoe": ["flex"],
    "khoe khoang": ["flex"],
    "người mình thích": ["crush"],
    "độc hại": ["toxic"],
    "buồn quá": ["suy quá"],
    "xấu hổ": ["quê", "cringe"],
    "ghen tị": ["gato"],
    "chia sẻ": ["share"],
    "bình luận": ["comment", "cmt"],
    "theo dõi": ["follow", "fl"],
    "hủy theo dõi": ["unfollow"],
    "đăng": ["post"],
    "trang điểm": ["makeup"],
    "tin giả": ["fake news"],
    "bạn thân": ["bff"],
    "hẹn hò": ["date"],
    "tạm biệt": ["bye", "bai"],
    "xin chào": ["hello", "hi"],
    "cuối tuần": ["weekend"],
    "dễ thương quá": ["cute quá"],
    "xu hướng": ["trend"],
    "người hâm mộ": ["fan"],
    "giảm giá": ["sale"],
    "tuyệt vời": ["đỉnh", "xịn sò", "perfect"],
    "rất hay": ["hay xỉu", "đỉnh"],
    "ngầu": ["cool", "chất"],
}

# Nguồn không nên đảo ngược (mục từ điển chỉ là chữ tắt của tên riêng / mơ hồ nghĩa)
SKIP_INVERSE_SOURCES = {"thằng", "lồn"}
# Biến thể mơ hồ, không sinh ra (dễ đổi nghĩa câu)
SKIP_VARIANTS = {"lol", "sub", "face", "kb", "vc", "z", "ce", "cu", "ku"}

# ---------------------------------------------------------------------------
# 4. Luật biến âm cho một từ (áp lên âm tiết tiếng Việt viết thường)
# ---------------------------------------------------------------------------
RULES = [
    ("qu→w", re.compile(r"^qu"), "w"),
    ("gi→j", re.compile(r"^gi"), "j"),
    ("v→z", re.compile(r"^v"), "z"),
    ("-ng→-g", re.compile(r"ng$"), "g"),
    ("-nh→-h", re.compile(r"(?<=[aăâeêioôơuưy])nh$"), "h"),
    ("ph→f", re.compile(r"^ph"), "f"),          # held-out
    ("c→k", re.compile(r"^c(?=[aăâoôơuư])"), "k"),  # held-out
]
HELDOUT_RULES = {"ph→f", "c→k"}
HELDOUT_RATIO = 0.3
VN_DIACRITIC = re.compile(r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]")
WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def nfc(s):
    return unicodedata.normalize("NFC", s)


def strip_accents(text):
    text = text.replace("đ", "d").replace("Đ", "D")
    return nfc(unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode())


def _is_heldout(source, variant):
    """Chia held-out cố định (không phụ thuộc thứ tự chạy) bằng hash."""
    h = int(hashlib.md5(f"{source}|{variant}".encode()).hexdigest(), 16)
    return (h % 1000) / 1000 < HELDOUT_RATIO


class TeencodeGenerator:
    def __init__(self, dict_path="teencode_dict.json"):
        with open(dict_path, encoding="utf-8") as f:
            tdict = {nfc(k): nfc(v) for k, v in json.load(f).items()}
        table = {}
        for variant, std in tdict.items():  # 1. từ điển ngược
            if std in SKIP_INVERSE_SOURCES or variant in SKIP_VARIANTS or variant == std:
                continue
            table.setdefault(std, []).append((variant, "từ điển"))
        for std, variants in EXTRA_VARIANTS.items():  # 2 + 3. bổ sung
            for v in variants:
                if v not in SKIP_VARIANTS:
                    table.setdefault(nfc(std), []).append((nfc(v), "bổ sung"))
        # mỗi biến thể: (variant, nguồn, held_out). Chỉ biến thể BỔ SUNG mới có thể held-out,
        # vì biến thể đã có trong từ điển v1 thì module vốn đã biết.
        self.table = {s: [(v, src, src == "bổ sung" and _is_heldout(s, v)) for v, src in vs]
                      for s, vs in table.items()}
        self.in_dict = set(tdict)
        # cụm nhiều từ trước, cụm dài trước
        sources = sorted(self.table, key=lambda s: (-len(s.split()), -len(s)))
        self.phrase_re = re.compile(
            r"(?<!\w)(" + "|".join(re.escape(s) for s in sources) + r")(?!\w)", re.UNICODE)

    @staticmethod
    def _is_vietnamese(word):
        """Chỉ biến âm âm tiết tiếng Việt: có dấu, hoặc không dấu nhưng không phải từ tiếng Anh phổ biến."""
        if re.search(r"[fjwz]", word):
            return False
        if VN_DIACRITIC.search(word):
            return True
        return zipf_frequency(word, "en") < 3.0

    # -- thống kê -----------------------------------------------------------
    def stats(self):
        pairs = [(s, v, src, h) for s, vs in self.table.items() for v, src, h in vs]
        return {
            "so_tu_chuan": len(self.table),
            "so_bien_the": len(pairs),
            "bien_the_held_out": sum(h for *_, h in pairs),
            "luat_bien_am": [r[0] for r in RULES],
            "luat_held_out": sorted(HELDOUT_RULES),
        }

    # -- sinh nhiễu ---------------------------------------------------------
    def noisify(self, text, p=0.3, mode="train", seed=None):
        """
        p: xác suất biến đổi mỗi vị trí đủ điều kiện (0.1 nhẹ, 0.3 vừa, 0.5 nặng).
        Trả về (câu đã nhiễu, danh sách thay đổi dạng dict).
        """
        rng = random.Random(seed)
        allow_heldout = mode == "test"
        text = nfc(str(text)).lower()
        changes = []

        # Bước A: từ / cụm có trong bảng biến thể
        def repl(m):
            src = m.group(0)
            options = [(v, s, h) for v, s, h in self.table[src] if allow_heldout or not h]
            if not options or rng.random() >= p:
                return "\x00" + src + "\x01"  # đánh dấu: không cho luật biến âm sửa tiếp
            v, s, h = rng.choice(options)
            changes.append({"goc": src, "moi": v, "loai": s, "held_out": h})
            return "\x00" + v + "\x01"

        text = self.phrase_re.sub(repl, text)

        # Bước B: luật biến âm + kéo dài ký tự cho các từ còn lại
        rules = [r for r in RULES if allow_heldout or r[0] not in HELDOUT_RULES]

        def word_repl(m):
            w = m.group(0)
            if not self._is_vietnamese(w):
                return w
            if rng.random() < p * 0.4:
                applicable = [(n, rx, to) for n, rx, to in rules if rx.search(w)]
                if applicable:
                    n, rx, to = rng.choice(applicable)
                    new = rx.sub(to, w, count=1)
                    changes.append({"goc": w, "moi": new, "loai": f"luật {n}", "held_out": n in HELDOUT_RULES})
                    return new
            if len(w) >= 2 and rng.random() < p * 0.05:
                new = w + w[-1] * rng.randint(2, 4)
                changes.append({"goc": w, "moi": new, "loai": "kéo dài ký tự", "held_out": False})
                return new
            return w

        parts = re.split(r"(\x00[^\x01]*\x01)", text)
        parts = [pt if pt.startswith("\x00") else WORD_RE.sub(word_repl, pt) for pt in parts]
        out = "".join(parts).replace("\x00", "").replace("\x01", "")
        return out, changes
