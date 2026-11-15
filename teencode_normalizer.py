"""
Module chuẩn hóa ngôn ngữ mạng v2 (Ní 1).

    from teencode_normalizer import TeencodeNormalizer
    norm = TeencodeNormalizer("teencode_dict.json", lexicon_texts=list_cau_co_dau)
    text, changes = norm.normalize("ko bít j lun á, wá đỉnh :))")
    # -> "không biết gì luôn á , quá đỉnh cười", [("ko","không"), ("bít","biết"), ...]

Các bước (theo thứ tự):
    1. Unicode NFC, chữ thường
    2. Emoji / emoticon / tiếng cười -> từ        (😂 :)) kkk haha lol -> "cười")
    3. Số + k / tr                                 (50k -> "50 nghìn", trừ "24k vàng")
    4. Tách token: tách dấu câu khỏi từ, GIỮ NGUYÊN từ có gạch nối (k-pop) và URL
    5. Rút gọn ký tự kéo dài                        (ngonnnn -> ngon)
    6. Tra cụm nhiều từ, rồi tra từ đơn trong từ điển v2
    7. Luật biến âm ngược, chỉ nhận kết quả là từ tiếng Việt có thật
       (wá -> quá, zui -> vui, khôg -> không ...)

Từ điển v2 = từ điển v1 (đã sửa lỗi) + biến thể bổ sung KHÔNG held-out của teencode_generator.
Biến thể và luật held-out cố ý KHÔNG đưa vào để tập test đo được khả năng tổng quát hóa.
"""
import json
import re
import unicodedata

# ---------------------------------------------------------------------------
# Sửa lỗi từ điển v1
# ---------------------------------------------------------------------------
DICT_FIXES = {
    "lol": "cười",      # v1: "lồn" -> sai, trong dữ liệu "lol" luôn mang nghĩa cười
    "vđ": "vấn đề",     # v1: "vãi đái"
    "vd": "video",      # v1: "ví dụ"; trong bình luận "vd" là video ("xem vd đọc cap")
    "cug": "cũng",
}
DICT_REMOVE = {"thằng", "aye", "face"}  # mục ánh xạ về chính nó / tiếng Anh ("aye aye captain")
# "DM" viết hoa = nhắn tin riêng ("DM tui đi"); "dm"/"Dm" = chửi thề -> xử lý TRƯỚC khi hạ chữ thường
DM_MESSAGE_RE = re.compile(r"(?<!\w)DM(?!\w)")
# Từ VIẾT HOA TOÀN BỘ thường là tên riêng / ký hiệu (IU, XL) -> giữ nguyên, trừ các viết tắt mạng phổ biến
CAPS_RE = re.compile(r"(?<!\w)[A-ZĐ]{2,}(?!\w)")
CAPS_STILL_NORMALIZE = {"qc", "vn", "ok", "lol", "lmao", "wtf", "vl", "vcl", "vkl", "cmm", "clmm", "dcm", "đm", "fb",
                        "ko", "k", "dc", "đc", "ny", "mn", "ae", "ace", "cmt", "stk", "tks", "ib"}
# Từ tiếng Việt bình thường: bộ sinh dùng làm "tiếng lóng", nhưng module KHÔNG được đổi ngược
# (vd "cơm quê", "chất lượng", "thiệt hại", "đỉnh núi")
NEVER_NORMALIZE = {"quê", "chất", "đỉnh", "thiệt", "hay xỉu"}

# ---------------------------------------------------------------------------
# Emoji, emoticon, tiếng cười
# ---------------------------------------------------------------------------
EMOJI_WORDS = {
    "cười": "😂🤣😆😹😁😄😃😀😅😝😜😛",
    "khóc": "😭😢😿🥲",
    "buồn": "😞😔😟🙁☹😥😓😩😫💔",
    "tức giận": "😡😠🤬👿💢",
    "yêu": "❤😍🥰😘💕💖💗💓💞♥😻",
    "vui": "😊☺😌🤗🥳🎉",   # 🙂 bỏ ra: tiếng Việt mạng hay dùng để mỉa mai
    "ngầu": "😎",
    "tốt": "👍👌💯🔥👏",
    "tệ": "👎💩🤮🤢",
    "chán": "🙄😒😑😐",
    "ngạc nhiên": "😮😯😲😱🤯😳",
    "suy nghĩ": "🤔",
}
EMOJI_MAP = {ch: w for w, chars in EMOJI_WORDS.items() for ch in chars}
EMOJI_RE = re.compile("|".join(re.escape(c) for c in sorted(EMOJI_MAP, key=len, reverse=True)))
OTHER_EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF☀-➿️‍]")
EMOTICONS = [
    (re.compile(r"(?<!\w)(?:[:=;]+[)\]]+|:v+|=\]+|:d+|xd+|\^\^+|\^_\^)(?!\w)", re.I), " cười "),
    (re.compile(r"(?<!\w)(?:[:=;]+[(\[]+|:'\(+|t_t|-_-)(?!\w)", re.I), " buồn "),
    (re.compile(r"(?<!\w)<3+(?!\w)"), " yêu "),
]
LAUGH_RE = re.compile(r"(?<!\w)(?:k{3,}|(?:ha){2,}h?|(?:he){2,}|(?:hi){2,}|(?:hj){2,}|(?:kk)+k*|lol|lmao|haizz+)(?!\w)",
                      re.I)

# ---------------------------------------------------------------------------
# Số, token
# ---------------------------------------------------------------------------
MONEY_RE = re.compile(r"(?<![\w.])(\d+(?:[.,]\d+)?)\s?(k|tr|củ|lít)(?!\w)", re.I)
MONEY_WORD = {"k": "nghìn", "tr": "triệu", "củ": "triệu", "lít": "trăm nghìn"}
TOKEN_RE = re.compile(r"https?://\S+|\w+(?:-\w+)+|\w+|[^\w\s]", re.UNICODE)
REPEAT_RE = re.compile(r"([^\W\d_])\1{2,}")  # chỉ chữ cái, không đụng tới số


def _base(ch):
    return unicodedata.normalize("NFD", ch)[0]


def collapse_repeats(word):
    """Rút chuỗi >= 3 ký tự cùng gốc (bỏ qua dấu) về 1 ký tự: "quáaaa" -> "quá", "ngonnn" -> "ngon"."""
    out, i = [], 0
    while i < len(word):
        j = i
        while j + 1 < len(word) and _base(word[j + 1]) == _base(word[i]) and word[i].isalpha():
            j += 1
        out.append(word[i] if j - i + 1 >= 3 else word[i:j + 1])
        i = j + 1
    return "".join(out)

VN_DIACRITIC = re.compile(r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]")

# Luật biến âm NGƯỢC (chỉ các luật không held-out). Ứng viên chỉ được nhận khi là từ có thật.
REVERSE_RULES = [
    (re.compile(r"^w"), ["qu"]),
    (re.compile(r"^j"), ["gi"]),
    (re.compile(r"^z"), ["v", "d", "gi"]),
    (re.compile(r"^dz"), ["v", "d"]),
    (re.compile(r"g$"), ["ng"]),   # chỉ áp cho từ CÓ DẤU (khôg, đúg) - tránh "bug" -> "bung"
    (re.compile(r"h$"), ["nh"]),   # chỉ áp cho từ CÓ DẤU (hìh)
]
NEEDS_ACCENT = {"g$", "h$"}


MIN_FREQ = 3  # từ kết quả phải xuất hiện ít nhất 3 lần trong dữ liệu


def nfc(s):
    return unicodedata.normalize("NFC", str(s))


class TeencodeNormalizer:
    def __init__(self, dict_path="teencode_dict.json", lexicon_texts=(), use_generator_extras=True):
        with open(dict_path, encoding="utf-8") as f:
            d = {nfc(k).lower(): nfc(v) for k, v in json.load(f).items()}
        for k in DICT_REMOVE:
            d.pop(k, None)
        d.update(DICT_FIXES)
        d = {k: v for k, v in d.items() if k != v}

        if use_generator_extras:  # biến thể bổ sung KHÔNG held-out, bỏ biến thể mơ hồ nhiều nghĩa
            from teencode_generator import TeencodeGenerator
            gen = TeencodeGenerator(dict_path)
            extra = {}
            for std, variants in gen.table.items():
                for v, src, held_out in variants:
                    if src == "bổ sung" and not held_out and len(v) > 1 and v not in NEVER_NORMALIZE:
                        extra.setdefault(v, set()).add(std)
            for v, stds in extra.items():
                if len(stds) == 1 and v not in d:
                    d[v] = stds.pop()

        self.single = {k: v for k, v in d.items() if " " not in k}
        multi = {k: v for k, v in d.items() if " " in k}
        self.multi = multi
        self.multi_re = (re.compile(r"(?<!\w)(" + "|".join(re.escape(k) for k in sorted(multi, key=len, reverse=True))
                                    + r")(?!\w)") if multi else None)

        # Từ vựng tiếng Việt có thật, để kiểm tra kết quả của luật biến âm ngược
        lex = {}
        for t in list(lexicon_texts) + list(d.values()):
            for w in re.findall(r"\w+", nfc(t).lower()):
                if VN_DIACRITIC.search(w) or len(w) > 1:
                    lex[w] = lex.get(w, 0) + 1
        self.lexicon = lex

    # -----------------------------------------------------------------------
    def _reverse_rule(self, w):
        """Nhận ứng viên nếu là từ có thật VÀ phổ biến hơn hẳn chính từ gốc trong dữ liệu."""
        if not re.fullmatch(r"[^\W\d_]+", w):
            return None
        own = self.lexicon.get(w, 0)
        best = None
        for rx, targets in REVERSE_RULES:
            if not rx.search(w) or (rx.pattern in NEEDS_ACCENT and not VN_DIACRITIC.search(w)):
                continue
            for t in targets:
                cand = rx.sub(t, w, count=1)
                freq = self.lexicon.get(cand, 0)
                if freq >= MIN_FREQ and freq > 3 * own and (best is None or freq > self.lexicon[best]):
                    best = cand
        return best

    def normalize(self, text):
        """Trả về (câu đã chuẩn hóa, danh sách thay đổi [(cũ, mới), ...])."""
        changes = []
        text = nfc(text).replace("\\n", " ").replace("\n", " ")
        text = DM_MESSAGE_RE.sub(lambda m: changes.append(("DM", "nhắn tin")) or " nhắn tin ", text)
        keep_caps = {w.lower() for w in CAPS_RE.findall(text)} - CAPS_STILL_NORMALIZE
        text = text.lower()

        def sub(rx, repl, s, label=None):
            def f(m):
                new = repl(m) if callable(repl) else repl
                changes.append((m.group(0), (label or new).strip()))
                return f" {new} "
            return rx.sub(f, s)

        # 2. emoji / emoticon / tiếng cười
        text = sub(EMOJI_RE, lambda m: EMOJI_MAP[m.group(0)], text)
        text = OTHER_EMOJI_RE.sub(" ", text)
        for rx, word in EMOTICONS:
            text = sub(rx, word, text)
        text = sub(LAUGH_RE, "cười", text)

        # 3. số + k / tr  (bỏ qua "24k vàng")
        def money(m):
            around = text[max(0, m.start() - 8):m.end() + 8]
            if m.group(2).lower() == "k" and ("vàng" in around or m.group(1) in {"2", "4", "8"}):  # vàng 24k, video 4k
                return m.group(0)
            return f"{m.group(1)} {MONEY_WORD[m.group(2).lower()]}"
        text = sub(MONEY_RE, money, text)

        # 6a. cụm nhiều từ
        if self.multi_re:
            text = sub(self.multi_re, lambda m: self.multi[m.group(0)], text)

        # 4-7. từng token
        out = []
        for tok in TOKEN_RE.findall(text):
            if not re.match(r"\w", tok) or "-" in tok or tok.startswith("http") or tok in keep_caps:
                out.append(tok)
                continue
            new = tok
            short = collapse_repeats(tok)
            if short != tok:
                new = short
            if new in self.single:
                new = self.single[new]
            else:
                r = self._reverse_rule(new)
                if r:
                    new = r
            if new != tok:
                changes.append((tok, new))
            out.append(new)
        return re.sub(r"\s+", " ", " ".join(out)).strip(), changes
