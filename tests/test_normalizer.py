"""
Unit test cho module chuẩn hóa v2.

    python tests/test_normalizer.py      (hoặc: pytest tests/)
"""
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from teencode_normalizer import TeencodeNormalizer  # noqa: E402

LEXICON = pd.read_csv("vne_dataset.csv")["comment"].astype(str).tolist()
NORM = TeencodeNormalizer("teencode_dict.json", lexicon_texts=LEXICON)

# (đầu vào, đầu ra mong đợi, nhóm tính năng)
CASES = [
    # Teencode trong từ điển
    ("ko biết", "không biết", "từ điển"),
    ("dc ko", "được không", "từ điển"),
    ("fb", "facebook", "từ điển"),
    ("ng yêu", "người yêu", "từ điển"),
    ("my fen ơi", "bạn tôi ơi", "cụm nhiều từ"),
    # Dính dấu câu (lỗi của v1)
    ("ko, ko biết", "không , không biết", "dấu câu"),
    ("vl!", "vãi lồn !", "dấu câu"),
    ("fb?", "facebook ?", "dấu câu"),
    ("(ko)", "( không )", "dấu câu"),
    # Ký tự kéo dài
    ("ngonnnn quáaaa", "ngon quá", "kéo dài"),
    ("1000 người", "1000 người", "kéo dài - không đụng số"),
    # Tiếng cười, emoji, emoticon
    ("kkk hay", "cười hay", "tiếng cười"),
    ("hahaha", "cười", "tiếng cười"),
    ("LOL, buồn cười", "cười , buồn cười", "lol = cười (sửa lỗi v1)"),
    ("đẹp quá 😍", "đẹp quá yêu", "emoji"),
    ("giá trên mây 😭", "giá trên mây khóc", "emoji"),
    ("hài ghê :))", "hài ghê cười", "emoticon"),
    ("buồn :((", "buồn buồn", "emoticon"),
    # Số tiền
    ("110k mua dĩa trứng", "110 nghìn mua dĩa trứng", "tiền"),
    ("lương 15tr", "lương 15 triệu", "tiền"),
    ("vàng 24k hả", "vàng 24k hả", "tiền - bỏ qua vàng"),
    ("24k vàng", "24k vàng", "tiền - bỏ qua vàng"),
    # Bẫy: không được đổi
    ("lớp học k-pop", "lớp học k-pop", "gạch nối"),
    ("y2k aesthetic", "y2k aesthetic", "chữ + số"),
    ("thằng này", "thằng này", "mục thừa đã bỏ"),
    # Biến thể bổ sung (không held-out) của bộ sinh
    ("ròi", "rồi", "bổ sung"),
    ("khum biết", "không biết", "bổ sung"),
    ("toxic quá", "độc hại quá", "từ lóng tiếng Anh"),
    # Held-out: module CỐ Ý không khôi phục (để đo khả năng tổng quát hóa)
    ("rùi", "rùi", "held-out"),
    # Từ tiếng Việt bình thường: không được đổi
    ("cơm quê ngon", "cơm quê ngon", "từ thường"),
    ("chất lượng tốt", "chất lượng tốt", "từ thường"),
    ("thiệt hại nặng", "thiệt hại nặng", "từ thường"),
    # Luật biến âm ngược
    ("wá", "quá", "biến âm w→qu"),
    ("zui wá", "vui quá", "biến âm z→v"),
    ("khôg", "không", "biến âm -g→-ng"),
    ("jờ", "giờ", "biến âm j→gi"),
    # Không đụng tới tiếng Anh
    ("wow what", "wow what", "tiếng Anh"),
    ("game nhiều bug", "game nhiều bug", "tiếng Anh - bug"),
    ("aye aye captain", "aye aye captain", "tiếng Anh - aye"),
    # Lỗi phát hiện khi chạy trên dữ liệu thật
    ("DM tui đi", "nhắn tin tui đi", "DM viết hoa = nhắn tin"),
    ("Dm, lừa đảo hả", "địt mẹ , lừa đảo hả", "dm = chửi thề"),
    ("t cug yk", "t cũng yk", "cug = cũng; t mơ hồ (tao/tôi) nên giữ nguyên"),
    ("xem vd", "xem video", "vd = video"),
    ("đúg r", "đúng rồi", "biến âm -g, từ có dấu"),
    ("video có 4k không", "video có 4k không", "4k = độ phân giải"),
    ("tưởng IU chứ", "tưởng iu chứ", "viết hoa = tên riêng"),
    ("giao size XL", "giao size xl", "viết hoa = ký hiệu"),
    ("QC nhiều quá", "quảng cáo nhiều quá", "viết hoa nhưng là viết tắt mạng"),
    ("full face", "full face", "face không phải facebook"),
]


def run():
    failed = []
    for inp, expected, group in CASES:
        got, _ = NORM.normalize(inp)
        if got != expected:
            failed.append((group, inp, expected, got))
    for g, i, e, got in failed:
        print(f"SAI  [{g}] {i!r}\n     mong đợi: {e!r}\n     nhận được: {got!r}")
    print(f"\n{len(CASES) - len(failed)}/{len(CASES)} test đạt")
    return failed


def test_all():  # cho pytest
    assert not run()


if __name__ == "__main__":
    sys.exit(1 if run() else 0)
