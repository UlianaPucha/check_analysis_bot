from datetime import datetime
from urllib.parse import parse_qs

import cv2
import numpy as np

try:
    import zxingcpp  # лучше штатного детектора OpenCV на мятых и наклонённых чеках
except ImportError:
    zxingcpp = None

OPERATION = {"1": "приход", "2": "возврат прихода", "3": "расход", "4": "возврат расхода"}


def _decode_texts(gray: np.ndarray):
    """Пробуем несколько вариантов картинки: часто QR читается только после размытия или уменьшения."""
    variants = [gray, cv2.GaussianBlur(gray, (3, 3), 0)]
    if max(gray.shape) > 1600:
        variants.append(cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
    for img in variants:
        if zxingcpp is not None:
            for r in zxingcpp.read_barcodes(img, formats=zxingcpp.BarcodeFormat.QRCode, try_rotate=True,
                                            try_downscale=True):
                yield r.text
        text, _, _ = cv2.QRCodeDetector().detectAndDecode(img)
        if text:
            yield text


def parse_qr(text: str) -> dict | None:
    """'t=20261106T2305&s=116.00&fn=...&i=...&fp=...&n=1' -> словарь; None, если это не QR чека."""
    q = {k: v[0] for k, v in parse_qs(text.strip()).items()}
    if not {"t", "s", "fn"} <= q.keys():
        return None
    try:
        total = float(q["s"].replace(",", "."))
    except ValueError:
        return None
    dt = None
    for fmt in ("%Y%m%dT%H%M%S", "%Y%m%dT%H%M"):
        try:
            dt = datetime.strptime(q["t"], fmt)
            break
        except ValueError:
            pass
    return {"total": total, "datetime": dt, "fn": q.get("fn"), "fd": q.get("i"), "fp": q.get("fp"),
            "operation": OPERATION.get(q.get("n", ""), None), "raw": text}


def read_receipt_qr(image_bytes: bytes) -> dict | None:
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    for text in _decode_texts(img):
        parsed = parse_qr(text)
        if parsed:
            return parsed
    return None
