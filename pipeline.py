import os
import re
from dataclasses import dataclass, field
from datetime import datetime

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import vlm_reader
from preprocess import normalize  # та же предобработка, что и при обучении классификатора
from qr_reader import read_receipt_qr

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model_real")

PLACEHOLDER = re.compile(r"^(товар|позиция|наименование|item)\s*\d*\.?$", re.IGNORECASE)
SERVICE_LINE = re.compile(r"скидк|бонус|баллы|округлен|\bитог|\bндс\b|оплат|сдача|терминал|мерчант", re.IGNORECASE)
# порог уверенности: у 95% настоящих чеков на val она выше 0.88
LOW_CONFIDENCE = 0.85
# порог для 'это чек?' (meta_lora), подобран на val
NOT_RECEIPT_BELOW = 0.3

_tokenizer = _classifier = None


def load():
    """Загрузить все модели заранее (при старте бота), чтобы первый чек не ждал загрузки."""
    global _tokenizer, _classifier
    vlm_reader.load()
    if _classifier is None:
        if not os.path.isdir(MODEL_DIR):
            raise SystemExit("Нет папки model_real/ — сначала запусти train_real.py, чтобы обучить классификатор.")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        _classifier = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).eval()


def classify(names: list[str]) -> list[str]:
    load()
    enc = _tokenizer([normalize(n) for n in names], padding=True, truncation=True, max_length=64, return_tensors="pt")
    with torch.no_grad():
        ids = _classifier(**enc).logits.argmax(-1).tolist()
    return [_classifier.config.id2label[i] for i in ids]


def is_real_item(item: dict) -> bool:
    # не покупки: модификаторы в кафе с ценой 0 ('-с кетчупом', 'С собой'),
    # заглушки 'Товар', 'Товар 1' (так модель отвечает, когда чека на фото нет),
    # служебные строки (скидка, округление, итог)
    name = item["name"].strip()
    return item["price"] > 0 and not PLACEHOLDER.match(name) and not SERVICE_LINE.search(name)


@dataclass
class Result:
    status: str                       # ok / not_receipt / not_recognized
    items: list = field(default_factory=list)
    total: float | None = None        # из QR, иначе прочитанный моделью
    qr: dict | None = None
    shop: str | None = None
    date: datetime | None = None      # None - не нашли
    date_source: str = "sent"         # qr / receipt / sent
    confidence: float = 1.0

    @property
    def items_sum(self) -> float:
        return round(sum(it["price"] for it in self.items), 2)

    @property
    def mismatch(self) -> bool:
        return bool(self.total) and abs(self.total - self.items_sum) > 1

    @property
    def needs_review(self) -> bool:
        return self.mismatch or self.confidence < LOW_CONFIDENCE


def _read(image_bytes, **kw):
    items, total, _, conf = vlm_reader.extract_with_confidence(image_bytes, **kw)
    return [it for it in items if is_real_item(it)], total, conf


def fix_lost_thousands(image_bytes: bytes, qr: dict | None, items, total, conf):
    """Модель иногда теряет тысячи в записи «1 980,00» (читает 980). Признак: сумма позиций меньше итога
    ровно на 1000, 2000, … Тогда перечитываем чек адаптером, дообученным на чеках с дорогими позициями,
    и берём его ответ, только если у него сумма позиций сошлась с итогом."""
    ref = qr["total"] if qr else total
    if not ref or not items or not vlm_reader.has_thousands_adapter():
        return items, total, conf
    diff = ref - sum(it["price"] for it in items)
    if diff < 999 or abs(diff - round(diff / 1000) * 1000) > 1:
        return items, total, conf
    t_items, t_total, t_conf = _read(image_bytes, adapter="thousands")
    if t_items and abs((qr["total"] if qr else t_total or 0) - sum(it["price"] for it in t_items)) <= 1:
        print(f"  потерянные тысячи исправлены вторым адаптером: {sum(i['price'] for i in items):.2f} -> "
              f"{sum(i['price'] for i in t_items):.2f}")
        return t_items, t_total, t_conf
    return items, total, conf


def recognize(image_bytes: bytes, qr: dict | None):
    """Позиции чека. Если модель не уверена или сумма не сходится с QR - пробуем повернуть фото
    (чек мог быть снят боком или вверх ногами) и берём самый уверенный вариант."""
    items, total, conf = _read(image_bytes)
    items, total, conf = fix_lost_thousands(image_bytes, qr, items, total, conf)
    mismatch = qr and abs(qr["total"] - sum(it["price"] for it in items)) > 1
    if conf >= LOW_CONFIDENCE and not mismatch:
        return items, total, conf
    best = (items, total, conf)
    for angle in (90, 270, 180):
        r_items, r_total, r_conf = _read(image_bytes, rotate=angle)
        if r_items and r_conf > best[2] + 0.03:
            best = (r_items, r_total, r_conf)
        if qr and r_items and abs(qr["total"] - sum(it["price"] for it in r_items)) <= 1:
            return r_items, r_total, r_conf  # совпало с QR - ориентация верная
    return best


def process(image_bytes: bytes, progress=lambda text: None) -> Result:
    """progress(text) - сообщить пользователю, на каком шаге мы сейчас. ValueError - файл не картинка."""
    # TODO: если на фото два чека, читается только один
    progress("🔎 Ищу QR-код и проверяю, что это чек…")
    qr = read_receipt_qr(image_bytes)
    # шапка чека (meta_lora): это чек? дата
    meta = vlm_reader.read_header(image_bytes)
    # нет QR и модель считает, что это не чек - не разбираем,
    # иначе можно 'прочитать' покупки в меню или ценнике
    if not qr and meta["p_receipt"] < NOT_RECEIPT_BELOW:
        return Result("not_receipt")
    progress("🧾 Читаю позиции и цены…")
    items, total, conf = recognize(image_bytes, qr)
    if qr:
        total = qr["total"]
    if not items:
        return Result("not_recognized", qr=qr, total=total,
                      date=qr["datetime"] if qr else None, date_source="qr" if qr else "sent")
    progress("🏷 Определяю категории и магазин…")
    need = [it for it in items if "category" not in it]  # категорию обычно даёт сама модель распознавания
    for item, category in zip(need, classify([i["name"] for i in need]) if need else []):
        item["category"] = category
    if vlm_reader.has_meta_adapter():
        # магазин читает исходная модель: на чеках кафе она точнее дообученной
        # (та иногда пишет 'Добро пожаловать!' или имя официанта)
        meta["shop"] = vlm_reader.read_meta(image_bytes)["shop"]
    if qr and qr.get("datetime"):
        date, source = qr["datetime"], "qr"
    elif meta["date"]:
        date, source = meta["date"], "receipt"
    else:
        date, source = None, "sent"
    return Result("ok", items=items, total=total, qr=qr, shop=meta["shop"], date=date, date_source=source,
                  confidence=conf)
