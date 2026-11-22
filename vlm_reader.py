import json
from datetime import datetime, timedelta
import os
import re

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_NAME = os.getenv("VLM_MODEL", "Qwen/Qwen3.5-2B")
# наш LoRA-адаптер (train_vlm.py); VLM_ADAPTER=none - без адаптера
# по умолчанию адаптер с категориями (--with-category)
_here = os.path.dirname(os.path.abspath(__file__))
_default_adapter = next((os.path.join(_here, d) for d in ("vlm_lora_2b_cat",)
                         if os.path.isdir(os.path.join(_here, d))), "")
ADAPTER = os.getenv("VLM_ADAPTER") or (_default_adapter if os.path.isdir(_default_adapter) else None)
if ADAPTER and ADAPTER.lower() == "none":
    ADAPTER = None
MAX_SIDE = int(os.getenv("VLM_MAX_SIDE", "1600"))  # px: больше - лучше мелкий шрифт, но медленнее
# второй адаптер (+ синтетические чеки с ценами от 1000 ₽), включается только если
# сумма позиций меньше итога ровно на 1000*k (pipeline.fix_lost_thousands)
THOUSANDS_ADAPTER = os.getenv("VLM_THOUSANDS_ADAPTER") or os.path.join(_here, "vlm_lora_2b_cat_v2")
# адаптер шапки чека (train_meta.py); VLM_META_ADAPTER=none - исходная модель
META_ADAPTER = os.getenv("VLM_META_ADAPTER") or os.path.join(_here, "meta_lora")
if META_ADAPTER.lower() == "none":
    META_ADAPTER = None

PROMPT = (
    "Это фото кассового чека. Выпиши все товары (позиции) из чека в формате JSON:\n"
    '{"items": [{"name": "название как в чеке", "price": итоговая сумма по позиции числом}], "total": итог чека числом}\n'
    "Не включай скидки, НДС, бонусы, пакеты-подарки и служебные строки. Если позиция весовая или "
    "несколько штук — price это итоговая сумма по позиции. Ответь только JSON."
)
CATEGORIES = ["еда", "напитки", "бытовое", "здоровье", "одежда", "транспорт", "развлечения", "другое"]
# промпт с категориями
PROMPT_CAT = (
    "Это фото кассового чека. Выпиши все товары (позиции) из чека в формате JSON:\n"
    '{"items": [{"name": "название как в чеке", "price": итоговая сумма по позиции числом, '
    '"category": "категория"}], "total": итог чека числом}\n'
    "Категории: еда, напитки, бытовое, здоровье, одежда, транспорт, развлечения, другое.\n"
    "Не включай скидки, НДС, бонусы, пакеты-подарки и служебные строки. Если позиция весовая или "
    "несколько штук — price это итоговая сумма по позиции. Ответь только JSON."
)


def task_of(adapter: str | None) -> dict:
    path = os.path.join(adapter, "receipt_task.json") if adapter else None
    return json.load(open(path, encoding="utf-8")) if path and os.path.exists(path) else {"category": False}


WITH_CATEGORY = task_of(ADAPTER)["category"]


def prompt() -> str:
    return PROMPT_CAT if WITH_CATEGORY else PROMPT

_model = _processor = None


def load():
    global _model, _processor
    if _model is None:
        _processor = AutoProcessor.from_pretrained(MODEL_NAME)
        _model = AutoModelForImageTextToText.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, device_map="cuda")
        if ADAPTER:
            from peft import PeftModel
            _model = PeftModel.from_pretrained(_model, ADAPTER)
            if THOUSANDS_ADAPTER and os.path.isdir(THOUSANDS_ADAPTER):
                _model.load_adapter(THOUSANDS_ADAPTER, adapter_name="thousands")
            if META_ADAPTER and os.path.isdir(META_ADAPTER):
                _model.load_adapter(META_ADAPTER, adapter_name="meta")
            _model.set_adapter("default")
        _model.eval()
    return _model, _processor


ROTATIONS = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def to_pil(image_bytes: bytes, rotate: int = 0) -> Image.Image:
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("не картинка")
    if rotate:
        img = cv2.rotate(img, ROTATIONS[rotate])
    scale = MAX_SIDE / max(img.shape[:2])
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def build_inputs(processor, image: Image.Image, answer: str | None = None):
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt()}]}]
    if answer is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    text = processor.apply_chat_template(messages, add_generation_prompt=answer is None, tokenize=False,
                                         enable_thinking=False)
    return processor(text=[text], images=[image], return_tensors="pt")


def to_number(x) -> float | None:
    if isinstance(x, (int, float)):
        return float(x)
    m = re.search(r"\d+(?:[.,]\d+)?", str(x or "").replace(" ", ""))
    return float(m.group().replace(",", ".")) if m else None


ITEM_RE = re.compile(r'\{\s*"name"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,\s*"price"\s*:\s*([-\d.,]+)'
                     r'(?:\s*,\s*"category"\s*:\s*"([^"]*)")?\s*\}')


def _item(name, price, category=None) -> dict | None:
    price = to_number(price)
    if not name or price is None:
        return None
    item = {"name": str(name).strip(), "price": price}
    if category in CATEGORIES:
        item["category"] = category
    return item


def parse_answer(text: str) -> tuple[list[dict], float | None]:
    """Достаём JSON из ответа модели (иногда она оборачивает его в ```json ... ```).
    Если ответ оборвался (очень длинный чек), собираем все целиком записанные позиции."""
    m = re.search(r"\{.*\}", text, re.S)
    try:
        data = json.loads(m.group()) if m else None
    except json.JSONDecodeError:
        data = None
    if data is None:
        items = [it for it in (_item(json.loads(f'"{n}"'), p, c) for n, p, c in ITEM_RE.findall(text)) if it]
        return items, None
    items = []
    for it in data.get("items") or []:
        if isinstance(it, dict):
            item = _item(it.get("name"), it.get("price"), it.get("category"))
            if item:
                items.append(item)
    return items, to_number(data.get("total"))


@torch.no_grad()
def extract(image_bytes: bytes, max_new_tokens: int = 2048,
            use_adapter: bool = True) -> tuple[list[dict], float | None, str]:
    items, total, text, _ = extract_with_confidence(image_bytes, max_new_tokens, use_adapter)
    return items, total, text


@torch.no_grad()
def extract_with_confidence(image_bytes: bytes, max_new_tokens: int = 2048, use_adapter: bool = True,
                            rotate: int = 0, adapter: str = "default"):
    """То же, что extract, плюс уверенность модели в ответе: средняя вероятность выбранных токенов
    в названиях и ценах (0..1). На выдуманных ответах (пустой лист, не чек) она заметно ниже."""
    model, processor = load()
    inputs = build_inputs(processor, to_pil(image_bytes, rotate)).to(model.device)
    kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False, return_dict_in_generate=True, output_scores=True)
    if ADAPTER and not use_adapter:
        with model.disable_adapter():
            out = model.generate(**inputs, **kwargs)
    elif adapter != "default":
        model.set_adapter(adapter)
        try:
            out = model.generate(**inputs, **kwargs)
        finally:
            model.set_adapter("default")
    else:
        out = model.generate(**inputs, **kwargs)
    new_tokens = out.sequences[:, inputs["input_ids"].shape[1]:]
    text = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
    items, total = parse_answer(text)
    # вероятность каждого сгенерированного токена
    logprobs = model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)[0]
    probs = logprobs.float().exp().cpu().tolist()
    pieces = [processor.decode([t]) for t in new_tokens[0].tolist()]
    # только буквы и цифры, без скобок и кавычек JSON
    content = [p for p, piece in zip(probs, pieces) if any(ch.isalnum() for ch in piece)
               and piece.strip() not in ("items", "name", "price", "total", "category")]
    confidence = sum(content) / len(content) if content else 0.0
    return items, total, text, confidence


IS_RECEIPT_QUESTION = (
    "Это фотография или скриншот кассового чека из магазина, кафе, аптеки или другой покупки — "
    "с перечнем купленных товаров или услуг и их стоимостью? Меню, ценник, страница текста, "
    "банковский перевод, слип оплаты картой без списка товаров — это НЕ кассовый чек. Ответь одним словом: да или нет.")


@torch.no_grad()
def receipt_probability(image_bytes: bytes) -> float:
    """Вероятность (0..1), что на фото кассовый чек с покупками. Спрашиваем исходную модель (без адаптера):
    наш адаптер обучен только на чеках и на этот вопрос не учился. На замере: у не-чеков (текст, меню, ценник,
    перевод, слип терминала, пустой лист, шум) - не выше 0.32, у настоящих чеков - не ниже 0.55."""
    model, processor = load()
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": IS_RECEIPT_QUESTION}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    enc = processor(text=[text], images=[to_pil(image_bytes)], return_tensors="pt").to(model.device)
    ids = lambda words: [processor.tokenizer.encode(w, add_special_tokens=False)[0] for w in words]
    if ADAPTER:
        with model.disable_adapter():
            logits = model(**enc).logits[0, -1].float()
    else:
        logits = model(**enc).logits[0, -1].float()
    probs = torch.softmax(logits, -1)
    yes, no = probs[ids(["да", "Да"])].sum().item(), probs[ids(["нет", "Нет"])].sum().item()
    return yes / (yes + no + 1e-9)


META_QUESTION = (
    "Посмотри на кассовый чек и ответь JSON: {\"shop\": \"название магазина или кафе\", "
    "\"date\": \"дата покупки ДД.ММ.ГГГГ\", \"time\": \"время ЧЧ:ММ\"}. "
    "Название обычно напечатано вверху чека или рядом с ИНН; не путай его с рекламой (например, «СберЧаевые»), "
    "кассовой программой или банком. Дата покупки — рядом со временем, обычно внизу чека или в шапке "
    "(«Открыт», «Дата»). Если чего-то нет на чеке — null.")


def parse_date(date_text, time_text=None, strict: bool = True) -> datetime | None:
    """'19.06.2026', '19.06.26', '19/06/26', '2026-06-19' (+ время '22:54') -> datetime; None, если не похоже на дату."""
    if not date_text:
        return None
    m = re.search(r"(\d{1,4})[./-](\d{1,2})[./-](\d{2,4})", str(date_text))
    if not m:
        return None
    a, b, c = m.groups()
    if len(a) == 4:
        y, mo, d = int(a), int(b), int(c)
    else:
        d, mo, y = int(a), int(b), int(c)
        y += 2000 if y < 100 else 0
    hh = mm = 0
    t = re.search(r"(\d{1,2}):(\d{2})", str(time_text or ""))  # только с двоеточием: «20.10» - это дата
    if t:
        hh, mm = int(t.group(1)), int(t.group(2))
    try:
        dt = datetime(y, mo, d, min(hh, 23), min(mm, 59))
    except ValueError:
        return None
    # чеки из будущего или слишком старые - ошибка чтения
    if strict and (dt > datetime.now() + timedelta(days=1) or dt.year < 2015):
        return None
    return dt


@torch.no_grad()
def read_meta(image_bytes: bytes) -> dict:
    """Название магазина и дата покупки с чека (исходная модель без адаптера): {"shop": str|None, "date": datetime|None}."""
    model, processor = load()
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": META_QUESTION}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    enc = processor(text=[text], images=[to_pil(image_bytes)], return_tensors="pt").to(model.device)
    if ADAPTER:
        with model.disable_adapter():
            out = model.generate(**enc, max_new_tokens=60, do_sample=False)
    else:
        out = model.generate(**enc, max_new_tokens=60, do_sample=False)
    answer = processor.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    m = re.search(r"\{.*\}", answer, re.S)
    try:
        data = json.loads(m.group()) if m else {}
    except json.JSONDecodeError:
        data = {}
    shop = str(data.get("shop") or "").strip().strip('."«»')
    if not shop or shop.lower() in ("null", "нет", "неизвестно") or len(shop) > 60:
        shop = None
    return {"shop": shop, "date": parse_date(data.get("date"), data.get("time"))}


def read_shop(image_bytes: bytes) -> str | None:
    return read_meta(image_bytes)["shop"]


def has_thousands_adapter() -> bool:
    model, _ = load()
    return "thousands" in getattr(model, "peft_config", {})


META_PROMPT = ("Это фото кассового чека? Ответь JSON: {\"is_receipt\": true или false, \"shop\": название магазина "
               "или кафе, \"date\": дата покупки ДД.ММ.ГГГГ, \"time\": время ЧЧ:ММ}. Если это не чек или чего-то "
               "нет — null.")  # тот же промпт, что при обучении (train_meta.py)


def has_meta_adapter() -> bool:
    model, _ = load()
    return "meta" in getattr(model, "peft_config", {})


@torch.no_grad()
def read_header(image_bytes: bytes) -> dict:
    """«Шапка» чека одним проходом дообученной модели: {"p_receipt": 0..1, "shop": str|None, "date": datetime|None}.
    p_receipt - вероятность ответа is_receipt=true. Без адаптера meta - два вопроса к исходной модели (как раньше)."""
    # TODO: время покупки meta_lora почти не выдаёт (в разметке датасета его мало), остаётся 00:00
    if not has_meta_adapter():
        meta = read_meta(image_bytes)
        return {"p_receipt": receipt_probability(image_bytes), **meta}
    model, processor = load()
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": META_PROMPT}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    enc = processor(text=[text], images=[to_pil(image_bytes)], return_tensors="pt").to(model.device)
    model.set_adapter("meta")
    try:
        out = model.generate(**enc, max_new_tokens=70, do_sample=False, return_dict_in_generate=True, output_scores=True)
    finally:
        model.set_adapter("default")
    new = out.sequences[0, enc["input_ids"].shape[1]:].tolist()
    answer = processor.decode(new, skip_special_tokens=True)
    # вероятность true/false в первой позиции, где модель между ними выбирает
    tok = processor.tokenizer
    true_ids = {t[0] for t in (tok.encode(w, add_special_tokens=False) for w in ("true", " true")) if t}
    false_ids = {t[0] for t in (tok.encode(w, add_special_tokens=False) for w in ("false", " false")) if t}
    p_receipt = None
    for step, token in enumerate(new):
        if token in true_ids or token in false_ids:
            probs = torch.softmax(out.scores[step][0].float(), -1)
            pt, pf = probs[list(true_ids)].sum().item(), probs[list(false_ids)].sum().item()
            p_receipt = pt / (pt + pf + 1e-9)
            break
    m = re.search(r"\{.*\}", answer, re.S)
    try:
        data = json.loads(m.group()) if m else {}
    except json.JSONDecodeError:
        data = {}
    if p_receipt is None:
        p_receipt = 1.0 if data.get("is_receipt") else 0.0
    shop = str(data.get("shop") or "").strip().strip('."«»')
    if not shop or shop.lower() in ("null", "нет") or len(shop) > 60:
        shop = None
    return {"p_receipt": p_receipt, "shop": shop, "date": parse_date(data.get("date"), data.get("time"))}
