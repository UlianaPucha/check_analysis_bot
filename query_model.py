import json
import os
import re

import torch

import vlm_reader as V

HERE = os.path.dirname(os.path.abspath(__file__))
QUERY_ADAPTER = os.getenv("QUERY_ADAPTER") or os.path.join(HERE, "query_lora")
PROMPT = ("Преобразуй вопрос пользователя о его тратах в JSON-запрос с полями metric, period, category, text.\n"
          "metric: sum, count, avg_receipt, top_items, top_categories, top_shops, max_item или none (если это не вопрос о тратах).\n"
          "period: today, yesterday, this_week, last_week, last_7_days, this_month, last_month, last_30_days, "
          "this_year, all, month:MM или month:MM.YYYY.\n"
          "category: еда, напитки, бытовое, здоровье, одежда, транспорт, развлечения, другое или null.\n"
          "text: слово для поиска в названиях товаров и магазинов или null.\nВопрос: ")


def chat_prompt(processor, question: str) -> str:
    messages = [{"role": "user", "content": [{"type": "text", "text": PROMPT + question}]}]
    return processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)


def available() -> bool:
    return os.path.isdir(QUERY_ADAPTER)


_loaded = False


def load():
    """Подключаем адаптер вопросов к уже загруженной модели распознавания (веса общие)."""
    global _loaded
    model, processor = V.load()
    if not _loaded:
        if not hasattr(model, "load_adapter"):
            raise RuntimeError("модель распознавания загружена без адаптера — некуда подключить адаптер вопросов")
        model.load_adapter(QUERY_ADAPTER, adapter_name="query")
        model.set_adapter("default")
        _loaded = True
    return model, processor


def parse(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    try:
        q = json.loads(m.group()) if m else None
    except json.JSONDecodeError:
        return None
    if not isinstance(q, dict) or q.get("metric") is None:
        return None
    return {k: q.get(k) for k in ("metric", "period", "category", "text")}


@torch.no_grad()
def question_to_query(question: str) -> dict | None:
    model, processor = load()
    enc = processor(text=[chat_prompt(processor, question)], return_tensors="pt").to(model.device)
    model.set_adapter("query")
    try:
        out = model.generate(**enc, max_new_tokens=80, do_sample=False)
    finally:
        model.set_adapter("default")  # вернуть адаптер распознавания чеков
    return parse(processor.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)[0])
