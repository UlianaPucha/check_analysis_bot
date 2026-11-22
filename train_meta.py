import env_setup  # noqa: F401 (до transformers)
import argparse
import json
import os
import random
import re
import time

import pandas as pd
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText, AutoProcessor, get_cosine_schedule_with_warmup

import vlm_reader as V
from train_vlm import load_split

HERE = os.path.dirname(os.path.abspath(__file__))
META_PROMPT = ("Это фото кассового чека? Ответь JSON: {\"is_receipt\": true или false, \"shop\": название магазина "
               "или кафе, \"date\": дата покупки ДД.ММ.ГГГГ, \"time\": время ЧЧ:ММ}. Если это не чек или чего-то "
               "нет — null.")
# юрлицо крупной сети -> название, которое знает покупатель
BRANDS = [(r"АГРОТОРГ|ПЯТ[ЁЕ]РОЧКА", "Пятёрочка"), (r"ТАНДЕР|МАГНИТ", "Магнит"), (r"\bЛЕНТА\b", "Лента"),
          (r"ПЕРЕКР[ЁЕ]СТОК", "Перекрёсток"), (r"ЭЛЕМЕНТ-ТРЕЙД|МОНЕТКА", "Монетка"), (r"ДИКСИ", "Дикси"),
          (r"СИСТЕМА ПБО", "Вкусно и точка"), (r"ВКУСВИЛЛ", "ВкусВилл"), (r"СПОРТМАСТЕР", "Спортмастер"),
          (r"ПЛАНЕТА УВЛЕЧЕНИЙ", "Леонардо"), (r"КАМЕЛОТ", "Ярче"), (r"ПЕТЕРБУРГСКИЙ МЕТРОПОЛИТЕН", "Метрополитен СПб")]


def shop_name(seller) -> str | None:
    s = str(seller or "").strip()
    if not s or s.lower() in ("не распознано", "nan"):
        return None
    up = s.upper()
    for pattern, brand in BRANDS:
        if re.search(pattern, up):
            return brand
    s = re.sub(r"^(ОБЩЕСТВО С ОГРАНИЧЕННОЙ ОТВЕТСТВЕННОСТЬЮ|АКЦИОНЕРНОЕ ОБЩЕСТВО|ИНДИВИДУАЛЬНЫЙ ПРЕДПРИНИМАТЕЛЬ)",
               lambda m: {"О": "ООО", "А": "АО", "И": "ИП"}[m.group()[0]], s, flags=re.I)
    return " ".join(s.replace("«", '"').replace("»", '"').split())[:60]


def target(row, is_receipt: bool) -> str:
    if not is_receipt:
        return json.dumps({"is_receipt": False, "shop": None, "date": None, "time": None}, ensure_ascii=False)
    dt = V.parse_date(row.date_text, row.date_text, strict=False)  # для разметки старые даты не отбрасываем
    has_time = bool(re.search(r"\d{1,2}:\d{2}", str(row.date_text or "")))
    return json.dumps({"is_receipt": True, "shop": shop_name(row.seller_text),
                       "date": dt.strftime("%d.%m.%Y") if dt else None,
                       "time": dt.strftime("%H:%M") if dt and has_time else None}, ensure_ascii=False)


def make_example(processor, image_bytes: bytes, answer: str):
    image = V.to_pil(image_bytes)
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": META_PROMPT}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    enc = processor(text=[prompt + answer + "<|im_end|>\n"], images=[image], return_tensors="pt")
    n_prompt = processor(text=[prompt], images=[image], return_tensors="pt")["input_ids"].shape[1]
    return enc, n_prompt


def answer_loss(model, enc, n_prompt):
    ids = enc["input_ids"]
    logits = model(**enc, logits_to_keep=ids.shape[1] - n_prompt + 1).logits[:, :-1]
    return torch.nn.functional.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), ids[:, n_prompt:].reshape(-1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--out", default="meta_lora")
    args = ap.parse_args()
    random.seed(0), torch.manual_seed(0)
    torch.cuda.set_per_process_memory_fraction(0.9)

    pos = load_split("train")
    neg = pd.read_parquet(os.path.join(HERE, "data", "nonreceipts_train.parquet"))
    data = [(r.image["bytes"], target(r, True)) for r in pos.itertuples()] + \
           [(r.image["bytes"], target(None, False)) for r in neg.itertuples()]
    val_pos = load_split("val").head(30)
    val_neg = pd.read_parquet(os.path.join(HERE, "data", "nonreceipts_test.parquet")).head(20)
    val = [(r.image["bytes"], target(r, True)) for r in val_pos.itertuples()] + \
          [(r.image["bytes"], target(None, False)) for r in val_neg.itertuples()]
    print(f"train: чеков {len(pos)}, не чеков {len(neg)}; пример ответа: {data[0][1]}")

    processor = AutoProcessor.from_pretrained(V.MODEL_NAME)
    model = AutoModelForImageTextToText.from_pretrained(V.MODEL_NAME, dtype=torch.bfloat16, device_map="cuda")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    targets = [n for n, m in model.named_modules()
               if isinstance(m, torch.nn.Linear) and "language_model" in n and "lm_head" not in n]
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=targets,
                                             task_type="CAUSAL_LM"))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)
    steps = (len(data) * args.epochs + args.accum - 1) // args.accum
    sched = get_cosine_schedule_with_warmup(opt, int(0.05 * steps), steps)

    @torch.no_grad()
    def val_loss():
        model.eval()
        out = [answer_loss(model, *(lambda e: (e[0].to(model.device), e[1]))(make_example(processor, b, a))).item()
               for b, a in val]
        model.train()
        return sum(out) / len(out)

    print(f"val loss до обучения: {val_loss():.4f}", flush=True)
    best = float("inf")
    model.train()
    for epoch in range(1, args.epochs + 1):
        random.shuffle(data)
        t0, losses = time.time(), []
        for i, (b, a) in enumerate(data, 1):
            enc, n_prompt = make_example(processor, b, a)
            try:
                loss = answer_loss(model, enc.to(model.device), n_prompt) / args.accum
                loss.backward()
                losses.append(loss.item() * args.accum)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                print("  пропущен пример: не хватило видеопамяти", flush=True)
            if i % args.accum == 0 or i == len(data):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(), sched.step(), opt.zero_grad()
                torch.cuda.empty_cache()
            if i % 50 == 0:
                print(f"  эпоха {epoch}: {i}/{len(data)}  loss={sum(losses[-50:]) / len(losses[-50:]):.4f}  "
                      f"{(time.time() - t0) / i:.1f} с/пример  VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} ГБ",
                      flush=True)
        vl = val_loss()
        print(f"эпоха {epoch}: train loss {sum(losses) / len(losses):.4f}, val loss {vl:.4f}", flush=True)
        if vl < best:
            best = vl
            model.save_pretrained(os.path.join(HERE, args.out))
            print(f"  сохранено в {args.out}/", flush=True)
    print(f"лучший val loss {best:.4f}")


if __name__ == "__main__":
    main()
