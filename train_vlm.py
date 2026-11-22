import env_setup  # noqa: F401 (до transformers)
import argparse
import glob
import json
import math
import os
import random
import time

import pandas as pd
import psutil
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText, AutoProcessor, get_cosine_schedule_with_warmup

import vlm_reader as V

HERE = os.path.dirname(os.path.abspath(__file__))


def load_split(split: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(HERE, "data", "receipt-ocr-ru", f"{split}-*.parquet")))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


CATEGORY_OF = {}  # нормализованное название -> категория (data/real_items_labeled.xlsx)


def load_categories():
    from preprocess import normalize
    df = pd.read_excel(os.path.join(HERE, "data", "real_items_labeled.xlsx"))
    for name, cat in zip(df["после нормализации"], df["категория"]):
        CATEGORY_OF[str(name)] = cat
    return normalize


def target_json(row, with_category=False) -> str:
    items = []
    # у синтетики категории есть в данных, у реальных чеков берём из разметки названий
    given = getattr(row, "item_categories", None)
    given = list(given) if given is not None and not isinstance(given, float) else [None] * len(row.item_texts)
    for name, s, cat_given in zip(row.item_texts, row.item_sums, given):
        price = V.to_number(s)
        if name and price is not None:
            item = {"name": str(name).strip(), "price": round(price, 2)}
            if with_category:
                cat = cat_given or CATEGORY_OF.get(NORMALIZE(str(name)), "другое")
                item["category"] = cat if cat in V.CATEGORIES else "другое"  # «не товар» (модификаторы) -> другое
            items.append(item)
    return json.dumps({"items": items, "total": V.to_number(row.total_text)}, ensure_ascii=False)


NORMALIZE = None
WITH_CATEGORY = False


def make_example(processor, row):
    """Картинка + вопрос + правильный ответ; n_prompt - сколько токенов занимает вопрос (их не учим)."""
    image = V.to_pil(row.image["bytes"])
    question = V.PROMPT_CAT if WITH_CATEGORY else V.PROMPT
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    full = prompt + target_json(row, WITH_CATEGORY) + "<|im_end|>\n"
    enc = processor(text=[full], images=[image], return_tensors="pt")
    n_prompt = processor(text=[prompt], images=[image], return_tensors="pt")["input_ids"].shape[1]
    return enc, n_prompt


def answer_loss(model, enc, n_prompt):
    """Потери только по токенам ответа. Вероятности по всему словарю (~250 тыс. слов) считаем лишь для ответа:
    для всей последовательности с картинкой эта таблица занимает до 2 ГБ видеопамяти - главный источник пиков."""
    ids = enc["input_ids"]
    keep = ids.shape[1] - n_prompt + 1           # позиции от последнего токена вопроса до конца
    logits = model(**enc, logits_to_keep=keep).logits[:, :-1]  # позиция t предсказывает токен t+1
    target = ids[:, n_prompt:]
    return torch.nn.functional.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), target.reshape(-1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--accum", type=int, default=8, help="накопление градиента: эффективный батч")
    ap.add_argument("--val-n", type=int, default=40, help="сколько val-чеков для контроля потерь")
    ap.add_argument("--limit", type=int, default=0, help="взять только первые N train-чеков (для пробного запуска)")
    ap.add_argument("--out", default="vlm_lora")
    ap.add_argument("--with-category", action="store_true", help="учить модель выдавать и категорию товара")
    ap.add_argument("--synthetic", default="", help="parquet с синтетическими чеками, добавляется к train")
    ap.add_argument("--vram-fraction", type=float, default=0.9, help="доля видеопамяти, которую можно занять")
    args = ap.parse_args()
    global NORMALIZE, WITH_CATEGORY
    WITH_CATEGORY = args.with_category
    if WITH_CATEGORY:
        NORMALIZE = load_categories()
    random.seed(0), torch.manual_seed(0)
    # на Windows при нехватке видеопамяти драйвер уходит в RAM и файл подкачки,
    # лучше упасть с OOM, чем повесить компьютер
    torch.cuda.set_per_process_memory_fraction(args.vram_fraction)
    proc = psutil.Process()

    processor = AutoProcessor.from_pretrained(V.MODEL_NAME)
    model = AutoModelForImageTextToText.from_pretrained(V.MODEL_NAME, dtype=torch.bfloat16, device_map="cuda")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    # LoRA только на языковую часть: градиенты по картинке не храним,
    # иначе не влезает в 12 ГБ
    targets = [n for n, m in model.named_modules()
               if isinstance(m, torch.nn.Linear) and "language_model" in n and "lm_head" not in n]
    print(f"LoRA-слоёв: {len(targets)}")
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                                             target_modules=targets, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    train = load_split("train")
    train = train[train.item_texts.map(len) > 0]  # чеки без разметки товаров ничему не научат
    if args.limit:
        train = train.head(args.limit)
    if args.synthetic:  # синтетические дорогие чеки
        syn = pd.read_parquet(os.path.join(HERE, args.synthetic))
        print(f"+ синтетических чеков: {len(syn)}")
        train = pd.concat([train, syn], ignore_index=True)
    val = load_split("val")
    val = val[val.item_texts.map(len) > 0].head(args.val_n)
    print(f"train {len(train)} чеков, val (контроль потерь) {len(val)}")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    steps = math.ceil(len(train) * args.epochs / args.accum)
    sched = get_cosine_schedule_with_warmup(opt, int(0.05 * steps), steps)

    @torch.no_grad()
    def val_loss():
        model.eval()
        losses = []
        for r in val.itertuples():
            enc, n_prompt = make_example(processor, r)
            losses.append(answer_loss(model, enc.to(model.device), n_prompt).item())
        model.train()
        return sum(losses) / len(losses)

    print(f"val loss до обучения: {val_loss():.4f}")
    best = float("inf")
    model.train()
    for epoch in range(1, args.epochs + 1):
        rows = list(train.itertuples())
        random.shuffle(rows)
        t0, running, skipped = time.time(), [], 0
        for i, row in enumerate(rows, 1):
            enc, n_prompt = make_example(processor, row)
            try:
                loss = answer_loss(model, enc.to(model.device), n_prompt) / args.accum
                loss.backward()
                running.append(loss.item() * args.accum)
            except torch.OutOfMemoryError:
                # слишком длинный чек не влез в память - пропускаем
                skipped += 1
                print(f"  пропущен чек {row.image_id}: не хватило видеопамяти", flush=True)
                del enc
                loss = None
                torch.cuda.empty_cache()
            if i % args.accum == 0 or i == len(rows):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(), sched.step(), opt.zero_grad()
                torch.cuda.empty_cache()
            if i % 10 == 0:
                print(f"  эпоха {epoch}: {i}/{len(rows)}  loss={sum(running[-50:]) / len(running[-50:]):.4f}  "
                      f"{(time.time() - t0) / i:.1f} с/чек  VRAM пик {torch.cuda.max_memory_allocated() / 2**30:.1f} ГБ  "
                      f"RAM процесса {proc.memory_info().rss / 2**30:.1f} ГБ",
                      flush=True)
        vl = val_loss()
        print(f"эпоха {epoch}: train loss {sum(running) / len(running):.4f}, val loss {vl:.4f}, пропущено {skipped}, "
              f"{(time.time() - t0) / len(rows):.1f} с/чек, VRAM пик {torch.cuda.max_memory_allocated() / 2**30:.1f} ГБ", flush=True)
        if vl < best:
            best = vl
            model.save_pretrained(os.path.join(HERE, args.out))
            with open(os.path.join(HERE, args.out, "receipt_task.json"), "w", encoding="utf-8") as f:
                json.dump({"category": WITH_CATEGORY}, f)
            print(f"  сохранено в {args.out}/")
    print(f"лучший val loss {best:.4f}")


if __name__ == "__main__":
    main()
