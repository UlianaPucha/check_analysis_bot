import env_setup  # noqa: F401 (до transformers)
import argparse
import json
import os
import random
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

import query_model as Q
import vlm_reader as V

HERE = os.path.dirname(os.path.abspath(__file__))


def read(path):
    return [json.loads(line) for line in open(os.path.join(HERE, path), encoding="utf-8")]


def answer_text(query: dict) -> str:
    return json.dumps(query, ensure_ascii=False)


class Tok:  # без картинок хватает токенизатора
    def __init__(self, tok):
        self.tokenizer = tok

    def apply_chat_template(self, *a, **k):
        return self.tokenizer.apply_chat_template(*a, **k)


def evaluate(model, tok, data, show=0):
    model.eval()
    full = 0
    fields = {k: 0 for k in ("metric", "period", "category", "text")}
    for n, ex in enumerate(data):
        prompt = Q.chat_prompt(Tok(tok), ex["question"])
        enc = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=80, do_sample=False)
        pred = Q.parse(tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)) or {}
        ok = all(pred.get(k) == ex["query"].get(k) for k in fields)
        full += ok
        for k in fields:
            fields[k] += pred.get(k) == ex["query"].get(k)
        if n < show or (show and not ok):
            print(f"   {'✓' if ok else '✗'} {ex['question']!r} -> {pred}" + ("" if ok else f"   (нужно {ex['query']})"))
    model.train()
    return full / len(data), {k: v / len(data) for k, v in fields.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--out", default="query_lora")
    args = ap.parse_args()
    random.seed(0), torch.manual_seed(0)
    torch.cuda.set_per_process_memory_fraction(0.9)

    tok = AutoTokenizer.from_pretrained(V.MODEL_NAME)
    # грузим полную модель (картинка + текст), чтобы адаптер подходил к модели бота
    from transformers import AutoModelForImageTextToText
    model = AutoModelForImageTextToText.from_pretrained(V.MODEL_NAME, dtype=torch.bfloat16, device_map="cuda")
    targets = [n for n, m in model.named_modules()
               if isinstance(m, torch.nn.Linear) and "language_model" in n and "lm_head" not in n]
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=targets,
                                             task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    train, test = read("data/query_train.jsonl"), read("data/query_test.jsonl")
    acc, fields = evaluate(model, tok, test)
    print(f"до обучения: полностью верных запросов {acc:.0%}, по полям {fields}")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)
    bs = 8
    steps = args.epochs * ((len(train) + bs - 1) // bs)
    sched = get_cosine_schedule_with_warmup(opt, int(0.05 * steps), steps)
    model.train()
    t0, step = time.time(), 0
    for epoch in range(args.epochs):
        random.shuffle(train)
        for s in range(0, len(train), bs):
            losses = []
            for ex in train[s:s + bs]:
                prompt = Q.chat_prompt(Tok(tok), ex["question"])
                full = prompt + answer_text(ex["query"]) + "<|im_end|>\n"
                ids = tok(full, return_tensors="pt")["input_ids"].to(model.device)
                n_prompt = len(tok(prompt)["input_ids"])
                labels = ids.clone()
                labels[:, :n_prompt] = -100
                loss = model(input_ids=ids, labels=labels).loss / bs
                loss.backward()
                losses.append(loss.item() * bs)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(), sched.step(), opt.zero_grad()
            step += 1
            if step % 50 == 0:
                print(f"  шаг {step}/{steps}  loss {sum(losses) / len(losses):.4f}  {time.time() - t0:.0f} с", flush=True)
    acc, fields = evaluate(model, tok, test, show=len(test))
    print(f"после обучения: полностью верных запросов {acc:.0%}, по полям {fields}")
    model.save_pretrained(os.path.join(HERE, args.out))
    print(f"сохранено в {args.out}/")


if __name__ == "__main__":
    main()
