import env_setup  # noqa: F401 (до transformers)
import argparse
import copy
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupShuffleSplit
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from preprocess import normalize

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_MODEL = "cointegrated/rubert-tiny2"
NOT_ITEM = "не товар"
SEED = 42


def load_data():
    syn = pd.read_csv(os.path.join(HERE, "data", "products.csv"))
    syn["clean"] = syn["text"].map(normalize)
    # разбиение синтетики как в train.ipynb: по товарам
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    a, b = next(gss.split(syn, groups=syn["group"]))
    syn_train_full, syn_test = syn.iloc[a], syn.iloc[b]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    a, b = next(gss.split(syn_train_full, groups=syn_train_full["group"]))
    syn_train, syn_val = syn_train_full.iloc[a], syn_train_full.iloc[b]

    real = pd.read_excel(os.path.join(HERE, "data", "real_items_labeled.xlsx"))
    real = real.rename(columns={"категория": "category", "выборка": "split", "название из чека": "text"})
    real = real[real["category"].notna() & (real["category"] != NOT_ITEM)].copy()
    real["clean"] = real["text"].map(normalize)
    parts = {s: real[real["split"] == s] for s in ["train", "val", "test"]}
    return syn_train, syn_val, syn_test, parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--real-weight", type=int, default=3, help="во сколько раз повторить реальные примеры в обучении")
    ap.add_argument("--out", default="model_real")
    args = ap.parse_args()

    torch.manual_seed(SEED), np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    syn_train, syn_val, syn_test, real = load_data()
    print(f"синтетика: train {len(syn_train)}, val {len(syn_val)}, test {len(syn_test)}; "
          f"реальные: train {len(real['train'])}, val {len(real['val'])}, test {len(real['test'])}")

    labels = sorted(syn_train["category"].unique())
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    cols = ["clean", "category"]
    # реальных примеров мало, повторяем их, чтобы не утонули в синтетике
    train = pd.concat([syn_train[cols]] + [real["train"][cols]] * args.real_weight, ignore_index=True)

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=len(labels), id2label=id2label, label2id=label2id).to(device)

    def predict(m, t, texts):
        m.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), 128):
                enc = t(list(texts[i:i + 128]), padding=True, truncation=True, max_length=64,
                        return_tensors="pt").to(m.device)
                out += [m.config.id2label[j] for j in m(**enc).logits.argmax(-1).tolist()]
        return out

    bs, lr = 32, 5e-5
    steps = int(np.ceil(len(train) / bs)) * args.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    best, best_state, best_epoch = -1, None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        idx = np.random.permutation(len(train))
        losses = []
        for s in range(0, len(idx), bs):
            part = train.iloc[idx[s:s + bs]]
            enc = tok(list(part["clean"]), padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
            y = torch.tensor([label2id[c] for c in part["category"]], device=device)
            loss = loss_fn(model(**enc).logits, y)
            loss.backward(), opt.step(), sched.step(), opt.zero_grad()
            losses.append(loss.item())
        # лучшая эпоха - по реальной val
        f1_real = f1_score(real["val"]["category"], predict(model, tok, list(real["val"]["clean"])), average="macro")
        f1_syn = f1_score(syn_val["category"], predict(model, tok, list(syn_val["clean"])), average="macro")
        print(f"epoch {epoch:2d}: loss={np.mean(losses):.4f}  val macro-F1 реальные={f1_real:.4f}  синтетика={f1_syn:.4f}")
        if f1_real > best:
            best, best_epoch, best_state = f1_real, epoch, copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= args.patience:
            print("Ранняя остановка")
            break
    model.load_state_dict(best_state)
    print(f"Лучшая эпоха {best_epoch}, val macro-F1 (реальные) = {best:.4f}")

    # сравнение со старой моделью model/, если она есть
    compare = [("новая", model, tok)]
    old_dir = os.path.join(HERE, "model")
    if os.path.isdir(old_dir):
        compare.insert(0, ("старая", AutoModelForSequenceClassification.from_pretrained(old_dir).to(device),
                           AutoTokenizer.from_pretrained(old_dir)))
    report = {}
    for name, df in [("реальный test", real["test"]), ("синтетический test", syn_test)]:
        texts, y = list(df["clean"]), list(df["category"])
        for mname, m, t in compare:
            p = predict(m, t, texts)
            report[f"{name} / {mname}"] = {"accuracy": round(accuracy_score(y, p), 3),
                                          "macro_f1": round(f1_score(y, p, average="macro"), 3)}
            if name == "реальный test" and mname == "новая":
                print(classification_report(y, p, zero_division=0))
    print(json.dumps(report, ensure_ascii=False, indent=2))

    out_dir = os.path.join(HERE, args.out)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"best_epoch": best_epoch, "val_macro_f1_real": round(best, 4), **report}, f,
                  ensure_ascii=False, indent=2)
    print(f"Сохранено в {out_dir}")


if __name__ == "__main__":
    main()
