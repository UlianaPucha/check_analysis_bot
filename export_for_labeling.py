import env_setup  # noqa: F401 (до transformers)
import glob
import os

import numpy as np
import pandas as pd
import torch
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from preprocess import normalize

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data", "receipt-ocr-ru")
OUT = os.path.join(HERE, "data", "real_items_to_label.xlsx")
NOT_ITEM = "не товар"  # мусорные строки разметки (ФИО, обрывки), в обучение не идут


def load_names() -> pd.DataFrame:
    rows = []
    for split in ["train", "val", "test"]:
        for f in sorted(glob.glob(os.path.join(DATA_DIR, f"{split}-*.parquet"))):
            df = pd.read_parquet(f, columns=["seller_text", "item_texts"])
            for seller, items in zip(df.seller_text, df.item_texts):
                rows += [(split, seller or "", t) for t in items]
    d = pd.DataFrame(rows, columns=["split", "seller", "name"])
    d["norm"] = d["name"].map(normalize)
    d = d[d["norm"].str.len() > 1]

    # одно нормализованное название = один товар.
    # выборка: train, если встречается в train-чеках, иначе val, иначе test
    # (в test только названия, которых модель не видела)
    def pick_split(s):
        return next(x for x in ["train", "val", "test"] if x in set(s))

    g = d.groupby("norm").agg(
        name=("name", lambda s: s.value_counts().index[0]),  # самое частое написание
        count=("name", "size"),
        split=("split", pick_split),
        shop=("seller", lambda s: s.value_counts().index[0][:40]),
    ).reset_index()
    return g


def suggest(names: list[str]) -> tuple[list[str], np.ndarray]:
    model_dir = os.path.join(HERE, "model_real")  # категории предлагает текущий классификатор
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(names), 128):
            enc = tok(names[i:i + 128], padding=True, truncation=True, max_length=64, return_tensors="pt")
            probs.append(torch.softmax(model(**enc).logits, -1).numpy())
    p = np.vstack(probs)
    return [model.config.id2label[i] for i in p.argmax(1)], p.max(1), list(model.config.id2label.values())


def main():
    g = load_names()
    pred, conf, labels = suggest(list(g["norm"]))
    g["предложено моделью"] = pred
    g["уверенность"] = conf.round(2)
    g["категория"] = pred  # предзаполняем, человек правит только ошибки
    g = g.sort_values(["уверенность", "count"], ascending=[True, False])
    out = g.rename(columns={"name": "название из чека", "count": "раз в чеках", "shop": "магазин",
                            "split": "выборка", "norm": "после нормализации"})
    out = out[["название из чека", "категория", "предложено моделью", "уверенность", "раз в чеках",
               "магазин", "выборка", "после нормализации"]]
    out.to_excel(OUT, index=False, sheet_name="разметка")

    # выпадающий список категорий, подсветка, ширина колонок
    wb = load_workbook(OUT)
    ws = wb.active
    options = labels + [NOT_ITEM]
    dv = DataValidation(type="list", formula1='"' + ",".join(options) + '"', allow_blank=False,
                        showErrorMessage=True, error="Выбери категорию из списка")
    ws.add_data_validation(dv)
    dv.add(f"B2:B{ws.max_row}")
    yellow = PatternFill("solid", fgColor="FFF2CC")
    for row in range(2, ws.max_row + 1):
        if ws.cell(row, 4).value < 0.6:
            for col in range(1, 5):
                ws.cell(row, col).fill = yellow
    for cell in ws[1]:
        cell.font = Font(bold=True)
    for col, width in zip("ABCDEFGH", [45, 14, 18, 12, 12, 30, 9, 40]):
        ws.column_dimensions[col].width = width
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    wb.save(OUT)

    print(f"Названий: {len(g)} (train {sum(g.split == 'train')}, val {sum(g.split == 'val')}, "
          f"test {sum(g.split == 'test')}); неуверенных (<0.6): {(g['уверенность'] < 0.6).sum()}")
    print("Предложенные категории:", g["предложено моделью"].value_counts().to_dict())
    print(f"Файл: {OUT}")


if __name__ == "__main__":
    main()
