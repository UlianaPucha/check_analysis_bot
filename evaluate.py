import env_setup  # noqa: F401 (до transformers)
import argparse
import glob
import json
import os
import time
from difflib import SequenceMatcher

import pandas as pd

from preprocess import normalize

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data", "receipt-ocr-ru")
NAME_MATCH = 0.6  # порог похожести названий


def to_float(s) -> float | None:
    try:
        return float(str(s).replace(",", ".").replace(" ", ""))
    except (TypeError, ValueError):
        return None


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize(a), normalize(b)).ratio()


def match_items(gt: list[dict], pred: list[dict]) -> list[tuple[int, int, float]]:
    """Жадно сопоставляем товары из разметки и найденные ботом.
    Пара засчитывается, если названия похожи или (название хоть немного похоже и совпала цена)."""
    pairs = []
    for i, g in enumerate(gt):
        for j, p in enumerate(pred):
            sim = similarity(g["name"], p["name"])
            same_price = g["sum"] is not None and abs(g["sum"] - p["price"]) < 0.01
            score = sim + (0.3 if same_price else 0)
            if sim >= NAME_MATCH or (same_price and sim >= 0.3):
                pairs.append((score, i, j, sim))
    used_g, used_p, result = set(), set(), []
    for _, i, j, sim in sorted(pairs, reverse=True):
        if i not in used_g and j not in used_p:
            used_g.add(i), used_p.add(j)
            result.append((i, j, sim))
    return result


def load_split(split: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(DATA_DIR, f"{split}-*.parquet")))
    if not files:
        raise SystemExit(f"Нет файлов {split}-*.parquet в {DATA_DIR}. Скачай датасет (см. README).")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "val", "train"])
    ap.add_argument("--limit", type=int, default=0, help="сколько чеков взять (0 = все)")
    ap.add_argument("--name", default=None, help="имя папки с отчётом в reports/")
    ap.add_argument("--worst", type=int, default=25, help="сколько худших чеков сохранить картинками")
    args = ap.parse_args()

    df = load_split(args.split)
    if args.limit:
        df = df.head(args.limit)
    import vlm_reader

    def parse_receipt(image_bytes):
        items, total, _ = vlm_reader.extract(image_bytes)
        return items, total

    run_name = args.name or f"{args.split}_{os.path.basename(vlm_reader.ADAPTER or 'base')}"
    out_dir = os.path.join(HERE, "reports", run_name)
    os.makedirs(os.path.join(out_dir, "worst"), exist_ok=True)

    receipt_rows, item_rows = [], []
    for n, r in enumerate(df.itertuples(), 1):
        gt = [{"name": t, "sum": to_float(s)} for t, s in zip(r.item_texts, r.item_sums)]
        gt_total = to_float(r.total_text)
        image_bytes = r.image["bytes"]

        start = time.time()
        try:
            pred, pred_total = parse_receipt(image_bytes)
            error = ""
        except Exception as e:  # ошибка на одном чеке не должна ронять замер
            pred, pred_total, error = [], None, repr(e)
        seconds = time.time() - start

        pairs = match_items(gt, pred)
        price_ok = sum(1 for i, j, _ in pairs if gt[i]["sum"] is not None and abs(gt[i]["sum"] - pred[j]["price"]) < 0.01)
        pred_sum = sum(p["price"] for p in pred)
        receipt_rows.append({
            "image_id": r.image_id, "seller": r.seller_text, "gt_items": len(gt), "pred_items": len(pred),
            "matched": len(pairs), "price_ok": price_ok,
            "recall": len(pairs) / len(gt) if gt else None,
            "precision": len(pairs) / len(pred) if pred else None,
            "gt_total": gt_total, "pred_total": pred_total, "pred_items_sum": round(pred_sum, 2),
            "total_ok": gt_total is not None and pred_total is not None and abs(gt_total - pred_total) < 0.01,
            "sum_ok": gt_total is not None and abs(gt_total - pred_sum) <= 1,
            "seconds": round(seconds, 1), "error": error,
        })
        matched_g = {i: (j, sim) for i, j, sim in pairs}
        for i, g in enumerate(gt):
            j, sim = matched_g.get(i, (None, 0))
            item_rows.append({"image_id": r.image_id, "gt_name": g["name"], "gt_sum": g["sum"],
                              "pred_name": pred[j]["name"] if j is not None else "",
                              "pred_price": pred[j]["price"] if j is not None else None,
                              "similarity": round(sim, 2)})
        matched_p = {j for _, j, _ in pairs}
        for j, p in enumerate(pred):
            if j not in matched_p:  # лишний товар (нет в разметке)
                item_rows.append({"image_id": r.image_id, "gt_name": "", "gt_sum": None,
                                  "pred_name": p["name"], "pred_price": p["price"], "similarity": 0})
        rec = receipt_rows[-1]
        print(f"[{n}/{len(df)}] {r.image_id}: товаров {len(pairs)}/{len(gt)} найдено, "
              f"лишних {len(pred) - len(pairs)}, цен верно {price_ok}, итог {'✓' if rec['total_ok'] else '✗'}, {seconds:.1f}с", flush=True)

    receipts = pd.DataFrame(receipt_rows)
    items = pd.DataFrame(item_rows)
    gt_n, pred_n, match_n = receipts.gt_items.sum(), receipts.pred_items.sum(), receipts.matched.sum()
    precision = match_n / pred_n if pred_n else 0
    recall = match_n / gt_n if gt_n else 0
    matched_items = items[items.pred_name.ne("") & items.gt_name.ne("")]
    summary = {
        "split": args.split, "receipts": len(receipts), "adapter": vlm_reader.ADAPTER,
        "gt_items": int(gt_n), "pred_items": int(pred_n), "matched_items": int(match_n),
        "item_recall": round(recall, 3),       # полнота
        "item_precision": round(precision, 3),  # точность
        "item_f1": round(2 * precision * recall / (precision + recall), 3) if precision + recall else 0,
        "price_accuracy": round(receipts.price_ok.sum() / match_n, 3) if match_n else 0,  # среди найденных
        "name_similarity": round(matched_items.similarity.mean(), 3) if len(matched_items) else 0,
        "total_found_ok": round(receipts.total_ok.mean(), 3),  # итог совпал
        "items_sum_ok": round(receipts.sum_ok.mean(), 3),       # сумма позиций = итог (±1 ₽)
        "receipts_without_items": int((receipts.pred_items == 0).sum()),
        "errors": int(receipts.error.ne("").sum()),
        "seconds_per_receipt": round(receipts.seconds.mean(), 1),
    }
    receipts.to_csv(os.path.join(out_dir, "receipts.csv"), index=False, encoding="utf-8-sig")
    items.to_csv(os.path.join(out_dir, "items.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # худшие чеки сохраняем картинками для разбора
    receipts["badness"] = (1 - receipts.recall.fillna(0)) + (1 - receipts.precision.fillna(0))
    worst = receipts.sort_values("badness", ascending=False).head(args.worst)
    images = dict(zip(df.image_id, df.image))
    for rank, rid in enumerate(worst.image_id, 1):
        with open(os.path.join(out_dir, "worst", f"{rank:02d}_{rid}.png"), "wb") as f:
            f.write(images[rid]["bytes"])

    print("\n" + json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Отчёт: {out_dir}")


if __name__ == "__main__":
    main()
