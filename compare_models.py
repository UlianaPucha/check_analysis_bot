import env_setup  # noqa: F401 (до transformers)
import os
import sys

import pandas as pd

import vlm_reader as V
from evaluate import match_items, to_float

HERE = os.path.dirname(os.path.abspath(__file__))


def score(gt, pred):
    pairs = match_items(gt, pred)
    ok = sum(1 for i, j, _ in pairs if gt[i]["sum"] is not None and abs(gt[i]["sum"] - pred[j]["price"]) < 0.01)
    return len(pairs), ok


def show(title, image_bytes, gt=None, gt_total=None):
    before, t_before, _ = V.extract(image_bytes, use_adapter=False)
    after, t_after, _ = V.extract(image_bytes, use_adapter=True)
    print("=" * 100 + f"\n{title}")
    if gt is not None:
        mb, pb = score(gt, before)
        ma, pa = score(gt, after)
        print(f"  товаров в разметке {len(gt)}, итог {gt_total}")
        print(f"  ДО:    найдено {mb}/{len(gt)}, цен верно {pb}, лишних {len(before) - mb}, итог {t_before}")
        print(f"  ПОСЛЕ: найдено {ma}/{len(gt)}, цен верно {pa}, лишних {len(after) - ma}, итог {t_after}")
    width = 48
    print(f"  {'ДО (исходная модель)':<{width}} | ПОСЛЕ (с нашим адаптером)")
    for k in range(max(len(before), len(after))):
        b = f"{before[k]['name'][:34]} {before[k]['price']:.2f}" if k < len(before) else ""
        a = f"{after[k]['name'][:34]} {after[k]['price']:.2f}" if k < len(after) else ""
        print(f"  {b:<{width}} | {a}")


def main():
    if not V.ADAPTER:
        sys.exit("Нет адаптера vlm_lora_2b_cat — сравнивать нечего")
    if len(sys.argv) > 1:
        for path in sys.argv[1:]:
            show(path, open(path, "rb").read())
        return
    # чеки с самой большой разницей до/после
    before = pd.read_csv(os.path.join(HERE, "reports", "test_qwen35_2b", "receipts.csv"))
    after = pd.read_csv(os.path.join(HERE, "reports", "test_qwen35_2b_lora", "receipts.csv"))
    diff = before[["image_id", "matched", "price_ok"]].merge(after[["image_id", "matched", "price_ok"]],
                                                             on="image_id", suffixes=("_before", "_after"))
    diff["gain"] = (diff.matched_after + diff.price_ok_after) - (diff.matched_before + diff.price_ok_before)
    ids = list(diff.sort_values("gain", ascending=False).image_id.head(5))
    df = pd.read_parquet(os.path.join(HERE, "data", "receipt-ocr-ru", "test-00000-of-00001.parquet"))
    for rid in ids:
        r = df[df.image_id == rid].iloc[0]
        gt = [{"name": t, "sum": to_float(s)} for t, s in zip(r.item_texts, r.item_sums)]
        show(f"{rid} ({r.seller_text})", r.image["bytes"], gt, r.total_text)


if __name__ == "__main__":
    main()
