import argparse
import io
import os
import random

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from make_dataset import PRODUCTS

HERE = os.path.dirname(os.path.abspath(__file__))
FONTS = [r"C:\Windows\Fonts\consola.ttf", r"C:\Windows\Fonts\cour.ttf", r"C:\Windows\Fonts\courbd.ttf",
         r"C:\Windows\Fonts\lucon.ttf", r"C:\Windows\Fonts\consolab.ttf"]
FONTS = [f for f in FONTS if os.path.exists(f)]
SHOPS = ['ООО "КЕЛЬТИКА"', 'ООО "РЕСТОРАН ХИНКАЛЬНАЯ"', 'ИП Смирнов А.В.', 'ООО "БАР НА РЕКЕ"', 'ООО "ГАСТРОНОМ"',
         'АО "ТОРГОВЫЙ ДОМ ПЕРЕКРЕСТОК"', 'ООО "СПОРТМАСТЕР"', 'ООО "АПТЕКА ЗДОРОВЬЕ"', 'ООО "КОФЕЙНЯ ЗЕРНО"']
CAT_PRICE = {  # диапазоны цен за штуку (от них зависит доля сумм больше 1000)
    "еда": (60, 1300), "напитки": (50, 1500), "бытовое": (50, 1100), "здоровье": (60, 1400),
    "одежда": (400, 6000), "транспорт": (150, 3000), "развлечения": (150, 3000), "другое": (20, 700)}


def money(x: float, style: str) -> str:
    rub, kop = int(x), round((x - int(x)) * 100)
    sep = {"space": " ", "nbsp": " ", "none": ""}[style]
    s = f"{rub:,}".replace(",", sep)
    return f"{s},{kop:02d}" if random.random() < 0.6 else f"{s}.{kop:02d}"


def pick_items():
    n = random.randint(2, 12)
    items = []
    for _ in range(n):
        cat = random.choices(list(CAT_PRICE), weights=[30, 25, 10, 8, 7, 5, 5, 10])[0]
        name = random.choice(PRODUCTS[cat])
        name = name.upper() if random.random() < 0.5 else name
        lo, hi = CAT_PRICE[cat]
        price = round(random.uniform(lo, hi) / 10) * 10 - (0.01 if random.random() < 0.3 else 0)
        if random.random() < 0.08:
            price = round(random.uniform(1000, 9999), 2)  # плюс круглые тысячи с копейками
        qty = random.choices([1, 2, 3, 4], weights=[70, 18, 8, 4])[0]
        items.append({"name": name, "qty": qty, "price": price, "sum": round(price * qty, 2), "category": cat})
    return items


def render(items, total):
    font_path = random.choice(FONTS)
    size = random.randint(22, 30)
    f = ImageFont.truetype(font_path, size)
    width_chars = random.choice([34, 38, 42])
    style = random.choices(["space", "nbsp", "none"], weights=[65, 15, 20])[0]
    layout = random.choice(["table", "fiscal", "simple"])
    lines = [f"{random.choice(SHOPS):^{width_chars}}", f"ИНН {random.randint(10**9, 10**10 - 1)}"]
    if layout == "table":
        lines += ["ГОСТЕВОЙ СЧЕТ".center(width_chars), f"Стол: {random.randint(1, 60)}   Заказ № {random.randint(1000, 99999)}",
                  "-" * width_chars, "Наименование".ljust(width_chars - 14) + "Кол-во   Сумма", "-" * width_chars]
        for it in items:
            right = f"{it['qty']:>3}  {money(it['sum'], style):>10}"
            name = it["name"]
            first = name[: width_chars - len(right) - 1]
            lines.append(first.ljust(width_chars - len(right)) + right)
            if len(name) > len(first):
                lines.append(" " + name[len(first):][: width_chars - 1])
        lines += ["-" * width_chars, "ИТОГО К ОПЛАТЕ:".ljust(width_chars - 12) + f"{money(total, style):>12}"]
    elif layout == "fiscal":
        lines += ["КАССОВЫЙ ЧЕК / ПРИХОД".center(width_chars), ""]
        for i, it in enumerate(items, 1):
            lines.append(f"{i}. {it['name']}"[:width_chars])
            calc = f"{it['qty']} x {money(it['price'], style)}"
            lines.append(calc + f"={money(it['sum'], style)}".rjust(width_chars - len(calc)))
            if random.random() < 0.4:
                lines.append("НДС 22%".rjust(width_chars))
        lines += ["-" * width_chars, "ИТОГ" + f"={money(total, style)}".rjust(width_chars - 4),
                  "БЕЗНАЛИЧНЫМИ" + money(total, style).rjust(width_chars - 12)]
    else:
        lines += [""]
        for it in items:
            amount = money(it["sum"], style)
            name = (it["name"] if it["qty"] == 1 else f"{it['name']} x{it['qty']}")[: width_chars - len(amount) - 1]
            lines.append(name.ljust(width_chars - len(amount)) + amount)
        lines += ["", "ИТОГО" + money(total, style).rjust(width_chars - 5)]
    lines += ["", "СПАСИБО! ЖДЕМ ВАС СНОВА!".center(width_chars)]

    step = int(size * random.uniform(1.25, 1.5))
    char_w = f.getbbox("Ш")[2]
    w, h = char_w * width_chars + 60, step * len(lines) + 80
    paper = random.randint(232, 255)
    img = Image.new("L", (w, h), paper)
    d = ImageDraw.Draw(img)
    ink = random.randint(0, 70)
    for i, line in enumerate(lines):
        d.text((30, 40 + i * step), line, font=f, fill=ink)
    return photo_like(np.array(img))


def photo_like(gray: np.ndarray) -> np.ndarray:
    """Делаем «как фото телефоном»: фон, наклон, перспектива, тень, размытие, шум, JPEG."""
    h, w = gray.shape
    bg_val = random.randint(30, 200)
    pad = random.randint(30, 200)
    canvas = np.full((h + 2 * pad, w + 2 * pad), bg_val, np.uint8)
    canvas[pad:pad + h, pad:pad + w] = gray
    H, W = canvas.shape
    angle = random.uniform(-4, 4)
    M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
    canvas = cv2.warpAffine(canvas, M, (W, H), borderValue=bg_val)
    if random.random() < 0.6:  # перспектива
        d = 0.04 * min(W, H)
        src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
        dst = src + np.float32(np.random.uniform(-d, d, (4, 2)))
        canvas = cv2.warpPerspective(canvas, cv2.getPerspectiveTransform(src, dst), (W, H), borderValue=bg_val)
    if random.random() < 0.5:  # тень/неравномерный свет
        grad = np.tile(np.linspace(random.uniform(0.6, 1), 1, W), (H, 1))
        canvas = (canvas * grad).astype(np.uint8)
    if random.random() < 0.5:
        k = random.choice([3, 5])
        canvas = cv2.GaussianBlur(canvas, (k, k), 0)
    noise = np.random.normal(0, random.uniform(2, 9), canvas.shape)
    canvas = np.clip(canvas + noise, 0, 255).astype(np.uint8)
    color = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    if random.random() < 0.5:  # тёплый оттенок как от лампы
        color = np.clip(color * np.array([0.9, 0.97, 1.05]), 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", color, [cv2.IMWRITE_JPEG_QUALITY, random.randint(60, 92)])
    return buf.tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    random.seed(args.seed), np.random.seed(args.seed)
    rows = []
    for k in range(args.n):
        items = pick_items()
        total = round(sum(it["sum"] for it in items), 2)
        rows.append({
            "image_id": f"synthetic_{k:04d}", "image": {"bytes": render(items, total)},
            "item_texts": [it["name"] for it in items], "item_sums": [f"{it['sum']:.2f}" for it in items],
            "item_categories": [it["category"] for it in items], "total_text": f"{total:.2f}",
        })
    df = pd.DataFrame(rows)
    out = os.path.join(HERE, "data", "synthetic_receipts.parquet")
    df.to_parquet(out)
    share = np.mean([float(s) >= 1000 for r in rows for s in r["item_sums"]])
    print(f"{len(df)} чеков, позиций {sum(len(r['item_sums']) for r in rows)}, из них от 1000 ₽: {share:.0%} -> {out}")
    for k in range(5):  # примеры для просмотра
        with open(os.path.join(HERE, "data", f"synthetic_example_{k}.jpg"), "wb") as fh:
            fh.write(rows[k]["image"]["bytes"])


if __name__ == "__main__":
    main()
