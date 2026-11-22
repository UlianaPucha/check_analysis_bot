import argparse
import os
import random

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from make_dataset import PRODUCTS
from make_synthetic_receipts import photo_like

HERE = os.path.dirname(os.path.abspath(__file__))
FONTS = [os.path.join(r"C:\Windows\Fonts", f) for f in
         ["arial.ttf", "arialbd.ttf", "calibri.ttf", "times.ttf", "georgia.ttf", "verdana.ttf", "tahoma.ttf",
          "consola.ttf", "cour.ttf", "Candara.ttf", "segoeui.ttf", "trebuc.ttf"]]
FONTS = [f for f in FONTS if os.path.exists(f)]
WORDS = ("система память данные город время работа проект вопрос дорога решение пример история место человек "
         "развитие условие процесс результат музей парк книга студент неделя встреча письмо задача модель погода "
         "магазин семья праздник поездка обучение программа экзамен лекция статья исследование").split()


def sentence(rnd, k=None):
    w = [rnd.choice(WORDS) for _ in range(k or rnd.randint(5, 11))]
    return (" ".join(w)).capitalize() + rnd.choice([".", ".", ",", ":", "!"])


def lines_image(rnd, lines, size=None, width=None, bg=None, fg=None, font=None, align_center=False):
    font = font or rnd.choice(FONTS)
    size = size or rnd.randint(20, 34)
    f = ImageFont.truetype(font, size)
    width = width or rnd.randint(600, 900)
    step = int(size * rnd.uniform(1.3, 1.6))
    h = 60 + step * len(lines)
    bg = bg or tuple(rnd.randint(225, 255) for _ in range(3))
    fg = fg or tuple(rnd.randint(0, 60) for _ in range(3))
    img = Image.new("RGB", (width, h), bg)
    d = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        x = 30
        if align_center:
            x = max(10, (width - d.textlength(line, font=f)) // 2)
        d.text((x, 30 + i * step), line, font=f, fill=fg)
    return img


def text_page(rnd):
    lines = [f"Глава {rnd.randint(1, 12)}. {sentence(rnd, 4)[:-1]}", ""]
    for _ in range(rnd.randint(8, 20)):
        lines.append(sentence(rnd))
    return lines_image(rnd, lines)


def menu(rnd):
    cat = rnd.choice(["еда", "напитки"])
    items = rnd.sample(PRODUCTS[cat], rnd.randint(5, 12))
    lines = [rnd.choice(["МЕНЮ", "Меню кафе", "НАПИТКИ", "Горячие блюда", "Завтраки"]), ""]
    for name in items:
        price = rnd.randint(9, 99) * 10
        lines.append(f"{name[:28]:.<34} {price} ₽")
    lines += ["", rnd.choice(["Ждём вас ежедневно!", "Работаем с 8:00 до 23:00", "Wi-Fi: cafe_guest"])]
    return lines_image(rnd, lines, align_center=rnd.random() < 0.3)


def price_tag(rnd):
    cat = rnd.choice(list(PRODUCTS))
    name = rnd.choice(PRODUCTS[cat])
    price = rnd.randint(30, 3000) + rnd.choice([0, 0.99, 0.49])
    bg = rnd.choice([(255, 236, 120), (255, 255, 255), (250, 210, 210), (220, 240, 255)])
    return lines_image(rnd, [name[:30].upper(), "", f"{price:.2f} ₽".replace(".", ","), "",
                             f"цена за 1 шт   {rnd.randint(1, 28):02d}.{rnd.randint(1, 12):02d}.2026"],
                       size=rnd.randint(30, 44), width=rnd.randint(520, 700), bg=bg, align_center=True)


def transfer(rnd):
    who = rnd.choice(["Иван Петрович С.", "Анна К.", "Михаил Сергеевич Л.", "Ольга Н."])
    amount = rnd.choice([500, 1500, 2300, 10000, 750, 3200]) + rnd.choice([0, 0, 50])
    lines = [rnd.choice(["Перевод выполнен", "Платёж успешно отправлен", "Перевод по номеру телефона"]), "",
             f"{amount:,} ₽".replace(",", " "), "", f"Получатель: {who}",
             f"Телефон: +7 9{rnd.randint(10, 99)} ***-**-{rnd.randint(10, 99)}",
             f"Банк получателя: {rnd.choice(['Т-Банк', 'Сбербанк', 'ВТБ', 'Альфа-Банк'])}",
             "Комиссия: 0 ₽", f"{rnd.randint(1, 28):02d}.{rnd.randint(1, 12):02d}.2026 {rnd.randint(8, 23)}:{rnd.randint(10, 59)}"]
    return lines_image(rnd, lines, bg=rnd.choice([(255, 255, 255), (245, 246, 250), (30, 30, 35)]),
                       fg=None, align_center=True)


def terminal_slip(rnd):
    amount = rnd.uniform(100, 9000)
    lines = [rnd.choice(["ПАО СБЕРБАНК", "ВТБ (ПАО)", "АО АЛЬФА-БАНК"]), "ОПЛАТА",
             f"ТЕРМИНАЛ {rnd.randint(10**7, 10**8 - 1)}", f"МЕРЧАНТ {rnd.randint(10**6, 10**7 - 1)}",
             f"КАРТА ****{rnd.randint(1000, 9999)}", "", f"СУММА (RUB)  {amount:,.2f}".replace(",", " "),
             "КОМИССИЯ  0.00", "", "ОДОБРЕНО", f"КОД АВТОРИЗАЦИИ {rnd.randint(100000, 999999)}",
             "ПОДПИСЬ КЛИЕНТА НЕ ТРЕБУЕТСЯ"]
    return lines_image(rnd, lines, font=rnd.choice([f for f in FONTS if "cour" in f or "consola" in f] or FONTS))


def table(rnd):
    lines = [rnd.choice(["Расписание занятий", "Список дел", "График дежурств", "Оценки за четверть"]), ""]
    days = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб"]
    for d in rnd.sample(days, rnd.randint(4, 6)):
        lines.append(f"{d}   {rnd.randint(8, 18)}:00   {rnd.choice(WORDS).capitalize()} {rnd.choice(WORDS)}")
    return lines_image(rnd, lines)


def plain(rnd):
    w, h = rnd.randint(600, 1200), rnd.randint(800, 1600)
    kind = rnd.choice(["white", "color", "gradient", "noise", "dark"])
    if kind == "white":
        arr = np.full((h, w, 3), rnd.randint(225, 255), np.uint8)
    elif kind == "color":
        arr = np.full((h, w, 3), [rnd.randint(0, 255) for _ in range(3)], np.uint8)
    elif kind == "gradient":
        g = np.tile(np.linspace(rnd.randint(0, 120), rnd.randint(130, 255), w, dtype=np.uint8), (h, 1))
        arr = np.stack([g] * 3, -1)
    elif kind == "noise":
        arr = np.random.default_rng(rnd.randint(0, 10**6)).integers(0, 255, (h, w, 3), dtype=np.uint8)
    else:
        arr = np.full((h, w, 3), rnd.randint(0, 30), np.uint8)
    return Image.fromarray(arr)


KINDS = [(text_page, 18), (menu, 18), (price_tag, 14), (transfer, 14), (terminal_slip, 14), (table, 8), (plain, 14)]


def to_photo(rnd, img: Image.Image) -> bytes:
    if rnd.random() < 0.75:
        return photo_like(np.array(img.convert("L")))
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buf.tobytes()  # часть - как чистые скриншоты


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=450)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default="data/nonreceipts_train.parquet")
    args = ap.parse_args()
    rnd = random.Random(args.seed)
    random.seed(args.seed), np.random.seed(args.seed)  # photo_like использует глобальный random
    funcs, weights = zip(*KINDS)
    rows = []
    for k in range(args.n):
        fn = rnd.choices(funcs, weights)[0]
        rows.append({"image_id": f"nonreceipt_{args.seed}_{k:04d}", "kind": fn.__name__,
                     "image": {"bytes": to_photo(rnd, fn(rnd))}})
    pd.DataFrame(rows).to_parquet(os.path.join(HERE, args.out))
    print(f"{len(rows)} картинок -> {args.out}:", pd.Series([r["kind"] for r in rows]).value_counts().to_dict())


if __name__ == "__main__":
    main()
