import env_setup  # noqa: F401 (импортировать первым)
import asyncio
import os
import re
import time
from datetime import datetime, timedelta

import torch
from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.types import (BotCommand, BufferedInputFile, CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup,
                           KeyboardButton, Message, ReplyKeyboardMarkup)
from dotenv import load_dotenv

import demo
import reports
import storage

load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
HERE = os.path.dirname(os.path.abspath(__file__))

if not torch.cuda.is_available():
    raise SystemExit("Не найдена видеокарта NVIDIA с CUDA: модели распознавания чека она нужна.")

import pipeline  # noqa: E402
import query_model  # noqa: E402
import vlm_reader  # noqa: E402

print(f"Загружаю модель распознавания {vlm_reader.MODEL_NAME}"
      f"{' + адаптер ' + vlm_reader.ADAPTER if vlm_reader.ADAPTER else ' (БЕЗ дообучения)'}...")
pipeline.load()

dp = Dispatcher()  # Bot создаём в main(), чтобы модуль импортировался без токена
db = storage.connect()
CATEGORIES = reports.CATEGORIES

# --- тексты и клавиатура ---
BTN_MONTH, BTN_HISTORY, BTN_ADD, BTN_LIMITS, BTN_PRICES, BTN_HELP = (
    "📊 Месяц", "🧾 Мои чеки", "➕ Трата без чека", "🎯 Лимиты", "📈 Цены", "❓ Помощь")
MENU = ReplyKeyboardMarkup(keyboard=[[KeyboardButton(text=BTN_MONTH), KeyboardButton(text=BTN_HISTORY)],
                                     [KeyboardButton(text=BTN_ADD), KeyboardButton(text=BTN_LIMITS)],
                                     [KeyboardButton(text=BTN_PRICES), KeyboardButton(text=BTN_HELP)]],
                           resize_keyboard=True, input_field_placeholder="Пришли фото чека или задай вопрос")
COMMANDS = [BotCommand(command="month", description="Траты за месяц"),
            BotCommand(command="history", description="Мои чеки: открыть и исправить"),
            BotCommand(command="add", description="Добавить трату без чека"),
            BotCommand(command="limits", description="Лимиты по категориям"),
            BotCommand(command="prices", description="Как менялись цены"),
            BotCommand(command="week", description="Траты за 7 дней"),
            BotCommand(command="export", description="Все траты файлом для Excel"),
            BotCommand(command="demo", description="Демо-данные для показа"),
            BotCommand(command="forget", description="Удалить все мои данные"),
            BotCommand(command="help", description="Что умеет бот")]
HELP = ("Я помогаю следить за тратами 🧾\n\n"
        "📷 Пришли фото чека — распознаю покупки, разложу по категориям и сохраню. Лучше отправлять фото "
        "«файлом» (без сжатия) — так мелкий шрифт читается точнее.\n"
        "📊 /month — траты за месяц: диаграммы, сравнение с прошлым месяцем, прогноз, лимиты\n"
        "🧾 /history — все чеки: открыть любой и исправить сумму, название, категорию, дату, магазин\n"
        "➕ /add — трата без чека, например «такси 450»\n"
        "🎯 /limits — лимиты на месяц по категориям\n"
        "📈 /prices — как менялись цены на одни и те же товары\n"
        "📁 /export — все траты файлом для Excel\n"
        "💬 Можно спросить словами: «сколько я потратил на кофе в октябре?», «где я больше всего трачу?»\n\n"
        "🎭 /demo — заполнить историю демо-тратами (для показа), 🗑 /forget — удалить все мои данные")
NOT_RECOGNIZED = ("Не смогла распознать товары на этом фото 😔\n"
                  "Скорее всего, снимок размытый, тёмный, под сильным углом или чек попал в кадр не целиком.\n"
                  "Сфотографируй чек ровно и целиком, при хорошем свете, и пришли ещё раз.")
NOT_A_RECEIPT = ("Не похоже на кассовый чек 🤔\n"
                 "Я разбираю чеки покупок — со списком товаров и ценами. Меню, ценники, переводы и слипы оплаты "
                 "картой я не обрабатываю.\nЕсли это всё-таки чек — сфотографируй его ровно и целиком, при хорошем свете.")
BAD_FILE = "Не получилось открыть картинку — пришли фото чека в формате JPG или PNG."
DATE_SOURCE = {"qr": "по QR-коду", "receipt": "с чека", "user": "указана вручную",
               "sent": "дата не найдена на чеке — поставила день отправки"}

recognize_lock = asyncio.Semaphore(1)  # модель одна на видеокарте, поэтому по одному
DEBUG_DIR = os.path.join(HERE, "debug_receipts") if os.getenv("DEBUG_SAVE") == "1" else None
CORRECTIONS = os.path.join(HERE, "data", "user_corrections.csv")  # исправления пользователей (на будущее дообучение)
CORRECTIONS_HEADER = "что;название;было;стало"
if os.path.exists(CORRECTIONS) and open(CORRECTIONS, encoding="utf-8-sig").readline().strip() != CORRECTIONS_HEADER:
    os.replace(CORRECTIONS, CORRECTIONS.replace(".csv", "_old.csv"))  # файл старого формата переименовываем


# TODO: дообучать модели на этих исправлениях (пока только копим)
def log_correction(kind: str, name: str, old, new) -> None:
    new_file = not os.path.exists(CORRECTIONS)
    with open(CORRECTIONS, "a", encoding="utf-8-sig") as f:
        f.write((CORRECTIONS_HEADER + "\n" if new_file else "") + f"{kind};{name};{old};{new}\n")


def chunks(text: str, limit: int = 3900) -> list[str]:
    """Telegram не принимает сообщения длиннее 4096 символов - делим по строкам."""
    out, cur = [], ""
    for line in text.split("\n"):
        if len(cur) + len(line) + 1 > limit:
            out.append(cur)
            cur = ""
        cur += line + "\n"
    return out + [cur] if cur.strip() else out


async def send(message: Message, text: str, kb=None) -> None:
    parts = chunks(text)
    for i, part in enumerate(parts):
        await message.answer(part, reply_markup=kb if i == len(parts) - 1 else None)


async def show(call: CallbackQuery, text: str, kb=None) -> None:
    try:
        await call.message.edit_text(chunks(text)[0], reply_markup=kb)
    except Exception:
        await send(call.message, text, kb)


def rub(x: float) -> str:
    return reports.rub(x)


# --- ожидание ввода текста ---
PENDING_TTL = 600  # ожидание сбрасывается через 10 минут
pending: dict[int, tuple[str, int, float]] = {}  # user_id -> (что вводим, к чему относится, когда начали)


def set_pending(user_id: int, kind: str, ref: int = 0) -> None:
    pending[user_id] = (kind, ref, time.time())


def get_pending(user_id: int):
    st = pending.get(user_id)
    if st and time.time() - st[2] > PENDING_TTL:
        pending.pop(user_id, None)
        return None
    return st


def parse_amount(text: str) -> float | None:
    """'1 980,50' / '1980.5' / '1 980 ₽' -> 1980.5"""
    clean = text.replace(" ", " ").replace("₽", "").replace("руб", "").strip()
    m = re.search(r"\d[\d ]*(?:[.,]\d{1,2})?", clean)
    if not m:
        return None
    value = float(m.group().replace(" ", "").replace(",", "."))
    return value if 0 < value < 10_000_000 else None


def parse_user_date(text: str) -> datetime | None:
    t = text.strip().lower()
    now = datetime.now()
    if t.startswith("сегодня"):
        return now.replace(second=0, microsecond=0)
    if t.startswith("вчера"):
        return (now - timedelta(days=1)).replace(second=0, microsecond=0)
    if t.startswith("позавчера"):
        return (now - timedelta(days=2)).replace(second=0, microsecond=0)
    m = re.search(r"(\d{1,2})[./](\d{1,2})(?:[./](\d{2,4}))?(?:\s+(\d{1,2})[:.](\d{2}))?", t)
    if not m:
        return None
    d, mo, y, hh, mm = m.groups()
    y = int(y) if y else now.year
    y += 2000 if y < 100 else 0
    try:
        dt = datetime(y, int(mo), int(d), int(hh or 12), int(mm or 0))
    except ValueError:
        return None
    return dt if dt <= now + timedelta(days=1) else None


# --- распознавание чека ---
def format_result(res: pipeline.Result) -> str:
    lines = []
    if res.shop:
        lines.append(f"🏪 {res.shop}")
    when = res.date or datetime.now()
    lines.append(f"📅 {when:%d.%m.%Y %H:%M} — {DATE_SOURCE[res.date_source]}")
    if res.qr and res.qr.get("operation") and res.qr["operation"] != "приход":
        lines.append(f"↩️ Это чек «{res.qr['operation']}» — суммы ниже не расход, а возврат")
    lines += ["", "🧾 Товары:"]
    by_cat = {}
    for it in res.items:
        by_cat[it["category"]] = by_cat.get(it["category"], 0) + it["price"]
        lines.append(f"• {it['name']} — {it['price']:.2f} ₽ ({it['category']})")
    lines += ["", "📊 По категориям:"] + [f"{c}: {a:.2f} ₽" for c, a in sorted(by_cat.items(), key=lambda x: -x[1])]
    lines += ["", f"💰 Сумма позиций: {res.items_sum:.2f} ₽"]
    if res.mismatch:
        src = "по QR-коду" if res.qr else "в чеке (как прочитала модель)"
        lines.append(f"⚠️ Итог {src} — {res.total:.2f} ₽, сумма позиций не сходится. Чек отмечен «требует проверки»: "
                     f"нажми «🔍 Проверить позиции» и поправь неточности.")
    elif res.total and res.qr:
        lines.append("✅ Совпадает с итогом по QR-коду")
    if res.confidence < pipeline.LOW_CONFIDENCE and not res.mismatch:
        lines.append("🔎 Модель не очень уверена в распознавании этого фото — проверь позиции.")
    return "\n".join(lines)


def saved_keyboard(rid: int, needs_review: bool) -> InlineKeyboardMarkup:
    first = [InlineKeyboardButton(text="🔍 Проверить позиции", callback_data=f"open:{rid}"),
             InlineKeyboardButton(text="✅ Всё верно", callback_data=f"ok:{rid}")] if needs_review else \
        [InlineKeyboardButton(text="✏️ Редактировать", callback_data=f"open:{rid}")]
    return InlineKeyboardMarkup(inline_keyboard=[first, [InlineKeyboardButton(text="🗑 Удалить чек",
                                                                               callback_data=f"del:{rid}")]])


async def answer_receipt(message: Message, file_obj) -> None:
    uid = message.from_user.id
    pending.pop(uid, None)  # пришёл чек - отменяем ожидание ввода
    status = await message.answer("Сейчас разбираю другой чек, твой следующий в очереди… ⏳"
                                  if recognize_lock.locked() else "⏳ Получила фото, начинаю…")
    loop = asyncio.get_running_loop()

    async def edit_status(text: str) -> None:
        try:
            await status.edit_text(text)
        except Exception:  # не критично (например, message is not modified)
            pass

    def progress(text: str) -> None:  # вызывается из потока распознавания
        # edit_text() в aiogram 3 не корутина, поэтому оборачиваем
        asyncio.run_coroutine_threadsafe(edit_status(text), loop)

    file = await message.bot.download(file_obj)
    data = file.read()
    if DEBUG_DIR:  # отладка: сохраняем присланные фото
        os.makedirs(DEBUG_DIR, exist_ok=True)
        with open(os.path.join(DEBUG_DIR, f"{time.strftime('%Y%m%d_%H%M%S')}.jpg"), "wb") as f:
            f.write(data)
    try:
        async with recognize_lock:
            started = time.time()
            res = await asyncio.to_thread(pipeline.process, data, progress)
            print(f"Чек от {uid} разобран за {time.time() - started:.1f} с: {res.status}, {len(res.items)} поз.")
    except ValueError:
        await status.edit_text(BAD_FILE)
        return
    except Exception as e:  # текст ошибки только в лог
        print(f"Ошибка при разборе чека: {e!r}")
        await status.edit_text("Что-то пошло не так при разборе чека 😔 Попробуй ещё раз или пришли другое фото.")
        return
    await asyncio.sleep(0.3)  # даём дойти последнему обновлению статуса
    try:
        await status.delete()
    except Exception:
        pass
    if res.status == "not_receipt":
        await message.answer(NOT_A_RECEIPT)
        return
    if res.status == "not_recognized":
        text = NOT_RECOGNIZED
        if res.qr:  # итог из QR точный, его можно показать
            text += f"\n\nПо QR-коду: 📅 {res.qr['datetime']:%d.%m.%Y %H:%M}, итог {res.qr['total']:.2f} ₽"
        await message.answer(text)
        return
    answer = format_result(res)
    dup = storage.find_duplicate(db, uid, data, res.items, res.qr)
    if dup:
        await send(message, answer + f"\n\nℹ️ Этот чек уже сохранён (#{dup}) — второй раз не учитываю.",
                   saved_keyboard(dup, False))
        return
    # TODO: предупреждать, если дата покупки не в текущем месяце (чек попадёт в отчёт за другой месяц)
    shop = storage.same_shop(db, uid, res.shop)  # «YONKS» и «Yonks» - один магазин
    rid = storage.save_receipt(db, uid, data, res.items, res.qr, shop=shop, purchased=res.date,
                               date_source=res.date_source, needs_review=res.needs_review)
    answer += f"\n\n💾 Сохранила как чек #{rid}."
    when = res.date or datetime.now()
    warnings = reports.limit_warnings(db, uid, {it["category"] for it in res.items}, when)
    if warnings:
        answer += "\n\n" + "\n".join(warnings)
    await send(message, answer, saved_keyboard(rid, res.needs_review))


@dp.message(F.photo)
async def handle_photo(message: Message):
    await answer_receipt(message, message.photo[-1])  # самое большое разрешение


@dp.message(F.document & F.document.mime_type.startswith("image/"))
async def handle_image_file(message: Message):
    await answer_receipt(message, message.document)


# --- карточка чека и список чеков ---
PAGE = 8


def receipt_card(uid: int, rid: int):
    r = storage.get_receipt(db, uid, rid)
    if not r:
        return None
    rows = storage.receipt_items(db, uid, rid)
    d = datetime.fromisoformat(r["purchased_at"])
    lines = [f"🧾 Чек #{rid}" + (" — ⚠️ требует проверки" if r["needs_review"] else ""),
             f"🏪 {r['shop'] or 'магазин не указан'}",
             f"📅 {d:%d.%m.%Y %H:%M} — {DATE_SOURCE.get(r['date_source'] or 'sent', '')}"]
    if (r["operation"] or "").startswith("возврат"):
        lines.append("↩️ возврат")
    if r["source"] == "demo":
        lines.append("🎭 демо-данные")
    lines += [""] + [f"{n}. {it['name']} — {it['price']:.2f} ₽ ({it['category']})" for n, it in enumerate(rows, 1)]
    lines += ["", f"💰 Итого: {r['total']:.2f} ₽", "", "Нажми на позицию, чтобы изменить её сумму, название или категорию."]
    kb = [[InlineKeyboardButton(text=f"{n}. {it['name'][:24]} — {it['price']:.0f} ₽", callback_data=f"item:{it['id']}")]
          for n, it in enumerate(rows[:40], 1)]
    kb.append([InlineKeyboardButton(text="📅 Дата", callback_data=f"date:{rid}"),
               InlineKeyboardButton(text="🏪 Магазин", callback_data=f"shop:{rid}"),
               InlineKeyboardButton(text="➕ Позиция", callback_data=f"add:{rid}")])
    last = [InlineKeyboardButton(text="🗑 Удалить чек", callback_data=f"del:{rid}"),
            InlineKeyboardButton(text="⬅️ Все чеки", callback_data="hist:0")]
    if r["needs_review"]:
        last.insert(0, InlineKeyboardButton(text="✅ Проверено", callback_data=f"ok:{rid}"))
    kb.append(last)
    return "\n".join(lines), InlineKeyboardMarkup(inline_keyboard=kb)


def history_page(uid: int, offset: int):
    rows, total = storage.receipts_page(db, uid, offset, PAGE)
    if not rows:
        return "Чеков пока нет. Пришли фото чека 🧾 или добавь трату без чека — /add", None
    text = (f"🧾 Твои чеки ({offset + 1}–{offset + len(rows)} из {total}), от новых к старым. "
            f"Нажми на чек, чтобы открыть и изменить. ⚠️ — требует проверки, ✍️ — без чека, 🎭 — демо")
    kb = []
    for r in rows:
        d = datetime.fromisoformat(r["purchased_at"])
        mark = "⚠️ " if r["needs_review"] else "✍️ " if r["source"] == "manual" else "🎭 " if r["source"] == "demo" else ""
        label = f"{mark}#{r['id']} · {d:%d.%m.%y} · {r['total']:.0f} ₽" + (f" · {r['shop'][:16]}" if r["shop"] else "")
        kb.append([InlineKeyboardButton(text=label, callback_data=f"open:{r['id']}")])
    nav = []
    if offset > 0:
        nav.append(InlineKeyboardButton(text="◀️ Новее", callback_data=f"hist:{max(0, offset - PAGE)}"))
    if offset + PAGE < total:
        nav.append(InlineKeyboardButton(text="Старее ▶️", callback_data=f"hist:{offset + PAGE}"))
    if nav:
        kb.append(nav)
    return text, InlineKeyboardMarkup(inline_keyboard=kb)


def item_card(row):
    text = f"Позиция: {row['name']}\nСумма: {row['price']:.2f} ₽\nКатегория: {row['category']}\n\nЧто изменить?"
    return text, InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🏷 Категорию", callback_data=f"icat:{row['id']}"),
         InlineKeyboardButton(text="💰 Сумму", callback_data=f"iprice:{row['id']}")],
        [InlineKeyboardButton(text="✏️ Название", callback_data=f"iname:{row['id']}"),
         InlineKeyboardButton(text="🗑 Удалить позицию", callback_data=f"idel:{row['id']}")],
        [InlineKeyboardButton(text="⬅️ К чеку", callback_data=f"open:{row['receipt_id']}")]])


def cancel_kb(back: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text="Отмена", callback_data=back)]])


@dp.callback_query(F.data.startswith("open:") | F.data.startswith("edit:"))
async def cb_open(call: CallbackQuery):
    card = receipt_card(call.from_user.id, int(call.data.split(":")[1]))
    if not card:
        await call.answer("Чек не найден — возможно, он удалён", show_alert=True)
        return
    pending.pop(call.from_user.id, None)
    if "Сохранила" in (call.message.text or "") or "уже сохранён" in (call.message.text or ""):
        await send(call.message, *card)  # ответ с распознанным чеком оставляем как есть
    else:
        await show(call, *card)
    await call.answer()


@dp.callback_query(F.data.startswith("ok:"))
async def cb_reviewed(call: CallbackQuery):
    rid = int(call.data.split(":")[1])
    storage.set_reviewed(db, call.from_user.id, rid)
    await call.answer("Отметила: чек проверен ✅")
    card = receipt_card(call.from_user.id, rid)
    if card and "Сохранила" not in (call.message.text or ""):
        await show(call, *card)
    elif card:
        await call.message.edit_reply_markup(reply_markup=saved_keyboard(rid, False))


@dp.callback_query(F.data.startswith("hist:"))
async def cb_hist(call: CallbackQuery):
    await show(call, *history_page(call.from_user.id, int(call.data.split(":")[1])))
    await call.answer()


def owned_item(call: CallbackQuery):
    return storage._owned_item(db, call.from_user.id, int(call.data.split(":")[1]))


@dp.callback_query(F.data.startswith("item:"))
async def cb_item(call: CallbackQuery):
    row = owned_item(call)
    if not row:
        await call.answer("Позиция не найдена", show_alert=True)
        return
    pending.pop(call.from_user.id, None)
    await show(call, *item_card(row))
    await call.answer()


@dp.callback_query(F.data.startswith("icat:"))
async def cb_icat(call: CallbackQuery):
    item_id = int(call.data.split(":")[1])
    kb = [[InlineKeyboardButton(text=c, callback_data=f"cat:{item_id}:{i}")
           for i, c in enumerate(CATEGORIES[k:k + 4], start=k)] for k in (0, 4)]
    kb.append([InlineKeyboardButton(text="⬅️ Назад", callback_data=f"item:{item_id}")])
    await show(call, "Выбери правильную категорию:", InlineKeyboardMarkup(inline_keyboard=kb))
    await call.answer()


@dp.callback_query(F.data.startswith("cat:"))
async def cb_cat(call: CallbackQuery):
    _, item_id, idx = call.data.split(":")
    category = CATEGORIES[int(idx)]
    row = storage.set_category(db, call.from_user.id, int(item_id), category)
    if not row:
        await call.answer("Позиция не найдена", show_alert=True)
        return
    log_correction("категория", row["name"], row["category"], category)
    await show(call, *receipt_card(call.from_user.id, row["receipt_id"]))
    await call.answer(f"Категория: {category}")


@dp.callback_query(F.data.startswith("iprice:") | F.data.startswith("iname:"))
async def cb_ask_item_text(call: CallbackQuery):
    row = owned_item(call)
    if not row:
        await call.answer("Позиция не найдена", show_alert=True)
        return
    kind = "price" if call.data.startswith("iprice:") else "name"
    set_pending(call.from_user.id, kind, row["id"])
    ask = (f"Пришли новую сумму для «{row['name'][:40]}» (сейчас {row['price']:.2f} ₽), например 1980 или 1 980,50"
           if kind == "price" else f"Пришли новое название вместо «{row['name'][:60]}»")
    await show(call, ask, cancel_kb(f"item:{row['id']}"))
    await call.answer()


@dp.callback_query(F.data.startswith("add:"))
async def cb_add_item(call: CallbackQuery):
    rid = int(call.data.split(":")[1])
    if not storage.get_receipt(db, call.from_user.id, rid):
        await call.answer("Чек не найден", show_alert=True)
        return
    set_pending(call.from_user.id, "add", rid)
    await show(call, f"Пришли позицию для чека #{rid}: название и сумму, например «Хлеб Бородинский 59,90»",
               cancel_kb(f"open:{rid}"))
    await call.answer()


@dp.callback_query(F.data.startswith("idel:"))
async def cb_idel(call: CallbackQuery):
    row = owned_item(call)
    if not row:
        await call.answer("Позиция не найдена", show_alert=True)
        return
    kb = InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="Да, удалить", callback_data=f"idelok:{row['id']}"),
        InlineKeyboardButton(text="Отмена", callback_data=f"item:{row['id']}")]])
    await show(call, f"Удалить позицию «{row['name'][:50]}» — {row['price']:.2f} ₽?", kb)
    await call.answer()


@dp.callback_query(F.data.startswith("idelok:"))
async def cb_idelok(call: CallbackQuery):
    row = storage.delete_item(db, call.from_user.id, int(call.data.split(":")[1]))
    if not row:
        await call.answer("Позиция не найдена", show_alert=True)
        return
    log_correction("удалена позиция", row["name"], row["price"], "")
    card = receipt_card(call.from_user.id, row["receipt_id"])
    await show(call, *(card or ("Это была последняя позиция — чек удалён 🗑", None)))
    await call.answer("Позиция удалена")


# дата и магазин чека
@dp.callback_query(F.data.startswith("date:"))
async def cb_date(call: CallbackQuery):
    rid = int(call.data.split(":")[1])
    set_pending(call.from_user.id, "date", rid)
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="Сегодня", callback_data=f"setd:{rid}:0"),
         InlineKeyboardButton(text="Вчера", callback_data=f"setd:{rid}:1"),
         InlineKeyboardButton(text="Позавчера", callback_data=f"setd:{rid}:2")],
        [InlineKeyboardButton(text="Отмена", callback_data=f"open:{rid}")]])
    await show(call, "Когда была покупка? Нажми кнопку или пришли дату, например 19.06.2026 или 19.06 22:54", kb)
    await call.answer()


@dp.callback_query(F.data.startswith("setd:"))
async def cb_setd(call: CallbackQuery):
    _, rid, days_ago = call.data.split(":")
    dt = (datetime.now() - timedelta(days=int(days_ago))).replace(second=0, microsecond=0)
    old = storage.get_receipt(db, call.from_user.id, int(rid))
    if old and storage.set_date(db, call.from_user.id, int(rid), dt):
        log_correction("дата", f"чек {rid}", old["purchased_at"], dt.isoformat(timespec="minutes"))
    pending.pop(call.from_user.id, None)
    card = receipt_card(call.from_user.id, int(rid))
    await show(call, *(card or ("Чек не найден", None)))
    await call.answer("Дата изменена")


@dp.callback_query(F.data.startswith("shop:"))
async def cb_shop(call: CallbackQuery):
    rid = int(call.data.split(":")[1])
    set_pending(call.from_user.id, "shop", rid)
    shops = storage.user_shops(db, call.from_user.id, 6)
    kb = [[InlineKeyboardButton(text=s[:30], callback_data=f"sets:{rid}:{i}")] for i, s in enumerate(shops)]
    kb.append([InlineKeyboardButton(text="Без магазина", callback_data=f"sets:{rid}:-1"),
               InlineKeyboardButton(text="Отмена", callback_data=f"open:{rid}")])
    await show(call, "Выбери магазин из уже встречавшихся или пришли название текстом:",
               InlineKeyboardMarkup(inline_keyboard=kb))
    await call.answer()


@dp.callback_query(F.data.startswith("sets:"))
async def cb_sets(call: CallbackQuery):
    _, rid, idx = call.data.split(":")
    shops = storage.user_shops(db, call.from_user.id, 6)
    shop = None if int(idx) < 0 or int(idx) >= len(shops) else shops[int(idx)]
    old = storage.get_receipt(db, call.from_user.id, int(rid))
    if old and storage.set_shop(db, call.from_user.id, int(rid), shop):
        log_correction("магазин", f"чек {rid}", old["shop"] or "", shop or "")
    pending.pop(call.from_user.id, None)
    card = receipt_card(call.from_user.id, int(rid))
    await show(call, *(card or ("Чек не найден", None)))
    await call.answer("Магазин изменён")


# удаление чека
@dp.callback_query(F.data.startswith("del:"))
async def cb_del(call: CallbackQuery):
    rid = int(call.data.split(":")[1])
    kb = InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text="Да, удалить", callback_data=f"delok:{rid}"),
                                                InlineKeyboardButton(text="Отмена", callback_data="cancel")]])
    await call.message.answer(f"Удалить чек #{rid} из учёта трат?", reply_markup=kb)
    await call.answer()


@dp.callback_query(F.data.startswith("delok:"))
async def cb_delok(call: CallbackQuery):
    rid = int(call.data.split(":")[1])
    ok = storage.delete_receipt(db, call.from_user.id, rid)
    await call.message.edit_text(f"🗑 Чек #{rid} удалён" if ok else "Чек не найден — возможно, уже удалён")
    await call.answer()


@dp.callback_query(F.data == "cancel")
async def cb_cancel(call: CallbackQuery):
    pending.pop(call.from_user.id, None)
    await call.message.edit_text("Отменено")
    await call.answer()


# --- отчёты ---
def month_nav(year: int, month: int) -> InlineKeyboardMarkup:
    prev = (year, month - 1) if month > 1 else (year - 1, 12)
    nxt = (year, month + 1) if month < 12 else (year + 1, 1)
    row = [InlineKeyboardButton(text=f"◀️ {reports.MONTHS[prev[1] - 1]}", callback_data=f"mon:{prev[0]}:{prev[1]}")]
    now = datetime.now()
    if nxt <= (now.year, now.month):
        row.append(InlineKeyboardButton(text=f"{reports.MONTHS[nxt[1] - 1]} ▶️", callback_data=f"mon:{nxt[0]}:{nxt[1]}"))
    return InlineKeyboardMarkup(inline_keyboard=[row])


async def send_month(message: Message, uid: int, year: int, month: int) -> None:
    text, charts = reports.month_report(db, uid, year, month)
    for i, chart in enumerate(charts):
        await message.answer_photo(BufferedInputFile(chart, f"month_{i}.png"))
    await send(message, text, month_nav(year, month))


@dp.message(Command("month"))
async def month_cmd(message: Message, command: CommandObject):
    pending.pop(message.from_user.id, None)
    now = datetime.now()
    year, month = now.year, now.month
    if command.args:
        try:
            m, y = command.args.strip().split(".")
            year, month = int(y), int(m)
        except ValueError:
            await message.answer("Формат: /month или /month 10.2026")
            return
    await send_month(message, message.from_user.id, year, month)


@dp.callback_query(F.data.startswith("mon:"))
async def cb_month(call: CallbackQuery):
    _, y, m = call.data.split(":")
    await call.answer()
    await send_month(call.message, call.from_user.id, int(y), int(m))


@dp.message(Command("week"))
async def week_cmd(message: Message):
    pending.pop(message.from_user.id, None)
    await message.answer(reports.answer_query(db, message.from_user.id,
                                              {"metric": "top_categories", "period": "last_7_days"}))


@dp.message(Command("history"))
async def history_cmd(message: Message):
    pending.pop(message.from_user.id, None)
    text, kb = history_page(message.from_user.id, 0)
    await message.answer(text, reply_markup=kb)


@dp.message(Command("receipt"))
async def receipt_cmd(message: Message, command: CommandObject):
    try:
        rid = int((command.args or "").strip().lstrip("#"))
    except ValueError:
        await message.answer("Формат: /receipt 12 — открыть чек №12. Список чеков — /history")
        return
    card = receipt_card(message.from_user.id, rid)
    await (send(message, *card) if card else message.answer("Чек не найден. Список чеков — /history"))


@dp.message(Command("prices"))
async def prices_cmd(message: Message):
    pending.pop(message.from_user.id, None)
    await send(message, reports.price_changes(db, message.from_user.id))


@dp.message(Command("export"))
async def export_cmd(message: Message):
    data = reports.export_csv(db, message.from_user.id)
    await message.answer_document(BufferedInputFile(data, f"траты_{datetime.now():%Y-%m-%d}.csv"),
                                  caption="Все траты — открывается в Excel")


# --- лимиты ---
def limits_view(uid: int):
    now = datetime.now()
    start, end = reports.month_range(now.year, now.month)
    cats = reports.by_category(storage.items_between(db, uid, start, end))
    lines = reports.limits_lines(db, uid, cats)
    text = ("🎯 Лимиты на " + reports.MONTHS[now.month - 1] + ":\n" + "\n".join(lines) if lines else
            "🎯 Лимитов пока нет. Выбери категорию и пришли сумму на месяц — я предупрежу, когда потрачено 80% "
            "и когда лимит превышен.")
    limits = storage.get_limits(db, uid)
    kb = [[InlineKeyboardButton(text=(f"{c}: {limits[c]:.0f} ₽" if c in limits else f"{c}: —"),
                                callback_data=f"lim:{i}") for i, c in enumerate(CATEGORIES[k:k + 2], start=k)]
          for k in range(0, len(CATEGORIES), 2)]
    return text + "\n\nНажми на категорию, чтобы задать или убрать лимит.", InlineKeyboardMarkup(inline_keyboard=kb)


@dp.message(Command("limits"))
async def limits_cmd(message: Message):
    pending.pop(message.from_user.id, None)
    text, kb = limits_view(message.from_user.id)
    await message.answer(text, reply_markup=kb)


@dp.callback_query(F.data.startswith("lim:"))
async def cb_limit(call: CallbackQuery):
    idx = int(call.data.split(":")[1])
    set_pending(call.from_user.id, "limit", idx)
    await show(call, f"Пришли лимит на месяц для «{CATEGORIES[idx]}», например 15000. Чтобы убрать лимит — пришли 0.",
               cancel_kb("cancel"))
    await call.answer()


# --- трата без чека ---
@dp.message(Command("add"))
async def add_cmd(message: Message, command: CommandObject):
    if command.args:
        await save_manual(message, command.args)
        return
    set_pending(message.from_user.id, "manual")
    await message.answer("Пришли трату одним сообщением: что и сколько, например «такси 450» или "
                         "«стрижка 1 500 вчера». Категорию подберу сама, её можно будет поправить.",
                         reply_markup=cancel_kb("cancel"))


async def save_manual(message: Message, text: str) -> bool:
    words = text.split()
    when = None
    # дата в конце: вчера/позавчера/сегодня или 12.11.2026 (12.11 не берём, это может быть цена)
    if len(words) > 2 and (words[-1].lower() in ("вчера", "сегодня", "позавчера")
                           or re.fullmatch(r"\d{1,2}\.\d{1,2}\.\d{2,4}", words[-1])):
        when, words = parse_user_date(words[-1]), words[:-1]
    m = re.search(r"(\d[\d  ]*(?:[.,]\d{1,2})?)\s*(?:₽|руб\.?|р\.?)?\s*$", " ".join(words))
    if not m:
        await message.answer("Не нашла сумму 🤔 Пришли так: «такси 450» или «стрижка 1 500 вчера». /cancel — отмена")
        return False
    price = parse_amount(m.group(1))
    name = " ".join(words)[:m.start()].strip(" —-:,")
    if not price or len(name) < 2:
        await message.answer("Пришли название и сумму, например «такси 450». /cancel — отмена")
        return False
    category = pipeline.classify([name])[0]
    rid = storage.save_manual(db, message.from_user.id, name, price, category, when)
    pending.pop(message.from_user.id, None)
    when_text = f"{when:%d.%m.%Y}" if when else "сегодня"
    text = f"✍️ Записала: {name} — {rub(price)} ({category}), {when_text}. Это чек #{rid}."
    warnings = reports.limit_warnings(db, message.from_user.id, {category}, when or datetime.now())
    if warnings:
        text += "\n\n" + "\n".join(warnings)
    await message.answer(text, reply_markup=saved_keyboard(rid, False))
    return True


# --- демо-данные и удаление своих данных ---
@dp.message(Command("demo"))
async def demo_cmd(message: Message):
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🎭 Добавить демо-траты", callback_data="demo:add")],
        [InlineKeyboardButton(text="🧹 Убрать демо-траты", callback_data="demo:clear")]])
    await message.answer("Демо-данные — правдоподобные траты за 3 прошлых месяца и текущий месяц, чтобы показать "
                         "отчёты, лимиты, динамику цен и прогноз. Твои настоящие чеки не трогаются, демо-траты "
                         "можно убрать одной кнопкой.", reply_markup=kb)


@dp.callback_query(F.data.startswith("demo:"))
async def cb_demo(call: CallbackQuery):
    uid = call.from_user.id
    if call.data == "demo:add":
        demo.clear(db, uid)
        n = demo.generate(db, uid)
        await call.message.edit_text(f"🎭 Добавила {n} демо-чеков. Посмотри /month, /history, /prices или спроси "
                                     f"«где я больше всего трачу?»")
    else:
        n = demo.clear(db, uid)
        await call.message.edit_text(f"🧹 Убрала {n} демо-чеков. Настоящие чеки остались.")
    await call.answer()


@dp.message(Command("forget"))
async def forget_cmd(message: Message):
    n = storage.receipts_page(db, message.from_user.id, 0, 1)[1]
    kb = InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text="Да, удалить всё", callback_data="forget:yes"),
                                                InlineKeyboardButton(text="Отмена", callback_data="cancel")]])
    await message.answer(f"Удалить все твои данные: {n} чек(ов) и лимиты? Это нельзя отменить.", reply_markup=kb)


@dp.callback_query(F.data == "forget:yes")
async def cb_forget(call: CallbackQuery):
    n = storage.delete_all(db, call.from_user.id)
    pending.pop(call.from_user.id, None)
    await call.message.edit_text(f"🗑 Удалено чеков: {n}. Лимиты тоже удалены.")
    await call.answer()


# --- общие команды ---
@dp.message(CommandStart())
async def start(message: Message):
    await message.answer("Привет! " + HELP, reply_markup=MENU)


@dp.message(Command("help"))
async def help_cmd(message: Message):
    await message.answer(HELP, reply_markup=MENU)


@dp.message(Command("cancel"))
async def cancel_cmd(message: Message):
    await message.answer("Отменено" if pending.pop(message.from_user.id, None) else "Нечего отменять 🙂", reply_markup=MENU)


# --- текстовые сообщения: кнопки меню, ввод, вопросы ---
async def handle_pending(message: Message) -> bool:
    """Пользователь вводит то, о чём мы попросили (сумму, название, дату, …). True - сообщение обработано."""
    uid, text = message.from_user.id, message.text.strip()
    st = get_pending(uid)
    if not st:
        return False
    kind, ref, _ = st
    if kind == "manual":
        await save_manual(message, text)
        return True
    if kind == "limit":
        amount = 0.0 if text.strip() in ("0", "нет", "убрать") else parse_amount(text)
        if amount is None:
            await message.answer("Пришли сумму, например 15000, или 0 — чтобы убрать лимит. /cancel — отмена")
            return True
        storage.set_limit(db, uid, CATEGORIES[ref], amount or None)
        pending.pop(uid, None)
        view_text, view_kb = limits_view(uid)
        await message.answer(view_text, reply_markup=view_kb)
        return True
    rid = None
    if kind == "price":
        price = parse_amount(text)
        if price is None:
            await message.answer("Не поняла сумму 🤔 Пришли число, например 1980 или 1 980,50. /cancel — отмена")
            return True
        row = storage.set_price(db, uid, ref, price)
        if row:
            log_correction("сумма", row["name"], row["price"], price)
            rid = row["receipt_id"]
    elif kind == "name":
        if len(text) < 2:
            await message.answer("Слишком короткое название. /cancel — отмена")
            return True
        row = storage.set_name(db, uid, ref, text[:120])
        if row:
            log_correction("название", row["name"], row["name"], text[:120])
            rid = row["receipt_id"]
    elif kind == "add":
        m = re.search(r"(\d[\d  ]*(?:[.,]\d{1,2})?)\s*(?:₽|руб\.?)?\s*$", text)
        price = parse_amount(m.group(1)) if m else None
        name = text[:m.start()].strip(" —-:,") if m else ""
        if price is None or len(name) < 2:
            await message.answer("Пришли название и сумму в конце, например «Хлеб Бородинский 59,90». /cancel — отмена")
            return True
        if storage.add_item(db, uid, ref, name, price, pipeline.classify([name])[0]):
            rid = ref
    elif kind == "date":
        dt = parse_user_date(text)
        if not dt:
            await message.answer("Не поняла дату 🤔 Пришли так: 19.06.2026, 19.06 или 19.06.2026 22:54. /cancel — отмена")
            return True
        old = storage.get_receipt(db, uid, ref)
        if old and storage.set_date(db, uid, ref, dt):
            log_correction("дата", f"чек {ref}", old["purchased_at"], dt.isoformat(timespec="minutes"))
            rid = ref
    elif kind == "shop":
        shop = storage.same_shop(db, uid, text[:60])
        old = storage.get_receipt(db, uid, ref)
        if old and storage.set_shop(db, uid, ref, shop):
            log_correction("магазин", f"чек {ref}", old["shop"] or "", shop)
            rid = ref
    pending.pop(uid, None)
    card = receipt_card(uid, rid) if rid else None
    if card:
        await send(message, "✅ Сохранено\n\n" + card[0], card[1])
    else:
        await message.answer("Не нашла этот чек — возможно, он удалён")
    return True


MENU_ACTIONS = {BTN_MONTH: "month", BTN_HISTORY: "history", BTN_ADD: "add", BTN_LIMITS: "limits",
                BTN_PRICES: "prices", BTN_HELP: "help"}
PERIOD_RU = {"today": "сегодня", "yesterday": "вчера", "this_week": "эта неделя", "last_week": "прошлая неделя",
             "last_7_days": "7 дней", "this_month": "этот месяц", "last_month": "прошлый месяц",
             "last_30_days": "30 дней", "this_year": "этот год", "all": "всё время"}
METRIC_RU = {"sum": "сумма трат", "count": "число покупок", "avg_receipt": "средний чек", "top_items": "топ товаров",
             "top_categories": "по категориям", "top_shops": "топ магазинов", "max_item": "самая дорогая покупка"}


@dp.message(F.text)
async def text_message(message: Message):
    action = MENU_ACTIONS.get(message.text.strip())
    if action:  # кнопка меню внизу экрана
        pending.pop(message.from_user.id, None)
        if action == "month":
            now = datetime.now()
            await send_month(message, message.from_user.id, now.year, now.month)
        elif action == "history":
            await history_cmd(message)
        elif action == "add":
            set_pending(message.from_user.id, "manual")
            await message.answer("Пришли трату одним сообщением: что и сколько, например «такси 450» или "
                                 "«стрижка 1 500 вчера».", reply_markup=cancel_kb("cancel"))
        elif action == "limits":
            await limits_cmd(message)
        elif action == "prices":
            await prices_cmd(message)
        else:
            await help_cmd(message)
        return
    if message.text.startswith("/"):
        pending.pop(message.from_user.id, None)
        await message.answer("Не знаю такой команды 🙂\n\n" + HELP, reply_markup=MENU)
        return
    if await handle_pending(message):
        return
    if not query_model.available():
        await message.answer(HELP, reply_markup=MENU)
        return
    async with recognize_lock:
        q = await asyncio.to_thread(query_model.question_to_query, message.text)
    if not q or q.get("metric") in (None, "none"):
        await message.answer("Я умею разбирать чеки и отвечать на вопросы о тратах 🙂\n\n" + HELP, reply_markup=MENU)
        return
    period = q.get("period") or ""
    understood = ", ".join(x for x in [METRIC_RU.get(q["metric"], q["metric"]),
                                       PERIOD_RU.get(period, period.replace("month:", "месяц ")),
                                       q.get("category"), q.get("text") and f"«{q['text']}»"] if x)
    await send(message, reports.answer_query(db, message.from_user.id, q) + f"\n\n🤖 Поняла вопрос так: {understood}")


@dp.message()
async def other(message: Message):
    await message.answer("Пришли, пожалуйста, фото чека 📷 или задай вопрос о тратах. /help — что я умею", reply_markup=MENU)


async def main():
    if not BOT_TOKEN:
        raise SystemExit("Нет BOT_TOKEN: скопируй .env.example в .env и впиши токен от @BotFather.")
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if query_model.available():
        query_model.load()
        print("Подключён адаптер вопросов:", query_model.QUERY_ADAPTER)
    bot = Bot(BOT_TOKEN)
    await bot.set_my_commands(COMMANDS)  # меню команд в Telegram
    print("Бот запущен")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
