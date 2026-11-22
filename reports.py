import calendar
import csv
import io
import re
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

import storage  # noqa: E402

CATEGORIES = ["еда", "напитки", "бытовое", "здоровье", "одежда", "транспорт", "развлечения", "другое"]
COLORS = {"еда": "#4C9F70", "напитки": "#3A7CA5", "бытовое": "#E0A458", "здоровье": "#D1495B",
          "одежда": "#8E6C8A", "транспорт": "#5C6B73", "развлечения": "#F2C14E", "другое": "#9DA3A4"}
MONTHS = ["январь", "февраль", "март", "апрель", "май", "июнь", "июль", "август", "сентябрь", "октябрь",
          "ноябрь", "декабрь"]
MONTHS_GEN = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября",
              "ноября", "декабря"]
PERIODS = ["today", "yesterday", "this_week", "last_week", "last_7_days", "this_month", "last_month",
           "last_30_days", "this_year", "all"]  # + "month:MM" и "month:MM.YYYY"


def rub(x: float) -> str:
    return f"{x:,.2f}".replace(",", " ").replace(".00", "") + " ₽"


# --- периоды ---
def month_range(year: int, month: int) -> tuple[datetime, datetime]:
    start = datetime(year, month, 1)
    end = datetime(year + (month == 12), month % 12 + 1, 1)
    return start, end


def resolve_period(period: str, today: date | None = None) -> tuple[datetime, datetime, str]:
    """Ключ периода -> (начало, конец, подпись по-русски)."""
    today = today or date.today()
    t0 = datetime(today.year, today.month, today.day)
    if period == "today":
        return t0, t0 + timedelta(days=1), "сегодня"
    if period == "yesterday":
        return t0 - timedelta(days=1), t0, "вчера"
    if period == "this_week":
        s = t0 - timedelta(days=today.weekday())
        return s, t0 + timedelta(days=1), "на этой неделе"
    if period == "last_week":
        s = t0 - timedelta(days=today.weekday() + 7)
        return s, s + timedelta(days=7), "на прошлой неделе"
    if period == "last_7_days":
        return t0 - timedelta(days=6), t0 + timedelta(days=1), "за последние 7 дней"
    if period == "last_30_days":
        return t0 - timedelta(days=29), t0 + timedelta(days=1), "за последние 30 дней"
    if period == "this_month":
        s, e = month_range(today.year, today.month)
        return s, e, "в этом месяце"
    if period == "last_month":
        y, m = (today.year, today.month - 1) if today.month > 1 else (today.year - 1, 12)
        s, e = month_range(y, m)
        return s, e, f"в прошлом месяце ({MONTHS[m - 1]})"
    if period == "this_year":
        return datetime(today.year, 1, 1), datetime(today.year + 1, 1, 1), f"в {today.year} году"
    if period.startswith("month:"):
        spec = period.split(":", 1)[1]
        if "." in spec:
            m, y = (int(x) for x in spec.split("."))
        else:
            m = int(spec)
            y = today.year if m <= today.month else today.year - 1  # «в октябре» - последний прошедший октябрь
        s, e = month_range(y, m)
        return s, e, f"за {MONTHS[m - 1]} {y}"
    return datetime(2000, 1, 1), datetime(2100, 1, 1), "за всё время"


# --- сводки ---
def by_category(rows) -> dict[str, float]:
    sums = defaultdict(float)
    for r in rows:
        sums[r["category"]] += r["amount"]
    return dict(sorted(sums.items(), key=lambda kv: -kv[1]))


def month_report(con, user_id: int, year: int, month: int) -> tuple[str, list[bytes]]:
    """Текст отчёта за месяц + картинки (круговая диаграмма по категориям и траты по дням)."""
    start, end = month_range(year, month)
    rows = storage.items_between(con, user_id, start, end)
    title = f"📊 Траты за {MONTHS[month - 1]} {year}"
    if not rows:
        return f"{title}\n\nЗа этот месяц трат пока нет. Пришли фото чека — и я его учту 🧾", []
    cats = by_category(rows)
    total = sum(cats.values())
    receipts = {r["receipt_id"] for r in rows}
    py, pm = (year, month - 1) if month > 1 else (year - 1, 12)
    prev_start, prev_end = month_range(py, pm)
    now = datetime.now()
    partial = (year, month) == (now.year, now.month)
    if partial:  # месяц не закончился - сравниваем с тем же числом дней прошлого
        last_day = calendar.monthrange(py, pm)[1]
        prev_end = datetime(py, pm, min(now.day, last_day)) + timedelta(days=1)
        compare_text = f"1–{min(now.day, last_day)} {MONTHS_GEN[pm - 1]}"
    else:
        compare_text = MONTHS[pm - 1]
    prev = by_category(storage.items_between(con, user_id, prev_start, prev_end))
    lines = [title, "", f"💰 Всего: {rub(total)} — {len(receipts)} чек(ов), средний чек {rub(total / len(receipts))}", ""]
    for cat, amount in cats.items():
        line = f"• {cat}: {rub(amount)} ({amount / total:.0%})"
        if prev.get(cat):
            change = (amount - prev[cat]) / prev[cat]
            line += f"  {'▲' if change > 0 else '▼'} {abs(change):.0%}" if abs(change) >= 0.01 else ""
        lines.append(line)
    if prev:
        prev_total = sum(prev.values())
        diff = total - prev_total
        lines += ["", f"Стрелки — сравнение с периодом {compare_text}: всего "
                      f"{'больше' if diff > 0 else 'меньше'} на {rub(abs(diff))} ({abs(diff) / prev_total:.0%})"]
    top = Counter()
    for r in rows:
        top[r["name"]] += r["amount"]
    lines += ["", "🏆 Самые крупные покупки:"] + [f"  {n[:40]} — {rub(a)}" for n, a in top.most_common(3)]
    limits = limits_lines(con, user_id, cats)
    if limits:
        lines += ["", "🎯 Лимиты на месяц:"] + limits
    if partial:
        import forecast  # прогноз и необычные траты
        lines += forecast.month_insights(con, user_id, now)
    review = sum(1 for r in storage.receipts_between(con, user_id, start, end) if r["needs_review"])
    if review:
        lines += ["", f"⚠️ {review} чек(ов) требуют проверки — сумма не сошлась с итогом. Открой их в /history"]
    charts = [pie_chart(cats, f"Траты за {MONTHS[month - 1]} {year}"), daily_chart(rows, year, month)]
    return "\n".join(lines), charts


def daily_chart(rows, year: int, month: int) -> bytes:
    """Столбцы - траты по дням месяца (цвет - категория), линия - средняя трата в день."""
    days = calendar.monthrange(year, month)[1]
    per = defaultdict(lambda: [0.0] * days)
    for r in rows:
        d = datetime.fromisoformat(r["purchased_at"]).day
        per[r["category"]][d - 1] += max(r["amount"], 0)
    fig, ax = plt.subplots(figsize=(9, 3.6), dpi=120)
    bottom = [0.0] * days
    x = list(range(1, days + 1))
    for cat in [c for c in CATEGORIES if c in per] + [c for c in per if c not in CATEGORIES]:
        ax.bar(x, per[cat], bottom=bottom, color=COLORS.get(cat, "#999"), label=cat, width=0.8)
        bottom = [b + v for b, v in zip(bottom, per[cat])]
    now = datetime.now()
    passed = now.day if (year, month) == (now.year, now.month) else days  # прошедшие дни месяца
    if sum(bottom) > 0:
        avg = sum(bottom) / passed
        ax.axhline(avg, color="#333", linestyle="--", linewidth=1)
        ax.text(days + 0.6, avg, f"в среднем\n{rub(avg)}/день", va="center", fontsize=8)
    ordered = sorted(bottom, reverse=True)
    # TODO: если траты только за один день, cap = 1 и подпись среднего уезжает вверх -
    #  картинка выходит огромной и Telegram её не принимает. ограничивать только при 2+ днях
    if ordered[0] > 0 and ordered[0] > 4 * max(ordered[1], 1):  # один огромный день сплющивает остальные столбцы
        cap = max(ordered[1] * 1.6, 1)
        ax.set_ylim(0, cap)
        d = bottom.index(ordered[0])
        ax.annotate(f"↑ {rub(ordered[0])}", (d + 1, cap * 0.97), ha="center", va="top", fontsize=8, fontweight="bold")
    ax.set_xlim(0.3, days + 0.7)
    ax.set_xticks([1, 5, 10, 15, 20, 25, days])
    ax.set_xlabel(f"день месяца ({MONTHS[month - 1]})")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", " ")))
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=8, ncol=8, frameon=False)
    ax.set_title("Траты по дням", fontsize=12)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    return buf.getvalue()


def limits_lines(con, user_id: int, cats: dict[str, float]) -> list[str]:
    out = []
    for cat, limit in storage.get_limits(con, user_id).items():
        spent = cats.get(cat, 0.0)
        share = spent / limit
        mark = "🔴" if share >= 1 else "🟡" if share >= 0.8 else "🟢"
        tail = f"превышен на {rub(spent - limit)}" if share >= 1 else f"осталось {rub(limit - spent)}"
        out.append(f"{mark} {cat}: {rub(spent)} из {rub(limit)} ({share:.0%}) — {tail}")
    return out


def limit_warnings(con, user_id: int, categories: set[str], when: datetime) -> list[str]:
    """После нового чека: предупреждения по лимитам затронутых категорий (80% и превышение)."""
    limits = storage.get_limits(con, user_id)
    if not limits:
        return []
    start, end = month_range(when.year, when.month)
    cats = by_category(storage.items_between(con, user_id, start, end))
    out = []
    for cat in categories & set(limits):
        spent, limit = cats.get(cat, 0.0), limits[cat]
        if spent >= limit:
            out.append(f"🔴 Лимит «{cat}» на {MONTHS[when.month - 1]} превышен: {rub(spent)} из {rub(limit)}")
        elif spent >= 0.8 * limit:
            out.append(f"🟡 «{cat}»: потрачено {spent / limit:.0%} лимита ({rub(spent)} из {rub(limit)})")
    return out


WEIGHT = ("вес", "кг", "весов")


def price_changes(con, user_id: int, n: int = 10) -> str:
    """Один и тот же товар в разных чеках: как менялась цена.
    Весовые товары пропускаем - у них сумма по позиции зависит от веса, а не от цены."""
    from preprocess import normalize
    rows = con.execute(
        "SELECT i.name, i.price, i.category, r.purchased_at, r.shop FROM items i JOIN receipts r ON r.id=i.receipt_id "
        "WHERE r.user_id=? AND r.operation NOT LIKE 'возврат%' ORDER BY r.purchased_at", (user_id,)).fetchall()
    groups = defaultdict(list)
    for r in rows:
        # весовые товары и такси/бензин: сумма зависит от количества, а не от цены
        if any(w in r["name"].lower() for w in WEIGHT) or r["category"] == "транспорт":
            continue
        # один товар в одном магазине (капучино в разных местах стоит по-разному)
        key = (re.sub(r"^\d+\s*", "", normalize(r["name"])), (r["shop"] or "").lower())
        groups[key].append(r)
    changes = []
    for rs in groups.values():
        if len({x["purchased_at"][:10] for x in rs}) < 2:
            continue
        first, last = rs[0], rs[-1]
        if first["price"] <= 0 or abs(last["price"] - first["price"]) < 0.01:
            continue
        ratio = last["price"] / first["price"]
        # разница в 2-5 раз - скорее другое количество штук, а не новая цена
        # (модель пишет сумму позиции, а не цену за штуку)
        if any(abs(ratio - k) < 0.08 * k or abs(1 / ratio - k) < 0.08 * k for k in (2, 3, 4, 5)):
            continue
        changes.append((ratio - 1, first, last))
    if not changes:
        return ("📈 Пока не с чем сравнить: нужно, чтобы один и тот же товар встретился хотя бы в двух чеках "
                "в разные дни (весовые товары не учитываю — у них сумма зависит от веса).")
    changes.sort(key=lambda c: -abs(c[0]))
    lines = ["📈 Как менялись цены на одни и те же товары:"]
    for ch, first, last in changes[:n]:
        d1, d2 = datetime.fromisoformat(first["purchased_at"]), datetime.fromisoformat(last["purchased_at"])
        lines.append(f"{'🔺' if ch > 0 else '🔻'} {last['name'][:38]}: {rub(first['price'])} ({d1:%d.%m}) → "
                     f"{rub(last['price'])} ({d2:%d.%m}), {ch:+.0%}")
    return "\n".join(lines)


def pie_chart(cats: dict[str, float], title: str) -> bytes:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    total = sum(cats.values())
    wedges, _ = ax.pie(list(cats.values()), colors=[COLORS.get(c, "#999") for c in cats], startangle=90,
                       counterclock=False, wedgeprops={"width": 0.42, "edgecolor": "white"})
    ax.text(0, 0, rub(total), ha="center", va="center", fontsize=14, fontweight="bold")
    # подписи в легенде: на круге мелкие доли налезают друг на друга
    ax.legend(wedges, [f"{c} — {rub(v)} ({v / total:.0%})" for c, v in cats.items()], loc="center left",
              bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=11)
    ax.set_title(title, fontsize=14)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    return buf.getvalue()


def history_text(con, user_id: int, n: int = 10) -> str:
    rows = storage.last_receipts(con, user_id, n)
    if not rows:
        return "Чеков пока нет. Пришли фото чека 🧾"
    lines = ["🧾 Последние чеки:"]
    for r in rows:
        d = datetime.fromisoformat(r["purchased_at"])
        ret = " (возврат)" if (r["operation"] or "").startswith("возврат") else ""
        lines.append(f"#{r['id']}  {d:%d.%m.%Y}  {rub(r['total'])}{ret} — {r['n_items']} поз."
                     f"{'  ' + r['shop'] if r['shop'] else ''}")
    return "\n".join(lines)


def export_csv(con, user_id: int) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf, delimiter=";")
    w.writerow(["дата", "чек", "магазин", "товар", "категория", "сумма"])
    for r in storage.all_items(con, user_id):
        w.writerow([r["purchased_at"][:16].replace("T", " "), r["receipt_id"], r["shop"] or "", r["name"],
                    r["category"], f"{r['amount']:.2f}".replace(".", ",")])
    return buf.getvalue().encode("utf-8-sig")  # BOM, чтобы Excel открыл кириллицу


# --- ответы на вопросы ---
METRICS = ["sum", "count", "avg_receipt", "top_items", "top_categories", "top_shops", "max_item"]


def answer_query(con, user_id: int, q: dict, today: date | None = None) -> str:
    start, end, period_text = resolve_period(q.get("period") or "this_month", today)
    rows = storage.items_between(con, user_id, start, end)
    cat, text = q.get("category"), (q.get("text") or "").strip().lower()
    if cat in CATEGORIES:
        rows = [r for r in rows if r["category"] == cat]
    if text:
        rows = [r for r in rows if text in r["name"].lower() or text in (r["shop"] or "").lower()]
    what = " ".join(x for x in [f"на категорию «{cat}»" if cat in CATEGORIES else "", f"по запросу «{text}»" if text else ""] if x)
    metric = q.get("metric") if q.get("metric") in METRICS else "sum"
    if not rows:
        return f"Не нашла трат {what + ' ' if what else ''}{period_text} 🤷"
    total = sum(r["amount"] for r in rows)
    receipts = {r["receipt_id"] for r in rows}
    if metric == "sum":
        return f"💰 {period_text.capitalize()} {('потрачено ' + what) if what else 'потрачено'}: {rub(total)} " \
               f"({len(rows)} поз. в {len(receipts)} чек(ах))"
    if metric == "count":
        return f"🧾 {period_text.capitalize()}: {len(receipts)} чек(ов), {len(rows)} поз. {what}".rstrip()
    if metric == "avg_receipt":
        return f"🧾 Средний чек {period_text}{' ' + what if what else ''}: {rub(total / len(receipts))} " \
               f"(по {len(receipts)} чек(ам))"
    if metric == "max_item":
        r = max(rows, key=lambda r: r["amount"])
        d = datetime.fromisoformat(r["purchased_at"])
        return f"💸 Самая дорогая покупка {period_text}{' ' + what if what else ''}: {r['name']} — {rub(r['amount'])} ({d:%d.%m.%Y})"
    groups = Counter()
    key = {"top_items": "name", "top_categories": "category", "top_shops": "shop"}[metric]
    for r in rows:
        groups[r[key] or "магазин не указан"] += r["amount"]
    title = {"top_items": "Больше всего потрачено на товары", "top_categories": "Траты по категориям",
             "top_shops": "Больше всего потрачено в магазинах"}[metric]
    return f"📊 {title} {period_text}:\n" + "\n".join(f"  {i}. {n[:40]} — {rub(a)}"
                                                       for i, (n, a) in enumerate(groups.most_common(5), 1))
