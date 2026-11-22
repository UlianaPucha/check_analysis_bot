import calendar
import math
from collections import defaultdict
from datetime import datetime

import numpy as np

import storage

MIN_MONTHS = 2          # мин. число полных прошлых месяцев для прогноза
MIN_RECEIPTS_IF = 20    # мин. число чеков для IsolationForest


def rub(x: float) -> str:
    return f"{x:,.0f}".replace(",", " ") + " ₽"


def month_start(dt: datetime) -> datetime:
    return datetime(dt.year, dt.month, 1)


def add_months(dt: datetime, k: int) -> datetime:
    m = dt.month - 1 + k
    return datetime(dt.year + m // 12, m % 12 + 1, 1)


def past_months(con, user_id: int, now: datetime, n: int = 6) -> list[list]:
    """Строки трат за n полных прошлых месяцев (только месяцы, где траты были)."""
    out = []
    for k in range(1, n + 1):
        s, e = add_months(month_start(now), -k), add_months(month_start(now), -k + 1)
        rows = storage.items_between(con, user_id, s, e)
        if rows:
            out.append((s, rows))
    return out


def spent_share_by_day(rows, start: datetime) -> np.ndarray:
    """Доля месячной суммы, потраченная к концу каждого дня (по 31 дню, дальше - 1.0)."""
    days = calendar.monthrange(start.year, start.month)[1]
    per_day = np.zeros(31)
    for r in rows:
        per_day[datetime.fromisoformat(r["purchased_at"]).day - 1] += max(r["amount"], 0)
    total = per_day.sum()
    cum = np.cumsum(per_day) / total if total > 0 else np.linspace(0, 1, 31)
    cum[days - 1:] = 1.0
    return cum


def forecast_month(con, user_id: int, now: datetime) -> dict | None:
    start = month_start(now)
    days = calendar.monthrange(now.year, now.month)[1]
    rows = storage.items_between(con, user_id, start, add_months(start, 1))
    spent = sum(r["amount"] for r in rows)
    if spent <= 0:
        return None
    # разовые крупные чеки (в 5+ раз больше обычного) не масштабируем на месяц
    totals = [r["total"] for r in con.execute("SELECT total FROM receipts WHERE user_id=? AND total>0", (user_id,))]
    big_limit = 5 * float(np.median(totals)) if len(totals) >= 10 else float("inf")
    receipt_sum = defaultdict(float)
    for r in rows:
        receipt_sum[r["receipt_id"]] += r["amount"]
    one_off = sum(v for v in receipt_sum.values() if v > big_limit)
    regular = spent - one_off
    history = past_months(con, user_id, now)
    day = now.day
    note = f"; разовые крупные покупки ({rub(one_off)}) учтены один раз" if one_off else ""
    if len(history) >= MIN_MONTHS:
        shares = np.array([spent_share_by_day(r, s)[day - 1] for s, r in history])
        shares = np.clip(shares, 0.05, 1.0)  # защита от деления на ~0 в начале месяца
        estimate = regular / shares.mean() + one_off
        lo, hi = regular / shares.max() + one_off, regular / shares.min() + one_off
        return {"spent": spent, "estimate": max(estimate, spent), "low": max(lo, spent), "high": max(hi, spent),
                "method": f"по твоим тратам за {len(history)} прошлых мес.{note}", "history": history, "rows": rows,
                "factor": 1 / shares.mean()}
    if day >= 5:  # в первые дни темп считать рано
        estimate = regular / day * days + one_off
        return {"spent": spent, "estimate": estimate, "low": None, "high": None,
                "method": f"по текущему темпу трат (истории пока мало){note}", "history": history, "rows": rows}
    return {"spent": spent, "estimate": None, "history": history, "rows": rows}


def category_anomalies(fc: dict, now: datetime) -> list[str]:
    """Категории, где в этом месяце (с учётом прогноза) тратится заметно больше обычного."""
    if len(fc["history"]) < MIN_MONTHS or not fc.get("estimate"):
        return []
    factor = fc.get("factor") or fc["estimate"] / fc["spent"]  # во сколько раз обычные траты вырастут к концу месяца
    now_cat = defaultdict(float)
    for r in fc["rows"]:
        now_cat[r["category"]] += r["amount"]
    past = defaultdict(list)
    for _, rows in fc["history"]:
        sums = defaultdict(float)
        for r in rows:
            sums[r["category"]] += r["amount"]
        for cat in set(now_cat) | set(sums):
            past[cat].append(sums.get(cat, 0.0))
    out = []
    for cat, spent in now_cat.items():
        projected = spent * factor
        usual = float(np.median(past[cat])) if past[cat] else 0.0
        if usual > 0 and projected > 1.5 * usual and projected - usual > 1000:
            out.append(f"📈 «{cat}»: к концу месяца выйдет около {rub(projected)} — "
                       f"в {projected / usual:.1f} раза больше обычного ({rub(usual)} в месяц)")
        elif usual == 0 and spent > 2000 and len(past[cat]) >= MIN_MONTHS:
            out.append(f"🆕 «{cat}»: {rub(spent)} — раньше трат в этой категории не было")
    return out


def unusual_receipts(con, user_id: int, now: datetime, n: int = 3) -> list[str]:
    """Необычные чеки этого месяца по модели IsolationForest, обученной на всех чеках пользователя."""
    rows = con.execute("SELECT r.*, (SELECT COUNT(*) FROM items i WHERE i.receipt_id=r.id) AS n_items "
                       "FROM receipts r WHERE user_id=? AND total>0", (user_id,)).fetchall()
    if len(rows) < MIN_RECEIPTS_IF:
        return []
    from sklearn.ensemble import IsolationForest
    X, dts = [], []
    for r in rows:
        dt = datetime.fromisoformat(r["purchased_at"])
        X.append([math.log1p(r["total"]), r["n_items"], dt.weekday(), dt.hour])
        dts.append(dt)
    X = np.array(X, dtype=float)
    model = IsolationForest(n_estimators=200, contamination=0.05, random_state=0).fit(X)
    scores = model.decision_function(X)  # меньше = необычнее
    median_total = float(np.median([r["total"] for r in rows]))
    found = []
    for r, dt, s, pred in zip(rows, dts, scores, model.predict(X)):
        # нужны только необычно крупные чеки
        if pred == -1 and (dt.year, dt.month) == (now.year, now.month) and r["total"] > 2 * median_total:
            found.append((s, r, dt))
    found.sort(key=lambda x: x[0])
    return [f"🔎 Необычно крупный чек #{r['id']} от {dt:%d.%m}: {rub(r['total'])}"
            f"{' (' + r['shop'] + ')' if r['shop'] else ''} — обычно чек около {rub(median_total)}"
            for s, r, dt in found[:n]]


def month_insights(con, user_id: int, now: datetime) -> list[str]:
    fc = forecast_month(con, user_id, now)
    if not fc:
        return []
    lines = []
    if fc.get("estimate"):
        rng = f" (от {rub(fc['low'])} до {rub(fc['high'])})" if fc.get("low") and fc["high"] - fc["low"] > 100 else ""
        lines += ["", f"🔮 Прогноз на весь месяц: около {rub(fc['estimate'])}{rng} — {fc['method']}"]
    extra = category_anomalies(fc, now) + unusual_receipts(con, user_id, now)
    if extra:
        lines += ["", "Обрати внимание:"] + extra
    return lines
