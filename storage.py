import hashlib
import os
import sqlite3
from datetime import datetime, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.getenv("EXPENSES_DB", os.path.join(HERE, "data", "expenses.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS receipts (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      INTEGER NOT NULL,
    created_at   TEXT NOT NULL,          -- когда прислали в бота
    purchased_at TEXT NOT NULL,          -- когда куплено (из QR) или = created_at
    shop         TEXT,
    total        REAL NOT NULL,          -- сумма позиций (итог по QR, если он есть и совпал)
    operation    TEXT DEFAULT 'приход',  -- приход / возврат прихода / ...
    qr_fn TEXT, qr_fd TEXT, qr_fp TEXT,
    image_hash   TEXT,
    items_key    TEXT                    -- «отпечаток» набора позиций для поиска дублей без QR
);
CREATE TABLE IF NOT EXISTS items (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    receipt_id INTEGER NOT NULL REFERENCES receipts(id) ON DELETE CASCADE,
    name       TEXT NOT NULL,
    price      REAL NOT NULL,
    category   TEXT NOT NULL,
    category_source TEXT DEFAULT 'model'  -- model / user
);
CREATE INDEX IF NOT EXISTS idx_receipts_user ON receipts(user_id, purchased_at);
CREATE INDEX IF NOT EXISTS idx_items_receipt ON items(receipt_id);
"""


# миграции: новые колонки добавляются к старой базе
MIGRATIONS = {
    "receipts": [("date_source", "TEXT DEFAULT 'sent'"),   # qr / receipt (прочитала модель) / sent / user
                 ("needs_review", "INTEGER DEFAULT 0"),      # 1 - сумма не сошлась с итогом или модель не уверена
                 ("source", "TEXT DEFAULT 'photo'"),         # photo / manual (трата без чека) / demo
                 ("note", "TEXT")],
}
EXTRA_SCHEMA = """
CREATE TABLE IF NOT EXISTS limits (
    user_id  INTEGER NOT NULL,
    category TEXT NOT NULL,
    amount   REAL NOT NULL,
    PRIMARY KEY (user_id, category)
);
"""


def connect(path: str = DB_PATH) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    con = sqlite3.connect(path, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(SCHEMA + EXTRA_SCHEMA)
    for table, cols in MIGRATIONS.items():
        have = {r["name"] for r in con.execute(f"PRAGMA table_info({table})")}
        for name, decl in cols:
            if name not in have:
                con.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
    con.commit()
    return con


def items_key(items: list[dict]) -> str:
    parts = sorted(f"{it['name'].strip().lower()}|{it['price']:.2f}" for it in items)
    return hashlib.sha1("\n".join(parts).encode()).hexdigest()


def find_duplicate(con, user_id: int, image_bytes: bytes, items: list[dict], qr: dict | None) -> int | None:
    """id уже сохранённого такого же чека или None."""
    if qr and qr.get("fn") and qr.get("fd"):
        row = con.execute("SELECT id FROM receipts WHERE user_id=? AND qr_fn=? AND qr_fd=?",
                          (user_id, qr["fn"], qr["fd"])).fetchone()
        if row:
            return row["id"]
    row = con.execute("SELECT id FROM receipts WHERE user_id=? AND image_hash=?",
                      (user_id, hashlib.sha256(image_bytes).hexdigest())).fetchone()
    if row:
        return row["id"]
    since = (datetime.now() - timedelta(days=14)).isoformat(timespec="seconds")
    row = con.execute("SELECT id FROM receipts WHERE user_id=? AND items_key=? AND created_at>=?",
                      (user_id, items_key(items), since)).fetchone()
    return row["id"] if row else None


def save_receipt(con, user_id: int, image_bytes: bytes, items: list[dict], qr: dict | None,
                 shop: str | None = None, purchased: datetime | None = None, date_source: str | None = None,
                 needs_review: bool = False, source: str = "photo") -> int:
    """purchased - дата покупки, прочитанная с чека; приоритет: QR > дата с чека > момент отправки."""
    now = datetime.now()
    # TODO: возврат без читаемого QR сохраняется как обычная покупка
    if qr and qr.get("datetime"):
        purchased, date_source = qr["datetime"], "qr"
    elif purchased is not None:
        date_source = date_source or "receipt"
    else:
        purchased, date_source = now, "sent"
    total = round(sum(it["price"] for it in items), 2)
    cur = con.execute(
        "INSERT INTO receipts (user_id, created_at, purchased_at, shop, total, operation, qr_fn, qr_fd, qr_fp,"
        " image_hash, items_key, date_source, needs_review, source) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (user_id, now.isoformat(timespec="seconds"), purchased.isoformat(timespec="seconds"), shop, total,
         (qr or {}).get("operation") or "приход", (qr or {}).get("fn"), (qr or {}).get("fd"), (qr or {}).get("fp"),
         hashlib.sha256(image_bytes).hexdigest() if image_bytes else None, items_key(items), date_source,
         int(needs_review), source))
    rid = cur.lastrowid
    con.executemany("INSERT INTO items (receipt_id, name, price, category) VALUES (?,?,?,?)",
                    [(rid, it["name"], it["price"], it["category"]) for it in items])
    con.commit()
    return rid


def receipt_items(con, user_id: int, receipt_id: int) -> list[sqlite3.Row]:
    return con.execute("SELECT i.* FROM items i JOIN receipts r ON r.id=i.receipt_id "
                       "WHERE r.user_id=? AND r.id=? ORDER BY i.id", (user_id, receipt_id)).fetchall()


def set_category(con, user_id: int, item_id: int, category: str) -> sqlite3.Row | None:
    row = con.execute("SELECT i.* FROM items i JOIN receipts r ON r.id=i.receipt_id WHERE r.user_id=? AND i.id=?",
                      (user_id, item_id)).fetchone()
    if row:
        con.execute("UPDATE items SET category=?, category_source='user' WHERE id=?", (category, item_id))
        con.commit()
    return row


def delete_receipt(con, user_id: int, receipt_id: int) -> bool:
    cur = con.execute("DELETE FROM receipts WHERE user_id=? AND id=?", (user_id, receipt_id))
    con.commit()
    return cur.rowcount > 0


def last_receipts(con, user_id: int, n: int = 10) -> list[sqlite3.Row]:
    return con.execute("SELECT r.*, (SELECT COUNT(*) FROM items i WHERE i.receipt_id=r.id) AS n_items "
                       "FROM receipts r WHERE user_id=? ORDER BY purchased_at DESC, id DESC LIMIT ?",
                       (user_id, n)).fetchall()


def items_between(con, user_id: int, start: datetime, end: datetime) -> list[sqlite3.Row]:
    """Все позиции пользователя за [start, end). Суммы по чекам-возвратам - со знаком минус."""
    return con.execute(
        "SELECT i.name, i.category, r.shop, r.purchased_at, r.id AS receipt_id,"
        " CASE WHEN r.operation LIKE 'возврат%' THEN -i.price ELSE i.price END AS amount "
        "FROM items i JOIN receipts r ON r.id=i.receipt_id "
        "WHERE r.user_id=? AND r.purchased_at>=? AND r.purchased_at<? ORDER BY r.purchased_at",
        (user_id, start.isoformat(timespec="seconds"), end.isoformat(timespec="seconds"))).fetchall()


def all_items(con, user_id: int) -> list[sqlite3.Row]:
    return items_between(con, user_id, datetime(2000, 1, 1), datetime(2100, 1, 1))


# --- редактирование ---
def _owned_item(con, user_id: int, item_id: int):
    return con.execute("SELECT i.* FROM items i JOIN receipts r ON r.id=i.receipt_id WHERE r.user_id=? AND i.id=?",
                       (user_id, item_id)).fetchone()


def recalc_total(con, receipt_id: int) -> None:
    con.execute("UPDATE receipts SET total=(SELECT COALESCE(ROUND(SUM(price), 2), 0) FROM items WHERE receipt_id=?) "
                "WHERE id=?", (receipt_id, receipt_id))


def set_price(con, user_id: int, item_id: int, price: float):
    row = _owned_item(con, user_id, item_id)
    if row:
        con.execute("UPDATE items SET price=? WHERE id=?", (round(price, 2), item_id))
        recalc_total(con, row["receipt_id"])
        con.commit()
    return row


def set_name(con, user_id: int, item_id: int, name: str):
    row = _owned_item(con, user_id, item_id)
    if row:
        con.execute("UPDATE items SET name=? WHERE id=?", (name.strip(), item_id))
        con.commit()
    return row


def delete_item(con, user_id: int, item_id: int):
    """Удаляет позицию; если в чеке позиций не осталось - удаляет и чек. Возвращает удалённую строку."""
    row = _owned_item(con, user_id, item_id)
    if row:
        con.execute("DELETE FROM items WHERE id=?", (item_id,))
        left = con.execute("SELECT COUNT(*) FROM items WHERE receipt_id=?", (row["receipt_id"],)).fetchone()[0]
        if left:
            recalc_total(con, row["receipt_id"])
        else:
            con.execute("DELETE FROM receipts WHERE id=?", (row["receipt_id"],))
        con.commit()
    return row


def add_item(con, user_id: int, receipt_id: int, name: str, price: float, category: str) -> int | None:
    if not get_receipt(con, user_id, receipt_id):
        return None
    cur = con.execute("INSERT INTO items (receipt_id, name, price, category, category_source) VALUES (?,?,?,?,'user')",
                      (receipt_id, name.strip(), round(price, 2), category))
    recalc_total(con, receipt_id)
    con.commit()
    return cur.lastrowid


def get_receipt(con, user_id: int, receipt_id: int):
    return con.execute("SELECT * FROM receipts WHERE user_id=? AND id=?", (user_id, receipt_id)).fetchone()


def receipts_page(con, user_id: int, offset: int = 0, n: int = 8) -> tuple[list[sqlite3.Row], int]:
    """Чеки пользователя от новых к старым (страница) и общее их число."""
    rows = con.execute("SELECT r.*, (SELECT COUNT(*) FROM items i WHERE i.receipt_id=r.id) AS n_items "
                       "FROM receipts r WHERE user_id=? ORDER BY purchased_at DESC, id DESC LIMIT ? OFFSET ?",
                       (user_id, n, offset)).fetchall()
    total = con.execute("SELECT COUNT(*) FROM receipts WHERE user_id=?", (user_id,)).fetchone()[0]
    return rows, total



# --- дата, магазин, проверка ---
def set_date(con, user_id: int, receipt_id: int, dt: datetime) -> bool:
    cur = con.execute("UPDATE receipts SET purchased_at=?, date_source='user' WHERE user_id=? AND id=?",
                      (dt.isoformat(timespec="seconds"), user_id, receipt_id))
    con.commit()
    return cur.rowcount > 0


def set_shop(con, user_id: int, receipt_id: int, shop: str | None) -> bool:
    cur = con.execute("UPDATE receipts SET shop=? WHERE user_id=? AND id=?", (shop, user_id, receipt_id))
    con.commit()
    return cur.rowcount > 0


def set_reviewed(con, user_id: int, receipt_id: int) -> None:
    con.execute("UPDATE receipts SET needs_review=0 WHERE user_id=? AND id=?", (user_id, receipt_id))
    con.commit()


def user_shops(con, user_id: int, n: int = 8) -> list[str]:
    return [r["shop"] for r in con.execute(
        "SELECT shop, COUNT(*) c FROM receipts WHERE user_id=? AND shop IS NOT NULL AND shop<>'' "
        "GROUP BY shop ORDER BY c DESC LIMIT ?", (user_id, n))]


def same_shop(con, user_id: int, shop: str | None) -> str | None:
    """Если у пользователя уже есть почти такое же название магазина - используем его (одно заведение
    не должно делиться в отчётах на «YONKS» и «Yonks»). Сравниваем без регистра, кавычек и ООО/ИП."""
    if not shop:
        return shop
    from difflib import SequenceMatcher
    import re as _re

    def key(x):
        x = _re.sub(r"\b(ооо|оао|зао|ао|ип|пао)\b|[\"'«»]", " ", x.lower())
        return " ".join(x.split())
    k = key(shop)
    best = max(((SequenceMatcher(None, k, key(s)).ratio(), s) for s in user_shops(con, user_id, 50)), default=(0, None))
    return best[1] if best[0] >= 0.8 else shop


def delete_all(con, user_id: int, source: str | None = None) -> int:
    """Удалить все чеки пользователя (или только определённого происхождения, например демо-данные)."""
    q, args = "DELETE FROM receipts WHERE user_id=?", [user_id]
    if source:
        q, args = q + " AND source=?", args + [source]
    n = con.execute(q, args).rowcount
    if not source:
        con.execute("DELETE FROM limits WHERE user_id=?", (user_id,))
    con.commit()
    return n


def save_manual(con, user_id: int, name: str, price: float, category: str, when: datetime | None = None,
                shop: str | None = None, source: str = "manual") -> int:
    return save_receipt(con, user_id, b"", [{"name": name, "price": price, "category": category}], None,
                        shop=shop, purchased=when or datetime.now(), date_source="user" if when else "sent",
                        source=source)


# --- лимиты ---
def set_limit(con, user_id: int, category: str, amount: float | None) -> None:
    if amount:
        con.execute("INSERT INTO limits (user_id, category, amount) VALUES (?,?,?) "
                    "ON CONFLICT(user_id, category) DO UPDATE SET amount=excluded.amount", (user_id, category, amount))
    else:
        con.execute("DELETE FROM limits WHERE user_id=? AND category=?", (user_id, category))
    con.commit()


def get_limits(con, user_id: int) -> dict[str, float]:
    return {r["category"]: r["amount"] for r in con.execute("SELECT * FROM limits WHERE user_id=?", (user_id,))}


def receipts_between(con, user_id: int, start: datetime, end: datetime) -> list[sqlite3.Row]:
    return con.execute("SELECT * FROM receipts WHERE user_id=? AND purchased_at>=? AND purchased_at<? "
                       "ORDER BY purchased_at", (user_id, start.isoformat(timespec="seconds"),
                                                 end.isoformat(timespec="seconds"))).fetchall()
