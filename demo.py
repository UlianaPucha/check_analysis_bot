import random
from datetime import datetime, timedelta

import storage

# магазин -> [(товар, категория, цена)], цены понемногу растут
SHOPS = {
    "Пятёрочка": [("Молоко Простоквашино 2,5% 930мл", "еда", 89), ("Хлеб Бородинский 300г", "еда", 54),
                  ("Яйцо куриное С1 10шт", "еда", 119), ("Сыр Российский 200г", "еда", 189),
                  ("Бананы", "еда", 129), ("Кефир 1% 900мл", "еда", 84), ("Вода Святой источник 1,5л", "напитки", 59),
                  ("Пакет-майка", "другое", 9), ("Гречка Мистраль 900г", "еда", 109), ("Сок Добрый яблоко 1л", "напитки", 129)],
    "Лента": [("Курица филе охл 1кг", "еда", 449), ("Макароны Barilla 450г", "еда", 139), ("Средство Fairy 450мл", "бытовое", 199),
              ("Бумага туалетная Zewa 8шт", "бытовое", 329), ("Кофе Jacobs Monarch 95г", "напитки", 349),
              ("Масло сливочное 82,5% 180г", "еда", 219), ("Корм Whiskas 75г", "бытовое", 39), ("Пиво Жигулевское 0,5л", "напитки", 79)],
    "Вкусно и точка": [("Биг Хит", "еда", 239), ("Картофель фри средний", "еда", 129), ("Кока-Кола 0,5л", "напитки", 119),
                       ("Капучино 300мл", "напитки", 159)],
    "Кофейня Зерно": [("Капучино 300мл", "напитки", 210), ("Круассан с миндалём", "еда", 190), ("Раф кофе 350мл", "напитки", 260)],
    "Аптека Здоровье": [("Нурофен 200мг 20таб", "здоровье", 279), ("Витамин D3 2000МЕ", "здоровье", 449),
                        ("Пластырь бактерицидный", "здоровье", 99)],
    "Яндекс Такси": [("Поездка Яндекс Такси", "транспорт", 420)],
    "Лукойл": [("Бензин АИ-95", "транспорт", 2400)],
    "Кинотеатр Октябрь": [("Билет в кино 2D", "развлечения", 450), ("Попкорн соленый", "еда", 290)],
}
PLAN = [  # (магазин, сколько раз в неделю в среднем, сколько позиций в чеке)
    ("Пятёрочка", 3.5, (3, 7)), ("Лента", 1.0, (4, 8)), ("Вкусно и точка", 1.2, (2, 3)), ("Кофейня Зерно", 2.0, (1, 2)),
    ("Аптека Здоровье", 0.25, (1, 2)), ("Яндекс Такси", 1.2, (1, 1)), ("Лукойл", 0.4, (1, 1)),
    ("Кинотеатр Октябрь", 0.3, (1, 2)),
]


def generate(con, user_id: int, months_back: int = 3, seed: int = 42) -> int:
    rnd = random.Random(seed)
    now = datetime.now()
    first = datetime(now.year, now.month, 1)
    for _ in range(months_back):
        first = (first - timedelta(days=1)).replace(day=1)
    total_days = (now - first).days + 1
    n = 0
    for d in range(total_days):
        day = first + timedelta(days=d)
        inflation = 1 + 0.04 * d / total_days  # рост цен ~4% за период
        for shop, per_week, (k1, k2) in PLAN:
            if rnd.random() >= per_week / 7:
                continue
            goods = rnd.sample(SHOPS[shop], min(len(SHOPS[shop]), rnd.randint(k1, k2)))
            items = []
            for name, cat, base in goods:
                qty = 1 if cat in ("транспорт", "здоровье") or rnd.random() < 0.8 else 2
                price = round(base * inflation * qty * (rnd.uniform(0.7, 1.5) if shop == "Яндекс Такси" else 1), 2)
                items.append({"name": name, "price": price, "category": cat})
            when = day.replace(hour=rnd.randint(8, 21), minute=rnd.randint(0, 59))
            storage.save_receipt(con, user_id, b"", items, None, shop=shop, purchased=when, date_source="user",
                                 source="demo")
            n += 1
    # одна крупная покупка, чтобы сработал поиск необычных трат
    storage.save_receipt(con, user_id, b"", [{"name": "Пуховик зимний", "price": 14990.0, "category": "одежда"},
                                             {"name": "Ботинки зимние", "price": 8990.0, "category": "одежда"}], None,
                         shop="Спортмастер", purchased=now.replace(hour=19, minute=10) - timedelta(days=min(2, now.day - 1)),
                         date_source="user", source="demo")
    return n + 1


def clear(con, user_id: int) -> int:
    return storage.delete_all(con, user_id, source="demo")
