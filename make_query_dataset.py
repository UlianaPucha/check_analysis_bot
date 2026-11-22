import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
MONTHS_IN = ["январе", "феврале", "марте", "апреле", "мае", "июне", "июле", "августе", "сентябре", "октябре",
             "ноябре", "декабре"]
MONTHS_ACC = ["январь", "февраль", "март", "апрель", "май", "июнь", "июль", "август", "сентябрь", "октябрь",
              "ноябрь", "декабрь"]

PERIODS = {
    "today": ["сегодня", "за сегодня", "за сегодняшний день"],
    "yesterday": ["вчера", "за вчера", "за вчерашний день"],
    "this_week": ["на этой неделе", "за эту неделю", "за текущую неделю"],
    "last_week": ["на прошлой неделе", "за прошлую неделю"],
    "last_7_days": ["за неделю", "за последние 7 дней", "за последнюю неделю", "за 7 дней"],
    "this_month": ["в этом месяце", "за этот месяц", "в текущем месяце", "за месяц"],
    "last_month": ["в прошлом месяце", "за прошлый месяц", "за предыдущий месяц"],
    "last_30_days": ["за последние 30 дней", "за 30 дней", "за последний месяц"],
    "this_year": ["в этом году", "за год", "за этот год", "с начала года"],
    "all": ["за всё время", "за все время", "вообще", "всего"],
}
CATEGORY_WORDS = {
    "еда": ["еду", "продукты", "питание", "еду и продукты"],
    "напитки": ["напитки", "алкоголь", "выпивку"],
    "бытовое": ["бытовое", "бытовую химию", "товары для дома", "хозтовары"],
    "здоровье": ["лекарства", "здоровье", "аптеку", "медицину"],
    "одежда": ["одежду", "обувь", "одежду и обувь", "шмотки"],
    "транспорт": ["транспорт", "проезд", "бензин и проезд"],
    "развлечения": ["развлечения", "досуг", "игры и кино"],
    "другое": ["прочее", "другое", "разное"],
}
TEXT_WORDS = {  # слово в вопросе -> что искать в названиях товаров/магазинов
    "кофе": "кофе", "пиво": "пиво", "молоко": "молоко", "хлеб": "хлеб", "сигареты": "сигарет",
    "шоколад": "шоколад", "сыр": "сыр", "воду": "вода", "бананы": "банан", "чипсы": "чипсы",
    "такси": "такси", "бензин": "бензин", "капучино": "капучино", "пиццу": "пицц", "пакеты": "пакет",
    "в Ленте": "лента", "в Магните": "магнит", "в Пятёрочке": "агроторг", "мясо": "мяс", "курицу": "кур",
}
TEMPLATES = {
    "sum": ["Сколько я потратил{F} {P}?", "Сколько я потратила{F} {P}?", "Сколько ушло денег{F} {P}?",
            "Сколько денег{F} {P}", "Мои траты{F} {P}", "Траты{F} {P}", "Сколько потрачено{F} {P}?",
            "{P} сколько потратил{F}?", "Покажи сумму трат{F} {P}", "сколько я слил{F} {P}"],
    "count": ["Сколько чеков {P}?", "Сколько раз я покупал{F2} {P}?", "Сколько покупок {P}?",
              "Как часто я покупаю{F2} {P}?"],
    "avg_receipt": ["Какой у меня средний чек {P}?", "Средний чек {P}", "Сколько в среднем я трачу за раз {P}?"],
    "top_items": ["На что я больше всего трачу {P}?", "Самые затратные товары {P}", "Топ покупок {P}",
                  "Что я чаще всего покупаю {P}?", "На какие товары уходит больше всего денег {P}?"],
    "top_categories": ["Покажи траты по категориям {P}", "На какие категории уходит больше всего {P}?",
                       "Распредели мои траты по категориям {P}", "Куда уходят деньги {P}?"],
    "top_shops": ["Где я больше всего трачу {P}?", "В каких магазинах я трачу больше всего {P}?",
                  "Топ магазинов {P}"],
    "max_item": ["Какая самая дорогая покупка {P}?", "Что я купил самое дорогое {P}?",
                 "Самая крупная трата{F} {P}"],
}
NOT_QUESTIONS = ["привет", "спасибо!", "как дела?", "что ты умеешь?", "кто ты", "ок", "понятно", "помощь",
                 "расскажи анекдот", "какая погода завтра?", "hello", "ты бот?", "круто", "👍", "help",
                 "как пользоваться ботом?", "что делать?", "отличная работа", "забудь", "тест"]


def period_variants(today_year=2026):
    out = [(k, p) for k, ps in PERIODS.items() for p in ps]
    for m in range(1, 13):
        out += [(f"month:{m:02d}", f"в {MONTHS_IN[m - 1]}"), (f"month:{m:02d}", f"за {MONTHS_ACC[m - 1]}")]
        y = random.choice([today_year - 1, today_year])
        out += [(f"month:{m:02d}.{y}", f"в {MONTHS_IN[m - 1]} {y}"), (f"month:{m:02d}.{y}", f"за {MONTHS_ACC[m - 1]} {y} года")]
    return out


def make_example():
    metric = random.choices(list(TEMPLATES) + ["none"], weights=[35, 8, 7, 8, 8, 6, 6, 5])[0]
    if metric == "none":
        return random.choice(NOT_QUESTIONS), {"metric": "none", "period": None, "category": None, "text": None}
    period, ptext = random.choice(period_variants())
    tpl = random.choice(TEMPLATES[metric])
    category = text = None
    f = f2 = ""
    r = random.random()
    if "{F}" in tpl or "{F2}" in tpl:
        if r < 0.4:
            category = random.choice(list(CATEGORY_WORDS))
            f = " на " + random.choice(CATEGORY_WORDS[category])
            f2 = " " + random.choice(CATEGORY_WORDS[category])
        elif r < 0.75:
            word = random.choice(list(TEXT_WORDS))
            text = TEXT_WORDS[word]
            f = (" " + word) if word.startswith("в ") else " на " + word
            f2 = "" if word.startswith("в ") else " " + word
    elif metric in ("top_items", "max_item") and r < 0.25:
        category = random.choice(list(CATEGORY_WORDS))
        tpl = tpl.replace("{P}", "{P} в категории " + category)
    if random.random() < 0.08 and period == "this_month":
        ptext = ""  # без периода = текущий месяц
    q = tpl.replace("{F}", f).replace("{F2}", f2).replace("{P}", ptext)
    q = " ".join(q.split()).replace(" ?", "?")
    if random.random() < 0.3:
        q = q.lower()
    if random.random() < 0.2:
        q = q.rstrip("?")
    return q, {"metric": metric, "period": period, "category": category, "text": text}


# ручные тестовые вопросы (в обучение не идут), сегодня = 15.11.2026
TEST_QUESTIONS = [
    ("сколько я прогулял денег в кафе в этом месяце", {"metric": "sum", "period": "this_month", "category": "еда", "text": None}),
    ("Скажи, сколько ушло на алкашку за прошлый месяц", {"metric": "sum", "period": "last_month", "category": "напитки", "text": None}),
    ("а на кофе сколько за неделю?", {"metric": "sum", "period": "last_7_days", "category": None, "text": "кофе"}),
    ("сколько я в октябре оставил в Ленте", {"metric": "sum", "period": "month:10", "category": None, "text": "лента"}),
    ("покажи что больше всего денег съедает", {"metric": "top_categories", "period": "this_month", "category": None, "text": None}),
    ("какой средний чек был в сентябре", {"metric": "avg_receipt", "period": "month:09", "category": None, "text": None}),
    ("сколько раз за месяц я брал пиво", {"metric": "count", "period": "this_month", "category": None, "text": "пиво"}),
    ("самая дорогая вещь которую я купил за год", {"metric": "max_item", "period": "this_year", "category": None, "text": None}),
    ("траты на аптеку вчера", {"metric": "sum", "period": "yesterday", "category": "здоровье", "text": None}),
    ("сколько всего потратил с начала года", {"metric": "sum", "period": "this_year", "category": None, "text": None}),
    ("в каком магазине я больше всего трачу", {"metric": "top_shops", "period": "this_month", "category": None, "text": None}),
    ("сколько стоили мои продукты за последние 30 дней", {"metric": "sum", "period": "last_30_days", "category": "еда", "text": None}),
    ("привет! что умеешь?", {"metric": "none", "period": None, "category": None, "text": None}),
    ("спасибо большое", {"metric": "none", "period": None, "category": None, "text": None}),
    ("сколько потратил сегодня", {"metric": "sum", "period": "today", "category": None, "text": None}),
    ("на что уходят деньги в этом году", {"metric": "top_categories", "period": "this_year", "category": None, "text": None}),
    ("какие товары я покупал чаще всего на прошлой неделе", {"metric": "top_items", "period": "last_week", "category": None, "text": None}),
    ("сколько я потратила на одежду в мае 2026", {"metric": "sum", "period": "month:05.2026", "category": "одежда", "text": None}),
    ("сколько чеков я загрузил за всё время", {"metric": "count", "period": "all", "category": None, "text": None}),
    ("траты на такси за неделю", {"metric": "sum", "period": "last_7_days", "category": None, "text": "такси"}),
    ("сколько ушло на бытовую химию в прошлом месяце", {"metric": "sum", "period": "last_month", "category": "бытовое", "text": None}),
    ("самая крупная покупка еды в этом месяце", {"metric": "max_item", "period": "this_month", "category": "еда", "text": None}),
    ("сколько я потратил на развлечения за этот год", {"metric": "sum", "period": "this_year", "category": "развлечения", "text": None}),
    ("топ магазинов за прошлый месяц", {"metric": "top_shops", "period": "last_month", "category": None, "text": None}),
    ("сколько я трачу на шоколад вообще", {"metric": "sum", "period": "all", "category": None, "text": "шоколад"}),
]


def main():
    random.seed(1)
    seen, train = set(), []
    while len(train) < 3000:
        q, target = make_example()
        if q.lower() not in seen:
            seen.add(q.lower())
            train.append({"question": q, "query": target})
    with open(os.path.join(HERE, "data", "query_train.jsonl"), "w", encoding="utf-8") as f:
        for ex in train:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with open(os.path.join(HERE, "data", "query_test.jsonl"), "w", encoding="utf-8") as f:
        for q, t in TEST_QUESTIONS:
            f.write(json.dumps({"question": q, "query": t}, ensure_ascii=False) + "\n")
    print(f"train {len(train)}, test (вручную) {len(TEST_QUESTIONS)}")
    for ex in random.sample(train, 6):
        print("  ", ex["question"], "->", ex["query"])


if __name__ == "__main__":
    main()
