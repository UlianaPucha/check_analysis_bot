"""Нормализация строки чека перед классификацией (одинаково при обучении и в боте)."""
import re

OCR_FIX = {"0": "о", "3": "з", "6": "б", "1": "л"}  # цифры, которые OCR путает с буквами
LATIN_TO_CYR = str.maketrans("aceopxykmtbh", "асеорхукмтвн")  # латинские буквы-двойники кириллицы


def fix_word(word: str) -> str:
    return word.translate(LATIN_TO_CYR) if re.search(r"[а-я]", word) else word


def normalize(text: str) -> str:
    text = text.lower().replace("ё", "е")
    text = re.sub(r"^\s*\*?\d{4,}\s*", "", text)        # артикул в начале: '*7479923 НОСКИ'
    text = re.sub(r"^\s*\d{1,3}[.)]\s*", "", text)       # номер позиции: '15. НУРОФЕН'
    # цифра между двумя буквами - почти всегда ошибка OCR: 'хл0ргексидин' -> 'хлоргексидин'
    text = re.sub(r"(?<=[а-я])[0136](?=[а-я])", lambda m: OCR_FIX[m.group()], text)
    return " ".join(fix_word(w) for w in text.split())
