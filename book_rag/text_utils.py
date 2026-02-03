import re
from typing import List, Set

def advanced_text_cleaning(text: str) -> str:
    """Улучшенная очистка текста"""
    # Базовая очистка
    text = re.sub(r'\s+', ' ', text)  # Множественные пробелы
    text = re.sub(r'-\s*\n\s*', '', text)  # Переносы слов
    text = re.sub(r'\n+', '\n', text)  # Множественные переносы строк

    # Удаление служебных символов, но сохранение структуры
    text = re.sub(r'[^\w\s.,!?:;()\[\]"«»№—–-]', '', text)

    # Исправление кавычек
    text = text.replace('«', '"').replace('»', '"')
    text = text.replace('„', '"').replace('"', '"')

    # Удаление лишних пробелов вокруг знаков препинания
    text = re.sub(r'\s+([.,!?:;])', r'\1', text)
    text = re.sub(r'([.,!?:;])\s+', r'\1 ', text)

    return text.strip()

def simple_sentence_split(text: str) -> List[str]:
    """Простое разделение на предложения"""
    sentences = re.split(r'[.!?]+\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]

def advanced_tokenize(text: str) -> List[str]:
    """Улучшенная токенизация для русского языка"""
    # Нормализация
    text = text.lower()

    # Разделение по знакам препинания с сохранением важных
    text = re.sub(r'([.!?,:;])', r' \1 ', text)
    text = re.sub(r'[^\w\s.!?,:;]', ' ', text)

    tokens = text.split()

    # Фильтрация стоп-слов (расширенный список)
    stop_words = {
        'а', 'и', 'но', 'или', 'да', 'нет', 'не', 'ни', 'в', 'во', 'на', 'по', 'за',
        'к', 'с', 'со', 'от', 'до', 'для', 'при', 'о', 'об', 'что', 'как', 'так',
        'это', 'то', 'те', 'тот', 'та', 'же', 'уже', 'еще', 'ещё', 'только', 'лишь',
        'будет', 'была', 'было', 'были', 'есть', 'быть', 'был'
    }

    # Фильтруем токены
    filtered_tokens = []
    for token in tokens:
        if (len(token) > 2 and
            token not in stop_words and
            not token.isdigit() and
            token not in '.!?,:;'):
            filtered_tokens.append(token)

    return filtered_tokens

def is_suspicious_query(question: str) -> bool:
    """Проверка на подозрительные запросы"""
    suspicious_patterns = [
        "ignore previous", "ignore above", "forget everything",
        "system prompt", "system message", "you are now",
        "act as", "pretend to be", "roleplay as", "jailbreak"
    ]

    question_lower = question.lower()
    return any(pattern in question_lower for pattern in suspicious_patterns)

def enhance_query(question: str) -> str:
    """Улучшение запроса пользователя"""
    # Нормализация
    enhanced = question.strip()

    # Добавление контекста для коротких вопросов
    if len(enhanced.split()) <= 3:
        context_hints = {
            "автор": "кто автор книги",
            "глава": "о чем глава в книге",
            "часть": "содержание части книги",
            "встреча": "что происходит во встрече",
            "бизнес": "бизнес советы из книги"
        }

        for hint, expansion in context_hints.items():
            if hint in enhanced.lower():
                enhanced = f"{expansion} {enhanced}"
                break

    return enhanced
