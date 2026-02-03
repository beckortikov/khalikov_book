# Руководство по работе с RAG системой

## Описание

Улучшенная RAG (Retrieval-Augmented Generation) система для работы с книгой "Бизнес в диалоге: от малого к невозможному" Анвара Халикова.

## Возможности

- 🔍 **Гибридный поиск**: Комбинация векторного поиска и BM25
- 📊 **Адаптивные веса**: Автоматическая оптимизация для разных типов запросов
- 💾 **Кэширование**: Быстрая загрузка при повторном запуске
- 📚 **Поддержка разделов**: Поиск по конкретным разделам книги
- 🎯 **Умные чанки**: Множественные размеры с оптимальным перекрытием

## Установка

```bash
# Установка зависимостей
pip install -r requirements.txt

# Настройка переменных окружения
cp .env.example .env
# Отредактируйте .env и добавьте ваши API ключи
```

## Использование

### 1. Программный интерфейс (Python)

#### Базовое использование

```python
from book_rag.core import ImprovedBookRAG

# Инициализация системы
rag = ImprovedBookRAG(
    use_sections=True,      # Использовать разделы книги
    cache_embeddings=True   # Кэшировать эмбеддинги
)

# Задать вопрос
answer = rag.ask_question("Кто автор книги?")
print(answer)
```

#### Поиск по конкретному разделу

```python
# Поиск в конкретном разделе
answer = rag.search_by_section(
    section_name="командообразование",
    query="Как формировать команду?"
)
print(answer)
```

#### Получение списка доступных разделов

```python
# Получить список разделов
sections = rag.get_available_sections()
print("Доступные разделы:", sections)
```

#### Статистика системы

```python
# Получить статистику
stats = rag.get_statistics()
print(f"Всего документов: {stats['total_documents']}")
print(f"Всего чанков: {stats['total_chunks']}")
print(f"Разделов: {stats['sections']}")
```

#### Пересоздание кэша

```python
# Принудительное пересоздание кэша
success = rag.force_rebuild_cache()
if success:
    print("Кэш успешно пересоздан")
```

### 2. Командная строка (CLI)

```bash
# Запуск демонстрации
python main.py
```

### 3. Web API (FastAPI)

#### Запуск сервера

```bash
# Запуск API сервера
python api.py

# Или с uvicorn
uvicorn api:app --host 0.0.0.0 --port 8001
```

#### Примеры запросов

**Задать вопрос:**

```bash
curl -X POST "http://localhost:8001/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "Кто автор книги?"}'
```

**Поиск по разделу:**

```bash
curl -X POST "http://localhost:8001/search_section" \
  -H "Content-Type: application/json" \
  -d '{
    "section_name": "командообразование",
    "query": "Как формировать команду?"
  }'
```

**Получить список разделов:**

```bash
curl -X GET "http://localhost:8001/sections"
```

**Проверка здоровья API:**

```bash
curl -X GET "http://localhost:8001/health"
```

### 4. Streamlit UI

```bash
# Запуск веб-интерфейса
streamlit run stremlit_ui.py
```

Откройте браузер по адресу `http://localhost:8501`

### 5. Python клиент для API

```bash
# Проверка статуса
python client.py status

# Задать вопрос
python client.py ask "Кто автор книги?"

# Интерактивный режим
python client.py interactive
```

## Методы класса ImprovedBookRAG

### `ask_question(question: str, section_filter: Optional[str] = None) -> str`

Основной метод для получения ответов на вопросы.

**Параметры:**
- `question` (str): Вопрос о книге (макс. 200 символов)
- `section_filter` (Optional[str]): Фильтр по разделу (опционально)

**Возвращает:**
- `str`: Ответ с источниками

**Пример:**

```python
# Общий вопрос
answer = rag.ask_question("Что такое бизнес?")

# Вопрос с фильтром по разделу
answer = rag.ask_question(
    "Что говорится о кризисе?",
    section_filter="кризис"
)
```

### `search_by_section(section_name: str, query: str = "") -> str`

Поиск информации в конкретном разделе.

**Параметры:**
- `section_name` (str): Название раздела
- `query` (str): Поисковый запрос (опционально)

**Возвращает:**
- `str`: Ответ из указанного раздела

**Пример:**

```python
# Общая информация о разделе
answer = rag.search_by_section("командообразование")

# Конкретный вопрос по разделу
answer = rag.search_by_section(
    "командообразование",
    "Какие этапы командообразования?"
)
```

### `get_available_sections() -> List[str]`

Получение списка доступных разделов книги.

**Возвращает:**
- `List[str]`: Список названий разделов

**Пример:**

```python
sections = rag.get_available_sections()
for section in sections:
    print(f"- {section}")
```

### `get_statistics() -> Dict`

Получение статистики о загруженных данных.

**Возвращает:**
- `Dict`: Словарь со статистикой

**Пример:**

```python
stats = rag.get_statistics()
print(f"Документов: {stats['total_documents']}")
print(f"Чанков: {stats['total_chunks']}")
print(f"Типы чанков: {stats['chunk_types']}")
```

### `force_rebuild_cache() -> bool`

Принудительное пересоздание кэша эмбеддингов.

**Возвращает:**
- `bool`: True если успешно, False в случае ошибки

**Пример:**

```python
if rag.force_rebuild_cache():
    print("Кэш пересоздан")
else:
    print("Ошибка при пересоздании кэша")
```

### `run_quality_tests() -> Dict`

Запуск тестов качества системы.

**Возвращает:**
- `Dict`: Результаты тестов

**Пример:**

```python
results = rag.run_quality_tests()
print(f"Успешно: {results['passed']}/{results['total_tests']}")
print(f"Процент успеха: {results['success_rate']:.1f}%")
```

## Полный пример использования

```python
from book_rag.core import ImprovedBookRAG

def main():
    # Инициализация
    print("Инициализация RAG системы...")
    rag = ImprovedBookRAG(
        use_sections=True,
        cache_embeddings=True
    )
    
    # Получение статистики
    stats = rag.get_statistics()
    print(f"\nЗагружено {stats['total_documents']} документов")
    print(f"Создано {stats['total_chunks']} чанков")
    
    # Получение разделов
    sections = rag.get_available_sections()
    print(f"\nДоступно {len(sections)} разделов")
    
    # Примеры вопросов
    questions = [
        "Кто автор книги?",
        "О чем книга?",
        "Что такое командообразование?"
    ]
    
    for question in questions:
        print(f"\n{'='*60}")
        print(f"Вопрос: {question}")
        print(f"{'='*60}")
        
        answer = rag.ask_question(question)
        print(answer)
    
    # Поиск по разделу
    print(f"\n{'='*60}")
    print("Поиск в разделе 'кризис'")
    print(f"{'='*60}")
    
    answer = rag.search_by_section(
        "кризис",
        "Как справляться с кризисом?"
    )
    print(answer)

if __name__ == "__main__":
    main()
```

## Структура проекта

```
khalikov_book/
├── book_rag/              # Основной пакет
│   ├── __init__.py
│   ├── config.py          # Конфигурация
│   ├── text_utils.py      # Утилиты обработки текста
│   ├── data.py            # Загрузка и чанкинг
│   ├── storage.py         # Векторное хранилище
│   ├── search.py          # Поисковые алгоритмы
│   └── core.py            # Основная логика RAG
├── book/                  # PDF файлы книги
├── cache/                 # Кэш эмбеддингов
├── main.py               # CLI интерфейс
├── api.py                # FastAPI сервер
├── stremlit_ui.py        # Streamlit UI
├── client.py             # Python клиент для API
├── diagnose_pinecone.py  # Диагностика Pinecone
└── requirements.txt      # Зависимости
```

## Переменные окружения

Создайте файл `.env` со следующими переменными:

```env
# OpenAI API ключ (обязательно)
OPENAI_API_KEY=sk-...

# Pinecone API ключ (опционально, для облачного хранилища)
PINECONE_API_KEY=...
PINECONE_ENVIRONMENT=gcp-starter
```

## Советы по использованию

### Типы вопросов

**Точные вопросы:**
```python
answer = rag.ask_question("Что именно говорится о ключевом факторе успеха?")
```

**Концептуальные вопросы:**
```python
answer = rag.ask_question("Как работает управление командой?")
```

**Вопросы с именами:**
```python
answer = rag.ask_question("Кто такая Шоира Музаффаровна?")
```

### Оптимизация производительности

1. **Используйте кэширование**: При первом запуске система создаст кэш, последующие запуски будут быстрее
2. **Фильтруйте по разделам**: Если знаете раздел, используйте `section_filter` для более точных результатов
3. **Короткие вопросы**: Система автоматически расширяет короткие вопросы, но лучше формулировать полные

### Обработка ошибок

```python
try:
    answer = rag.ask_question("Ваш вопрос")
    print(answer)
except Exception as e:
    print(f"Ошибка: {e}")
```

## Диагностика

### Проверка Pinecone

```bash
python diagnose_pinecone.py
```

### Очистка индекса Pinecone

```bash
python diagnose_pinecone.py --clear
```

## Поддержка

При возникновении проблем проверьте:
1. Наличие PDF файлов в папке `book/`
2. Корректность `OPENAI_API_KEY` в `.env`
3. Доступность интернета
4. Достаточность места на диске для кэша
5. Логи в файле `rag.log`
