"""
Пример использования RAG системы через программный интерфейс
"""

from book_rag.core import ImprovedBookRAG
from typing import Optional


class RAGClient:
    """Удобный клиент для работы с RAG системой"""
    
    def __init__(self, use_sections: bool = True, cache_embeddings: bool = True):
        """
        Инициализация клиента
        
        Args:
            use_sections: Использовать разделы книги
            cache_embeddings: Кэшировать эмбеддинги для быстрой загрузки
        """
        self.rag = ImprovedBookRAG(
            use_sections=use_sections,
            cache_embeddings=cache_embeddings
        )
    
    def ask(self, question: str, section: Optional[str] = None) -> str:
        """
        Задать вопрос системе
        
        Args:
            question: Вопрос о книге
            section: Опциональный фильтр по разделу
            
        Returns:
            Ответ системы
            
        Example:
            >>> client = RAGClient()
            >>> answer = client.ask("Кто автор книги?")
            >>> print(answer)
        """
        return self.rag.ask_question(question, section_filter=section)
    
    def search_section(self, section_name: str, query: str = "") -> str:
        """
        Поиск в конкретном разделе
        
        Args:
            section_name: Название раздела
            query: Поисковый запрос (опционально)
            
        Returns:
            Ответ из раздела
            
        Example:
            >>> client = RAGClient()
            >>> answer = client.search_section("командообразование", "Как формировать команду?")
            >>> print(answer)
        """
        return self.rag.search_by_section(section_name, query)
    
    def get_sections(self) -> list:
        """
        Получить список доступных разделов
        
        Returns:
            Список названий разделов
            
        Example:
            >>> client = RAGClient()
            >>> sections = client.get_sections()
            >>> print(sections)
        """
        return self.rag.get_available_sections()
    
    def get_stats(self) -> dict:
        """
        Получить статистику системы
        
        Returns:
            Словарь со статистикой
            
        Example:
            >>> client = RAGClient()
            >>> stats = client.get_stats()
            >>> print(f"Документов: {stats['total_documents']}")
        """
        return self.rag.get_statistics()
    
    def rebuild_cache(self) -> bool:
        """
        Пересоздать кэш эмбеддингов
        
        Returns:
            True если успешно, False в случае ошибки
            
        Example:
            >>> client = RAGClient()
            >>> if client.rebuild_cache():
            ...     print("Кэш пересоздан")
        """
        return self.rag.force_rebuild_cache()
    
    def run_tests(self) -> dict:
        """
        Запустить тесты качества
        
        Returns:
            Результаты тестов
            
        Example:
            >>> client = RAGClient()
            >>> results = client.run_tests()
            >>> print(f"Успешно: {results['passed']}/{results['total_tests']}")
        """
        return self.rag.run_quality_tests()


def example_basic_usage():
    """Пример базового использования"""
    print("=== Базовое использование ===\n")
    
    # Инициализация
    client = RAGClient()
    
    # Задать вопрос
    answer = client.ask("Кто автор книги?")
    print(f"Вопрос: Кто автор книги?")
    print(f"Ответ: {answer}\n")


def example_section_search():
    """Пример поиска по разделу"""
    print("=== Поиск по разделу ===\n")
    
    client = RAGClient()
    
    # Получить список разделов
    sections = client.get_sections()
    print(f"Доступные разделы: {', '.join(sections[:5])}...\n")
    
    # Поиск в разделе
    answer = client.search_section(
        "командообразование",
        "Какие этапы командообразования?"
    )
    print(f"Раздел: командообразование")
    print(f"Вопрос: Какие этапы командообразования?")
    print(f"Ответ: {answer}\n")


def example_statistics():
    """Пример получения статистики"""
    print("=== Статистика системы ===\n")
    
    client = RAGClient()
    
    stats = client.get_stats()
    print(f"Всего документов: {stats['total_documents']}")
    print(f"Всего чанков: {stats['total_chunks']}")
    print(f"Разделов: {stats['sections']}")
    print(f"Страниц: {stats['pages']}")
    print(f"Типы чанков: {stats['chunk_types']}\n")


def example_multiple_questions():
    """Пример множественных вопросов"""
    print("=== Множественные вопросы ===\n")
    
    client = RAGClient()
    
    questions = [
        "Кто автор книги?",
        "О чем книга?",
        "Что такое командообразование?",
        "Как справляться с кризисом?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"{i}. Вопрос: {question}")
        answer = client.ask(question)
        # Показываем только первые 200 символов ответа
        short_answer = answer[:200] + "..." if len(answer) > 200 else answer
        print(f"   Ответ: {short_answer}\n")


def example_with_section_filter():
    """Пример с фильтром по разделу"""
    print("=== Вопросы с фильтром по разделу ===\n")
    
    client = RAGClient()
    
    # Вопрос с фильтром
    answer = client.ask(
        "Что говорится о кризисе?",
        section="кризис"
    )
    print(f"Вопрос: Что говорится о кризисе?")
    print(f"Раздел: кризис")
    print(f"Ответ: {answer}\n")


def example_error_handling():
    """Пример обработки ошибок"""
    print("=== Обработка ошибок ===\n")
    
    try:
        client = RAGClient()
        
        # Слишком длинный вопрос
        long_question = "Вопрос " * 100
        answer = client.ask(long_question)
        print(answer)
        
    except Exception as e:
        print(f"Ошибка: {e}\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Примеры использования RAG клиента")
    print("=" * 60 + "\n")
    
    # Запуск примеров
    example_basic_usage()
    example_section_search()
    example_statistics()
    example_multiple_questions()
    example_with_section_filter()
    example_error_handling()
    
    print("=" * 60)
    print("Все примеры выполнены!")
    print("=" * 60)
