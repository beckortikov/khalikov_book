from book_rag.core import ImprovedBookRAG

if __name__ == "__main__":
    print("=== Улучшенная RAG система для книг 300+ страниц ===\n")

    try:
        # Инициализация
        print("Инициализация системы...")
        rag = ImprovedBookRAG(use_sections=True, cache_embeddings=True)
        print("✓ Система инициализирована успешно\n")

        # Статистика
        stats = rag.get_statistics()
        print("📊 Статистика загруженных данных:")
        print(f"   Документов: {stats.get('total_documents', 0)}")
        print(f"   Чанков: {stats.get('total_chunks', 0)}")
        print(f"   Разделов: {stats.get('sections', 0)}")
        print(f"   Страниц: {stats.get('pages', 0)}")
        print(f"   Типы чанков: {stats.get('chunk_types', {})}\n")

        # Тесты качества
        print("🧪 Запуск тестов качества...")
        test_results = rag.run_quality_tests()
        print(f"   Успешно: {test_results['passed']}/{test_results['total_tests']}")
        print(f"   Процент успеха: {test_results['success_rate']:.1f}%\n")

        # Демонстрация возможностей
        print("🔍 Демонстрация улучшений:")

        demo_questions = [
            "Кто автор книги?",
            "Расскажи о командообразовании",
            "Что говорится о кризисе в бизнесе?"
        ]

        for question in demo_questions:
            print(f"\nВопрос: {question}")
            answer = rag.ask_question(question)
            print(f"Ответ: {answer[:300]}{'...' if len(answer) > 300 else ''}")

        print("\n" + "="*50)
        print("КЛЮЧЕВЫЕ УЛУЧШЕНИЯ РЕАЛИЗОВАНЫ:")
        print("="*50)
        print("✓ КЭШИРОВАНИЕ - быстрая загрузка при повторном запуске")
        print("✓ УМНЫЕ ЧАНКИ - множественные размеры с оптимальным перекрытием")
        print("✓ ПРОДВИНУТЫЙ BM25 - настроенный для русского языка")
        print("✓ АДАПТИВНЫЕ ВЕСА - динамическая настройка dense/sparse поиска")
        print("✓ УЛУЧШЕННЫЙ КОНТЕКСТ - группировка по страницам")
        print("✓ КАЧЕСТВЕННЫЕ ПРОМПТЫ - структурированные инструкции")
        print("✓ ОБРАБОТКА ОШИБОК - информативные сообщения")
        print("✓ МОНИТОРИНГ КАЧЕСТВА - автоматические тесты")
        print("✓ МАСШТАБИРУЕМОСТЬ - поддержка 300+ страниц")
        print("✓ ПРОИЗВОДИТЕЛЬНОСТЬ - оптимизированные модели и батчинг")

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        print("Проверьте:")
        print("- Наличие PDF файлов в папке 'book/'")
        print("- Корректность OPENAI_API_KEY")
        print("- Доступность интернета")
        print("- Достаточность места на диске для кэша")
