#!/usr/bin/env python3
"""
Диагностический скрипт для проверки состояния Pinecone
"""

import os
import sys
from dotenv import load_dotenv
from pinecone import Pinecone as PineconeClient

load_dotenv()

def diagnose_pinecone():
    """Диагностика состояния Pinecone"""
    print("🔍 Диагностика Pinecone")
    print("="*50)

    # Проверка переменных окружения
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        print("❌ PINECONE_API_KEY не найден в .env файле")
        return False

    print(f"✅ PINECONE_API_KEY найден: {pinecone_api_key[:8]}...")

    try:
        # Инициализация клиента
        pc = PineconeClient(api_key=pinecone_api_key)
        print("✅ Pinecone клиент инициализирован")

        # Получение списка индексов
        indexes = list(pc.list_indexes())
        print(f"📋 Найдено индексов: {len(indexes)}")

        for idx in indexes:
            print(f"  📄 Индекс: {idx.name}")
            print(f"     Размерность: {idx.dimension}")
            print(f"     Метрика: {idx.metric}")
            print(f"     Хост: {idx.host}")

            # Получение статистики индекса
            try:
                index = pc.Index(idx.name)
                stats = index.describe_index_stats()
                print(f"     📊 Статистика:")
                print(f"        Всего векторов: {stats.get('total_vector_count', 0)}")
                print(f"        Пространств имен: {len(stats.get('namespaces', {}))}")

                # Показываем информацию по namespace
                for ns_name, ns_info in stats.get('namespaces', {}).items():
                    print(f"        Namespace '{ns_name}': {ns_info.get('vector_count', 0)} векторов")

            except Exception as e:
                print(f"     ❌ Ошибка получения статистики: {str(e)}")

        return True

    except Exception as e:
        print(f"❌ Ошибка подключения к Pinecone: {str(e)}")
        return False

def clear_pinecone_index():
    """Очистка индекса Pinecone"""
    print("\n🧹 Очистка индекса Pinecone")
    print("="*50)

    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        print("❌ PINECONE_API_KEY не найден")
        return False

    index_name = "book-rag-index"

    try:
        pc = PineconeClient(api_key=pinecone_api_key)

        # Проверяем существует ли индекс
        indexes = [idx.name for idx in pc.list_indexes()]

        if index_name in indexes:
            print(f"🗑️  Удаление индекса {index_name}...")
            pc.delete_index(index_name)
            print("✅ Индекс удален")
            return True
        else:
            print(f"⚠️  Индекс {index_name} не найден")
            return False

    except Exception as e:
        print(f"❌ Ошибка удаления индекса: {str(e)}")
        return False

def test_embeddings():
    """Тестирование создания эмбеддингов"""
    print("\n🧪 Тестирование эмбеддингов")
    print("="*50)

    try:
        from langchain.embeddings import OpenAIEmbeddings

        # Тестируем text-embedding-3-large
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        test_text = "Это тестовый текст для проверки эмбеддингов"

        print(f"📝 Тестовый текст: {test_text}")

        embedding = embeddings.embed_query(test_text)
        print(f"✅ Эмбеддинг создан")
        print(f"   Размерность: {len(embedding)}")
        print(f"   Первые 5 значений: {embedding[:5]}")

        return True

    except Exception as e:
        print(f"❌ Ошибка создания эмбеддинга: {str(e)}")
        return False

if __name__ == "__main__":
    print("🔧 Диагностика системы Pinecone")
    print("="*60)

    if not os.path.exists(".env"):
        print("❌ Файл .env не найден!")
        sys.exit(1)

    # Диагностика
    pinecone_ok = diagnose_pinecone()
    embeddings_ok = test_embeddings()

    print("\n" + "="*60)
    print("📋 Результаты диагностики:")
    print(f"   Pinecone: {'✅' if pinecone_ok else '❌'}")
    print(f"   Эмбеддинги: {'✅' if embeddings_ok else '❌'}")

    if pinecone_ok:
        print("\n🔧 Доступные действия:")
        print("   python diagnose_pinecone.py --clear  # Очистить индекс")

    # Обработка аргументов командной строки
    if len(sys.argv) > 1 and sys.argv[1] == "--clear":
        clear_pinecone_index()