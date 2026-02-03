import os
import time
import pickle
import logging
from typing import List, Optional, Tuple, Any, Dict
from pinecone import Pinecone as PineconeClient

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.vectorstores import Pinecone
from langchain.schema import Document

from book_rag.config import logger, SECTIONS_MAPPING

class StorageManager:
    def __init__(self, cache_dir: str = "cache", pinecone_api_key: Optional[str] = None):
        self.cache_dir = cache_dir
        self.pinecone_api_key = pinecone_api_key
        self.pinecone_environment = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")
        self.index_name = "book-rag-index-v2"
        self.pinecone_index = None
        self.pinecone_available = False
        
        os.makedirs(self.cache_dir, exist_ok=True)
        
        if self.pinecone_api_key:
             self.pinecone_available = self._init_pinecone()
        else:
             logger.info("Pinecone API ключ не предоставлен, будет использован локальный FAISS")

    def _init_pinecone(self) -> bool:
        """Инициализация Pinecone с улучшенной обработкой ошибок"""
        try:
            self.pc = PineconeClient(api_key=self.pinecone_api_key)

            # Проверяем существование индекса
            existing_indexes = [index.name for index in self.pc.list_indexes()]

            if self.index_name not in existing_indexes:
                logger.info(f"Создание нового Pinecone индекса: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=3072,  # text-embedding-3-large
                    metric="cosine",
                    spec={"serverless": {"cloud": "aws", "region": "us-east-1"}}
                )
                # Ждем готовности индекса
                time.sleep(10)

            self.pinecone_index = self.pc.Index(self.index_name)
            logger.info("Pinecone успешно инициализирован")
            return True

        except Exception as e:
            logger.warning(f"Не удалось инициализировать Pinecone: {str(e)}. Используем FAISS")
            return False

    def check_cache_validity(self) -> bool:
        """Проверка валидности кэша"""
        cache_file = os.path.join(self.cache_dir, "processed_data.pkl")
        try:
            if not os.path.exists(cache_file):
                return False
                
            cache_time = os.path.getmtime(cache_file)

            # Проверяем время изменения PDF файлов
            for file_path in SECTIONS_MAPPING.values():
                if os.path.exists(file_path):
                    file_time = os.path.getmtime(file_path)
                    if file_time > cache_time:
                        return False

            return True
        except Exception:
            return False

    def load_from_cache(self) -> Optional[Dict[str, Any]]:
        """Загрузка данных из кэша"""
        cache_file = os.path.join(self.cache_dir, "processed_data.pkl")
        embeddings_cache = os.path.join(self.cache_dir, "embeddings.pkl")
        
        if not self.check_cache_validity():
            logger.info("Кэш устарел или отсутствует")
            return None

        logger.info("Загрузка данных из кэша...")
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            # Векторное хранилище загружаем отдельно
            if self.pinecone_available and self._has_pinecone_data():
                 cached_data['vectorstore'] = self._load_pinecone_vectorstore()
            elif os.path.exists(embeddings_cache):
                logger.info("Загрузка FAISS из кэша...")
                with open(embeddings_cache, 'rb') as f:
                    cached_data['vectorstore'] = pickle.load(f)
            else:
                 logger.warning("Кэш эмбеддингов не найден")
                 return None

            logger.info("Данные успешно загружены из кэша")
            return cached_data

        except Exception as e:
            logger.warning(f"Ошибка загрузки кэша: {e}")
            return None

    def _has_pinecone_data(self) -> bool:
        """Проверка наличия данных в Pinecone"""
        try:
            if not self.pinecone_index:
                return False
            stats = self.pinecone_index.describe_index_stats()
            return stats.get('total_vector_count', 0) > 0
        except Exception:
            return False

    def create_vectorstore(self, chunks: List[Document]):
        """Создание векторного хранилища"""
        logger.info("Создание векторного хранилища...")

        # Используем улучшенные эмбеддинги
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            dimensions=1536  # Уменьшенная размерность для экономии
        )

        if self.pinecone_available:
            return self._create_pinecone_vectorstore(chunks, embeddings)
        else:
            return self._create_faiss_vectorstore(chunks, embeddings)

    def _create_pinecone_vectorstore(self, chunks: List[Document], embeddings):
        """Создание Pinecone векторного хранилища"""
        try:
            # Создаем батчами для стабильности
            batch_size = 50
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]

                if i == 0:
                    Pinecone.from_documents(
                        batch, embeddings, index_name=self.index_name
                    )
                else:
                    Pinecone.from_documents(
                        batch, embeddings, index_name=self.index_name
                    )

                logger.info(f"Обработан батч {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")

            return Pinecone(
                index=self.pinecone_index,
                embedding=embeddings,
                text_key="text"
            )

        except Exception as e:
            logger.error(f"Ошибка создания Pinecone: {e}")
            return self._create_faiss_vectorstore(chunks, embeddings)

    def _create_faiss_vectorstore(self, chunks: List[Document], embeddings):
        """Создание FAISS векторного хранилища"""
        try:
            batch_size = 100
            vectorstore = None

            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]

                if vectorstore is None:
                    vectorstore = FAISS.from_documents(batch, embeddings)
                else:
                    batch_vs = FAISS.from_documents(batch, embeddings)
                    vectorstore.merge_from(batch_vs)

                logger.info(f"Обработан батч {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")

            return vectorstore

        except Exception as e:
            logger.error(f"Ошибка создания FAISS: {e}")
            raise

    def _load_pinecone_vectorstore(self):
        """Загрузка существующего Pinecone векторного хранилища"""
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            dimensions=1536
        )

        return Pinecone(
            index=self.pinecone_index,
            embedding=embeddings,
            text_key="text"
        )
        
    def save_to_cache(self, data: Dict[str, Any]):
        """Сохранение в кэш"""
        cache_file = os.path.join(self.cache_dir, "processed_data.pkl")
        embeddings_cache = os.path.join(self.cache_dir, "embeddings.pkl")
        
        try:
            # Мы не пиклим vectorsore для Pinecone
            vectorstore = data.pop('vectorstore', None)
            
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)

            # Сохраняем FAISS если не используем Pinecone
            if vectorstore and not self.pinecone_available:
                with open(embeddings_cache, 'wb') as f:
                    pickle.dump(vectorstore, f)
            
            # Возвращаем vectorstore обратно в словарь (на случай если он нужен вызывающему)
            if vectorstore:
                data['vectorstore'] = vectorstore

            logger.info("Данные сохранены в кэш")

        except Exception as e:
            logger.warning(f"Не удалось сохранить кэш: {e}")

    def clear_cache(self):
        """Очистка кэша"""
        cache_files = [
            os.path.join(self.cache_dir, "processed_data.pkl"),
            os.path.join(self.cache_dir, "embeddings.pkl")
        ]

        for cache_file in cache_files:
            if os.path.exists(cache_file):
                os.remove(cache_file)

        if self.pinecone_available:
            try:
                self.pinecone_index.delete(delete_all=True)
                logger.info("Pinecone индекс очищен")
            except Exception as e:
                logger.warning(f"Не удалось очистить Pinecone: {e}")

