import asyncio
import aiofiles
import fitz  # PyMuPDF
import os
import logging
import re
import glob
import json
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.vectorstores import Pinecone
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.prompts import PromptTemplate
from pinecone import Pinecone as PineconeClient
from rank_bm25 import BM25Okapi
import spacy
import numpy as np
from typing import List, Tuple, Dict, Optional
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import pickle
import hashlib
import time

# Константы безопасности
MAX_QUESTION_LENGTH = 200

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,  # Изменили с DEBUG на INFO для производительности
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag.log', mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
if openai_key:
    os.environ["OPENAI_API_KEY"] = openai_key


class ImprovedBookRAG:
    def __init__(self, use_sections=True, pdf_directory="book", cache_embeddings=True):
        try:
            logger.info(f"Инициализация улучшенной RAG системы с кэшированием: {cache_embeddings}")
            self.use_sections = use_sections
            self.pdf_directory = pdf_directory
            self.cache_embeddings = cache_embeddings
            self.sections_mapping = self._get_sections_mapping()

            # Кэширование для производительности
            self.cache_dir = "cache"
            os.makedirs(self.cache_dir, exist_ok=True)

            # Инициализация Pinecone (опционально)
            self.pinecone_available = self._init_pinecone()

            # Загрузка или создание данных
            self._load_or_create_data()

            # Инициализация моделей
            self._initialize_models()

            # История чата
            self.chat_history = []

            logger.info("Улучшенная RAG система инициализирована успешно")

        except Exception as e:
            logger.error(f"Критическая ошибка при инициализации RAG: {str(e)}", exc_info=True)
            raise

    def _get_sections_mapping(self):
        """Маппинг разделов книги к файлам"""
        return {
            "book": "book/book.pdf",
            "часть1": "book/chast_1_put.pdf",
            "часть_1": "book/chast_1_put.pdf",
            "chast_1_put": "book/chast_1_put.pdf",
            "путь": "book/chast_1_put.pdf",
            "встреча1": "book/chast_2_vstrecha_1.pdf",
            "встреча_1": "book/chast_2_vstrecha_1.pdf",
            "chast_2_vstrecha_1": "book/chast_2_vstrecha_1.pdf",
            "с_чего_начинается_бизнес": "book/chast_2_vstrecha_1.pdf",
            "встреча2": "book/chast_2_vstrecha_2.pdf",
            "встреча_2": "book/chast_2_vstrecha_2.pdf",
            "chast_2_vstrecha_2": "book/chast_2_vstrecha_2.pdf",
            "фундамент_бизнеса": "book/chast_2_vstrecha_2.pdf",
            "встреча3": "book/chast_2_vstrecha_3.pdf",
            "встреча_3": "book/chast_2_vstrecha_3.pdf",
            "chast_2_vstrecha_3": "book/chast_2_vstrecha_3.pdf",
            "выбор_ниши": "book/chast_2_vstrecha_3.pdf",
            "встреча4": "book/chast_2_vstrecha_4.pdf",
            "встреча_4": "book/chast_2_vstrecha_4.pdf",
            "chast_2_vstrecha_4": "book/chast_2_vstrecha_4.pdf",
            "ключевой_фактор_успеха": "book/chast_2_vstrecha_4.pdf",
            "встреча5": "book/chast_2_vstrecha_5.pdf",
            "встреча_5": "book/chast_2_vstrecha_5.pdf",
            "chast_2_vstrecha_5": "book/chast_2_vstrecha_5.pdf",
            "командообразование": "book/chast_2_vstrecha_5.pdf",
            "встреча6": "book/chast_2_vstrecha_6.pdf",
            "встреча_6": "book/chast_2_vstrecha_6.pdf",
            "chast_2_vstrecha_6": "book/chast_2_vstrecha_6.pdf",
            "кризис": "book/chast_2_vstrecha_6.pdf",
            "встреча7": "book/chast_2_vstrecha_7.pdf",
            "встреча_7": "book/chast_2_vstrecha_7.pdf",
            "chast_2_vstrecha_7": "book/chast_2_vstrecha_7.pdf",
            "оргструктура": "book/chast_2_vstrecha_7.pdf",
            "часть3": "book/chast3.pdf",
            "часть_3": "book/chast3.pdf",
            "chast3": "book/chast3.pdf",
            "новые_начинания": "book/chast3.pdf"
        }

    def _init_pinecone(self) -> bool:
        """Инициализация Pinecone с улучшенной обработкой ошибок"""
        try:
            self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
            if not self.pinecone_api_key:
                logger.info("Pinecone API ключ не найден, используем локальный FAISS")
                return False

            self.pinecone_environment = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")
            self.index_name = "book-rag-index-v2"  # Новая версия индекса

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

    def _load_or_create_data(self):
        """Загрузка или создание данных с кэшированием"""
        cache_file = os.path.join(self.cache_dir, "processed_data.pkl")
        embeddings_cache = os.path.join(self.cache_dir, "embeddings.pkl")

        # Проверяем кэш
        if (self.cache_embeddings and
            os.path.exists(cache_file) and
            self._is_cache_valid(cache_file)):

            logger.info("Загрузка данных из кэша...")
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    self.documents = cached_data['documents']
                    self.chunks = cached_data['chunks']
                    self.bm25_index = cached_data['bm25_index']
                    self.metadata_index = cached_data['metadata_index']

                # Загружаем векторное хранилище
                if self.pinecone_available and self._has_pinecone_data():
                    self.vectorstore = self._load_pinecone_vectorstore()
                elif os.path.exists(embeddings_cache):
                    logger.info("Загрузка FAISS из кэша...")
                    with open(embeddings_cache, 'rb') as f:
                        self.vectorstore = pickle.load(f)
                else:
                    self.vectorstore = self._create_vectorstore()

                logger.info("Данные успешно загружены из кэша")
                return

            except Exception as e:
                logger.warning(f"Ошибка загрузки кэша: {e}. Пересоздаем данные...")

        # Создаем данные с нуля
        logger.info("Создание данных с нуля...")
        self._create_fresh_data()

        # Сохраняем в кэш
        if self.cache_embeddings:
            self._save_to_cache(cache_file, embeddings_cache)

    def _is_cache_valid(self, cache_file: str) -> bool:
        """Проверка валидности кэша"""
        try:
            cache_time = os.path.getmtime(cache_file)

            # Проверяем время изменения PDF файлов
            for file_path in self.sections_mapping.values():
                if os.path.exists(file_path):
                    file_time = os.path.getmtime(file_path)
                    if file_time > cache_time:
                        return False

            return True
        except Exception:
            return False

    def _has_pinecone_data(self) -> bool:
        """Проверка наличия данных в Pinecone"""
        try:
            stats = self.pinecone_index.describe_index_stats()
            return stats.get('total_vector_count', 0) > 0
        except Exception:
            return False

    def _create_fresh_data(self):
        """Создание данных с нуля"""
        # Загрузка документов
        if self.use_sections:
            self.documents = asyncio.run(self._load_all_sections())
        else:
            self.documents = asyncio.run(self._load_full_book())

        # Создание улучшенных чанков
        self.chunks = self._create_smart_chunks()

        # Создание индексов
        self._create_search_indexes()

        # Создание векторного хранилища
        self.vectorstore = self._create_vectorstore()

    def _save_to_cache(self, cache_file: str, embeddings_cache: str):
        """Сохранение в кэш"""
        try:
            cache_data = {
                'documents': self.documents,
                'chunks': self.chunks,
                'bm25_index': self.bm25_index,
                'metadata_index': self.metadata_index
            }

            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)

            # Сохраняем FAISS если не используем Pinecone
            if not self.pinecone_available:
                with open(embeddings_cache, 'wb') as f:
                    pickle.dump(self.vectorstore, f)

            logger.info("Данные сохранены в кэш")

        except Exception as e:
            logger.warning(f"Не удалось сохранить кэш: {e}")

    async def _load_all_sections(self) -> List[Document]:
        """Улучшенная загрузка всех разделов"""
        file_to_sections = defaultdict(list)

        for section_name, file_path in self.sections_mapping.items():
            file_to_sections[file_path].append(section_name)

        tasks = []
        for path, sections in file_to_sections.items():
            if os.path.exists(path):
                tasks.append(self._load_and_clean_pdf_async(path, sections))
            else:
                logger.warning(f"Файл не найден: {path}")

        if not tasks:
            raise FileNotFoundError("Не найдено ни одного PDF файла")

        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_documents = []
        for result in results:
            if isinstance(result, list):
                all_documents.extend(result)
            else:
                logger.error(f"Ошибка при загрузке: {result}")

        logger.info(f"Загружено {len(all_documents)} страниц из {len(tasks)} файлов")
        return all_documents

    async def _load_full_book(self) -> List[Document]:
        """Загрузка полной книги"""
        book_path = "book/book.pdf"
        if not os.path.exists(book_path):
            raise FileNotFoundError(f"Файл {book_path} не найден")
        return await self._load_and_clean_pdf_async(book_path, ["book"])

    async def _load_and_clean_pdf_async(self, pdf_path: str, section_names: List[str]) -> List[Document]:
        """Улучшенная асинхронная загрузка PDF"""
        try:
            async with aiofiles.open(pdf_path, mode='rb') as f:
                pdf_data = await f.read()

            doc = fitz.open(stream=pdf_data, filetype="pdf")
            documents = []

            primary_section = section_names[0]

            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                text = page.get_text("text")

                # Улучшенная очистка текста
                cleaned_text = self._advanced_text_cleaning(text)

                if len(cleaned_text.strip()) < 50:  # Пропускаем слишком короткие страницы
                    continue

                # Создаем документ для каждой секции
                for section_name in section_names:
                    metadata = {
                        "source_file": pdf_path,
                        "page": page_num + 1,
                        "section": section_name,
                        "primary_section": primary_section,
                        "all_sections": section_names,
                        "char_count": len(cleaned_text),
                        "word_count": len(cleaned_text.split()),
                        "file_hash": hashlib.md5(pdf_data[:1000]).hexdigest()  # Для версионирования
                    }

                    documents.append(Document(
                        page_content=cleaned_text,
                        metadata=metadata
                    ))

            doc.close()
            logger.info(f"Обработано {len(documents)} документов из {pdf_path}")
            return documents

        except Exception as e:
            logger.error(f"Ошибка загрузки {pdf_path}: {str(e)}")
            return []

    def _advanced_text_cleaning(self, text: str) -> str:
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

    def _create_smart_chunks(self) -> List[Document]:
        """Создание умных чанков с учетом семантики"""
        logger.info("Создание умных чанков...")

        # Попытка загрузить русскую модель spaCy
        nlp = None
        try:
            nlp = spacy.load("ru_core_news_lg")
        except OSError:
            try:
                nlp = spacy.load("ru_core_news_sm")
            except OSError:
                logger.warning("spaCy модели не найдены, используем простое разделение")

        all_chunks = []

        # Конфигурации чанков для разных типов поиска
        chunk_configs = [
            {"size": 800, "overlap": 100, "type": "standard", "weight": 1.0},
            {"size": 1200, "overlap": 200, "type": "extended", "weight": 0.8},
            {"size": 400, "overlap": 50, "type": "precise", "weight": 1.2}
        ]

        for doc in self.documents:
            text = doc.page_content

            # Разделение на предложения
            if nlp:
                try:
                    spacy_doc = nlp(text)
                    sentences = [sent.text.strip() for sent in spacy_doc.sents]
                except Exception:
                    sentences = self._simple_sentence_split(text)
            else:
                sentences = self._simple_sentence_split(text)

            # Создание чанков разных размеров
            for config in chunk_configs:
                chunks = self._create_chunks_with_config(
                    sentences, doc, config
                )
                all_chunks.extend(chunks)

        # Удаление дубликатов и ранжирование
        unique_chunks = self._deduplicate_and_rank_chunks(all_chunks)

        logger.info(f"Создано {len(unique_chunks)} умных чанков")
        return unique_chunks

    def _create_chunks_with_config(self, sentences: List[str], doc: Document, config: dict) -> List[Document]:
        """Создание чанков с определенной конфигурацией"""
        chunks = []
        current_chunk = ""
        current_length = 0
        chunk_index = 0

        for sent in sentences:
            sent_len = len(sent)

            # Проверяем, нужно ли создать новый чанк
            if current_length + sent_len > config["size"] and current_chunk:
                # Создаем чанк
                chunk_doc = self._create_chunk_document(
                    current_chunk.strip(), doc, chunk_index, config
                )
                chunks.append(chunk_doc)

                # Создаем перекрытие
                overlap_text = self._create_smart_overlap(current_chunk, config["overlap"])
                current_chunk = overlap_text + " " + sent if overlap_text else sent
                current_length = len(current_chunk)
                chunk_index += 1
            else:
                current_chunk += (" " + sent if current_chunk else sent)
                current_length += sent_len

        # Добавляем последний чанк
        if current_chunk.strip():
            chunk_doc = self._create_chunk_document(
                current_chunk.strip(), doc, chunk_index, config
            )
            chunks.append(chunk_doc)

        return chunks

    def _create_chunk_document(self, text: str, original_doc: Document,
                             chunk_index: int, config: dict) -> Document:
        """Создание документа чанка с метаданными"""
        metadata = original_doc.metadata.copy()
        metadata.update({
            "chunk_index": chunk_index,
            "chunk_type": config["type"],
            "chunk_size": len(text),
            "chunk_weight": config["weight"],
            "word_count": len(text.split()),
            "has_numbers": bool(re.search(r'\d+', text)),
            "has_questions": '?' in text,
            "sentence_count": len(self._simple_sentence_split(text))
        })

        return Document(page_content=text, metadata=metadata)

    def _create_smart_overlap(self, text: str, overlap_size: int) -> str:
        """Создание умного перекрытия по границам предложений"""
        if len(text) <= overlap_size:
            return text

        sentences = self._simple_sentence_split(text)

        # Берем последние предложения, которые помещаются в overlap
        overlap_text = ""
        for sentence in reversed(sentences):
            candidate = sentence + ". " + overlap_text
            if len(candidate) <= overlap_size:
                overlap_text = candidate
            else:
                break

        return overlap_text.strip()

    def _simple_sentence_split(self, text: str) -> List[str]:
        """Простое разделение на предложения"""
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    def _deduplicate_and_rank_chunks(self, chunks: List[Document]) -> List[Document]:
        """Дедупликация и ранжирование чанков"""
        # Группируем по содержимому (полный текст для точности)
        content_groups = defaultdict(list)
        for chunk in chunks:
            # Используем хеш содержимого для ключа
            key = hashlib.md5(chunk.page_content.encode()).hexdigest()
            content_groups[key].append(chunk)

        # Выбираем лучший чанк из каждой группы
        unique_chunks = []
        for group in content_groups.values():
            if len(group) == 1:
                unique_chunks.append(group[0])
            else:
                # Выбираем чанк с наибольшим весом
                # Если веса равны, предпочитаем тот, что НЕ из секции "book" (более специфичный)
                best_chunk = max(group, key=lambda x: (
                    x.metadata.get('chunk_weight', 1.0),
                    0 if x.metadata.get('section') == 'book' else 1
                ))
                unique_chunks.append(best_chunk)

        logger.info(f"Дедупликация: {len(chunks)} -> {len(unique_chunks)}")
        return unique_chunks

    def _create_search_indexes(self):
        """Создание поисковых индексов"""
        logger.info("Создание поисковых индексов...")

        # Подготовка текстов для BM25
        texts = [chunk.page_content.lower() for chunk in self.chunks]
        tokenized_texts = [self._advanced_tokenize(text) for text in texts]

        # Создание BM25 индекса с оптимизированными параметрами для русского
        self.bm25_index = BM25Okapi(tokenized_texts, k1=1.2, b=0.75)

        # Создание метаданных индекса для быстрого поиска по фильтрам
        self.metadata_index = self._create_metadata_index()

        logger.info("Поисковые индексы созданы")

    def _create_metadata_index(self) -> Dict:
        """Создание индекса метаданных для быстрой фильтрации"""
        index = {
            'by_section': defaultdict(list),
            'by_page': defaultdict(list),
            'by_chunk_type': defaultdict(list),
            'with_numbers': [],
            'with_questions': []
        }

        for i, chunk in enumerate(self.chunks):
            metadata = chunk.metadata

            index['by_section'][metadata.get('section', 'unknown')].append(i)
            index['by_page'][metadata.get('page', 0)].append(i)
            index['by_chunk_type'][metadata.get('chunk_type', 'standard')].append(i)

            if metadata.get('has_numbers', False):
                index['with_numbers'].append(i)
            if metadata.get('has_questions', False):
                index['with_questions'].append(i)

        return index

    def _advanced_tokenize(self, text: str) -> List[str]:
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

    def _create_vectorstore(self):
        """Создание векторного хранилища"""
        logger.info("Создание векторного хранилища...")

        # Используем улучшенные эмбеддинги
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            dimensions=1536  # Уменьшенная размерность для экономии
        )

        if self.pinecone_available:
            return self._create_pinecone_vectorstore(embeddings)
        else:
            return self._create_faiss_vectorstore(embeddings)

    def _create_pinecone_vectorstore(self, embeddings):
        """Создание Pinecone векторного хранилища"""
        try:
            # Создаем батчами для стабильности
            batch_size = 50
            for i in range(0, len(self.chunks), batch_size):
                batch = self.chunks[i:i + batch_size]

                if i == 0:
                    vectorstore = Pinecone.from_documents(
                        batch, embeddings, index_name=self.index_name
                    )
                else:
                    Pinecone.from_documents(
                        batch, embeddings, index_name=self.index_name
                    )

                logger.info(f"Обработан батч {i//batch_size + 1}/{(len(self.chunks) + batch_size - 1)//batch_size}")

            return Pinecone(
                index=self.pinecone_index,
                embedding=embeddings,
                text_key="text"
            )

        except Exception as e:
            logger.error(f"Ошибка создания Pinecone: {e}")
            return self._create_faiss_vectorstore(embeddings)

    def _create_faiss_vectorstore(self, embeddings):
        """Создание FAISS векторного хранилища"""
        try:
            batch_size = 100
            vectorstore = None

            for i in range(0, len(self.chunks), batch_size):
                batch = self.chunks[i:i + batch_size]

                if vectorstore is None:
                    vectorstore = FAISS.from_documents(batch, embeddings)
                else:
                    batch_vs = FAISS.from_documents(batch, embeddings)
                    vectorstore.merge_from(batch_vs)

                logger.info(f"Обработан батч {i//batch_size + 1}/{(len(self.chunks) + batch_size - 1)//batch_size}")

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

    def _initialize_models(self):
        """Инициализация языковых моделей"""
        logger.info("Инициализация языковых моделей...")

        # Основная модель для ответов
        self.llm = ChatOpenAI(
            model_name="gpt-4o-mini",  # Более экономичная модель
            temperature=0.1,
            max_tokens=2000
        )

        # Легкая модель для анализа запросов
        self.light_llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0,
            max_tokens=500
        )

        # Создание цепочки QA с улучшенным ретривером
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            self.llm,
            retriever=self._create_enhanced_retriever(),
            return_source_documents=True,
            verbose=False
        )

        logger.info("Языковые модели инициализированы")

    def _create_enhanced_retriever(self):
        """Создание улучшенного ретривера"""
        base_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 15}  # Увеличенное количество для лучшего покрытия
        )

        return base_retriever

    def ask_question(self, question: str, section_filter: Optional[str] = None) -> str:
        """Основной метод для ответа на вопросы"""
        try:
            # Валидация входных данных
            if not question or not question.strip():
                return "Ошибка: Вопрос не может быть пустым."

            if len(question) > MAX_QUESTION_LENGTH:
                return f"Ошибка: Вопрос слишком длинный (максимум {MAX_QUESTION_LENGTH} символов)."

            # Проверка на подозрительные запросы
            if self._is_suspicious_query(question):
                return "Ошибка: Обнаружен подозрительный запрос. Задайте вопрос о книге."

            logger.info(f"Обработка вопроса: {question[:100]}...")

            # Анализ и улучшение запроса
            enhanced_query = self._enhance_query(question)

            # Гибридный поиск релевантных документов
            relevant_docs = self._advanced_hybrid_search(enhanced_query, section_filter)

            if not relevant_docs:
                return self._handle_no_results(question, section_filter)

            # Создание контекста и ответа
            context = self._create_optimized_context(relevant_docs)
            prompt = self._create_smart_prompt(question, context, section_filter)

            # Получение ответа от модели
            # Используем прямой вызов LLM вместо qa_chain, чтобы использовать наш оптимизированный контекст
            # qa_chain делает свой собственный поиск (retrieval), который может игнорировать наши результаты hybrid search
            from langchain.schema import HumanMessage, SystemMessage
            
            messages = [
                SystemMessage(content="Ты полезный ассистент, который отвечает на вопросы по книге."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            answer_text = response.content

            # Постобработка ответа
            final_answer = self._postprocess_answer(answer_text, relevant_docs)

            # Сохранение в историю
            self.chat_history.append((question, answer_text))

            # Ограничиваем историю для производительности
            if len(self.chat_history) > 10:
                self.chat_history = self.chat_history[-10:]

            return final_answer

        except Exception as e:
            logger.error(f"Ошибка при обработке вопроса: {str(e)}", exc_info=True)
            return self._handle_error(e)

    def _is_suspicious_query(self, question: str) -> bool:
        """Проверка на подозрительные запросы"""
        suspicious_patterns = [
            "ignore previous", "ignore above", "forget everything",
            "system prompt", "system message", "you are now",
            "act as", "pretend to be", "roleplay as", "jailbreak"
        ]

        question_lower = question.lower()
        return any(pattern in question_lower for pattern in suspicious_patterns)

    def _enhance_query(self, question: str) -> str:
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

    def _advanced_hybrid_search(self, query: str, section_filter: Optional[str] = None) -> List[Document]:
        """Продвинутый гибридный поиск"""
        logger.debug(f"Выполняется гибридный поиск: {query[:50]}...")

        # Параметры поиска
        k_vector = 30
        k_bm25 = 30
        k_final = 25

        try:
            # 1. Векторный поиск
            vector_docs = self._vector_search(query, section_filter, k_vector)

            # 2. BM25 поиск
            bm25_docs = self._bm25_search(query, section_filter, k_bm25)

            # 3. Объединение результатов с умным скорингом
            combined_docs = self._smart_combine_results(query, vector_docs, bm25_docs)

            # 4. Финальное ранжирование
            final_docs = self._final_ranking(query, combined_docs, k_final)

            logger.debug(f"Найдено {len(final_docs)} релевантных документов")
            return final_docs

        except Exception as e:
            logger.error(f"Ошибка в гибридном поиске: {e}")
            return []

    def _vector_search(self, query: str, section_filter: Optional[str], k: int) -> List[Tuple[Document, float]]:
        """Векторный поиск с фильтрацией"""
        try:
            if section_filter and self.pinecone_available:
                # Фильтр для Pinecone
                filter_dict = {"section": {"$eq": section_filter}}
                docs = self.vectorstore.similarity_search_with_score(
                    query, k=k, filter=filter_dict
                )
            else:
                docs = self.vectorstore.similarity_search_with_score(query, k=k)

                # Фильтрация для FAISS
                if section_filter:
                    docs = [(doc, score) for doc, score in docs
                           if doc.metadata.get('section') == section_filter]

            return docs

        except Exception as e:
            logger.warning(f"Ошибка векторного поиска: {e}")
            return []

    def _bm25_search(self, query: str, section_filter: Optional[str], k: int) -> List[Tuple[Document, float]]:
        """BM25 поиск с фильтрацией"""
        try:
            # Токенизация запроса
            tokenized_query = self._advanced_tokenize(query.lower())

            if not tokenized_query:
                return []

            # Получение скоров BM25
            scores = self.bm25_index.get_scores(tokenized_query)

            # Создание списка документов с скорами
            scored_docs = []
            for i, score in enumerate(scores):
                if score > 0 and i < len(self.chunks):
                    doc = self.chunks[i]

                    # Применяем фильтр по секции
                    if section_filter and doc.metadata.get('section') != section_filter:
                        continue

                    # Бонусы за точные совпадения
                    bonus = self._calculate_bm25_bonus(query, doc.page_content)
                    final_score = score * (1 + bonus)

                    scored_docs.append((doc, final_score))

            # Сортировка по убыванию скора
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            return scored_docs[:k]

        except Exception as e:
            logger.warning(f"Ошибка BM25 поиска: {e}")
            return []

    def _calculate_bm25_bonus(self, query: str, text: str) -> float:
        """Расчет бонуса для BM25 на основе точных совпадений"""
        query_words = set(query.lower().split())
        text_words = set(text.lower().split())

        # Точные совпадения слов
        exact_matches = len(query_words & text_words)
        exact_bonus = exact_matches / len(query_words) if query_words else 0

        # Бонус за наличие всего запроса в тексте
        phrase_bonus = 0.5 if query.lower() in text.lower() else 0

        # Бонус за тип чанка
        chunk_type_bonus = 0.1  # Базовый бонус

        return exact_bonus * 0.3 + phrase_bonus + chunk_type_bonus

    def _smart_combine_results(self, query: str, vector_docs: List[Tuple[Document, float]],
                             bm25_docs: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
        """Умное объединение результатов векторного и BM25 поиска"""

        # Вычисляем адаптивные веса на основе характеристик запроса
        weights = self._calculate_adaptive_weights(query, vector_docs, bm25_docs)

        # Объединяем результаты
        combined_scores = {}

        # Добавляем векторные результаты
        for doc, score in vector_docs:
            key = self._get_doc_key(doc)
            combined_scores[key] = {
                'doc': doc,
                'vector_score': score,
                'bm25_score': 0.0
            }

        # Добавляем BM25 результаты
        for doc, score in bm25_docs:
            key = self._get_doc_key(doc)
            if key in combined_scores:
                combined_scores[key]['bm25_score'] = score
            else:
                combined_scores[key] = {
                    'doc': doc,
                    'vector_score': 0.0,
                    'bm25_score': score
                }

        # Вычисляем финальные скоры
        final_results = []
        for item in combined_scores.values():
            # Нормализация скоров
            norm_vector = min(item['vector_score'], 1.0)
            norm_bm25 = min(item['bm25_score'] / 10.0, 1.0)  # BM25 скоры могут быть большими

            # Взвешенная комбинация
            final_score = (
                weights['vector'] * norm_vector +
                weights['bm25'] * norm_bm25
            )

            # Дополнительные бонусы
            final_score *= self._get_metadata_bonus(item['doc'])

            final_results.append((item['doc'], final_score))

        return final_results

    def _get_doc_key(self, doc: Document) -> str:
        """Создание уникального ключа документа"""
        return f"{doc.metadata.get('page', 0)}_{doc.metadata.get('section', '')}_{doc.page_content[:50]}"

    def _calculate_adaptive_weights(self, query: str, vector_docs: List, bm25_docs: List) -> Dict[str, float]:
        """Вычисление адаптивных весов для комбинирования"""

        # Анализ запроса
        query_words = query.lower().split()
        query_length = len(query_words)

        # Базовые веса - смещаем баланс в пользу векторного поиска (он лучше справляется с опечатками и смыслом)
        base_vector_weight = 0.7
        base_bm25_weight = 0.3

        # Корректировки на основе характеристик запроса

        # Длинные запросы лучше для векторного поиска
        if query_length > 6:
            base_vector_weight += 0.1
            base_bm25_weight -= 0.1

        # Короткие запросы - проверяем, есть ли имена (с большой буквы)
        # Если есть имена, лучше верить вектору, так как в именах часто опечатки
        has_capitalized = any(word[0].isupper() for word in query.split() if len(word) > 1)
        if has_capitalized:
             base_vector_weight += 0.2  # Boost vector more for names
             base_bm25_weight -= 0.2
        
        # Концептуальные вопросы лучше для векторного поиска
        conceptual_words = ['как', 'почему', 'что', 'зачем', 'объясни', 'расскажи', 'кто']
        if any(word in query.lower() for word in conceptual_words):
            base_vector_weight += 0.1
            base_bm25_weight -= 0.1

        # Точные запросы лучше для BM25, но только если мы уверены в точности терминов
        exact_words = ['точно', 'именно', 'конкретно', 'название', 'цифра']
        if any(word in query.lower() for word in exact_words):
            base_vector_weight -= 0.2
            base_bm25_weight += 0.2

        # Нормализация (защита от выхода за границы)
        base_vector_weight = max(0.1, min(0.9, base_vector_weight))
        base_bm25_weight = 1.0 - base_vector_weight
        
        return {
            'vector': base_vector_weight,
            'bm25': base_bm25_weight
        }

    def _get_metadata_bonus(self, doc: Document) -> float:
        """Расчет бонуса на основе метаданных документа"""
        bonus = 1.0
        metadata = doc.metadata

        # Бонус за тип чанка
        chunk_weight = metadata.get('chunk_weight', 1.0)
        bonus *= chunk_weight

        # Бонус за длину (оптимальная длина)
        chunk_size = metadata.get('chunk_size', 0)
        if 600 <= chunk_size <= 1200:
            bonus *= 1.1

        # Бонус за наличие вопросов (часто содержат важную информацию)
        if metadata.get('has_questions', False):
            bonus *= 1.05

        return bonus

    def _final_ranking(self, query: str, docs: List[Tuple[Document, float]], k: int) -> List[Document]:
        """Финальное ранжирование результатов"""

        # Сортировка по скору
        docs.sort(key=lambda x: x[1], reverse=True)

        # Дедупликация похожих документов
        seen_content = set()
        unique_docs = []

        for doc, score in docs:
            # Создаем отпечаток содержимого
            content_fingerprint = doc.page_content[:100].lower()

            if content_fingerprint not in seen_content:
                seen_content.add(content_fingerprint)
                unique_docs.append(doc)

            if len(unique_docs) >= k:
                break

        return unique_docs

    def _handle_no_results(self, question: str, section_filter: Optional[str]) -> str:
        """Обработка случая, когда результаты не найдены"""
        base_msg = "Информация по данному вопросу в книге не найдена."

        if section_filter:
            sections = ', '.join(self.get_available_sections())
            return f"{base_msg}\n\nВозможно, попробуйте поискать в других разделах: {sections}"
        else:
            return f"{base_msg}\n\nПопробуйте переформулировать вопрос или использовать другие ключевые слова."

    def _create_optimized_context(self, docs: List[Document]) -> str:
        """Создание оптимизированного контекста"""
        
        # 1. Отбираем документы, пока не достигнем лимита, сохраняя порядок релевантности
        selected_docs = []
        current_length = 0
        max_length = 15000
        
        for doc in docs:
            doc_len = len(doc.page_content) + 100 # +100 на метаданные
            if current_length + doc_len > max_length:
                break
            selected_docs.append(doc)
            current_length += doc_len
            
        # 2. Теперь сортируем отобранные документы по страницам для связности чтения
        # Группируем по страницам
        page_groups = defaultdict(list)
        for doc in selected_docs:
            page = doc.metadata.get('page', 0)
            page_groups[page].append(doc)

        context_parts = []
        # Добавляем контекст по страницам
        for page in sorted(page_groups.keys()):
            page_docs = page_groups[page]
            # Объединяем контент, убирая дубликаты (если чанки перекрываются)
            unique_contents = set()
            page_content_list = []
            for doc in page_docs:
                if doc.page_content not in unique_contents:
                    unique_contents.add(doc.page_content)
                    page_content_list.append(doc.page_content)
            
            page_content = ' '.join(page_content_list)

            # Добавляем информацию о странице
            section = page_docs[0].metadata.get('section', '')
            context_parts.append(f"[Страница {page}, Раздел: {section}]\n{page_content}")

        full_context = '\n\n'.join(context_parts)
        return full_context

    def _create_smart_prompt(self, question: str, context: str, section_filter: Optional[str]) -> str:
        """Создание умного промпта"""

        section_info = f"\nРАЗДЕЛ ПОИСКА: {section_filter}" if section_filter else ""

        prompt = f"""Ты - эксперт по книге "Бизнес в диалоге: от малого к невозможному" Анвара Халикова.

ПРАВИЛА ОТВЕТА:
1. Внимательно изучи предоставленный контекст.
2. Отвечай точно и по существу на основе найденной информации.
3. Если в вопросе есть имена (например, Шоира Музаффаровна), ищи похожие имена в тексте (возможны опечатки в книге или запросе, например Музафаровна).
4. Если информации мало, используй всё, что есть, даже если это просто упоминание в благодарностях или списке наставников.
5. Если прямого ответа нет, но есть связанная информация - адаптируй её.
6. Структурируй ответ логично и понятно.
7. Указывай автора книги (Анвар Халиков) при необходимости.

КОНТЕКСТ ИЗ КНИГИ:
{context}
{section_info}

ВОПРОС: {question}

ОТВЕТ:"""

        return prompt

    def _postprocess_answer(self, answer: str, source_docs: List[Document]) -> str:
        """Постобработка ответа с добавлением источников"""

        # Очистка ответа
        cleaned_answer = answer.strip()

        # Убираем дублирующиеся фразы "Ответ:"
        if cleaned_answer.startswith("Ответ:"):
            cleaned_answer = cleaned_answer[6:].strip()

        # Добавляем "Ответ:" если его нет
        if not cleaned_answer.startswith("Ответ:"):
            cleaned_answer = "Ответ: " + cleaned_answer

        # Собираем информацию об источниках
        pages = set()
        sections = set()

        for doc in source_docs:
            page = doc.metadata.get('page')
            section = doc.metadata.get('section', 'unknown')

            if page:
                pages.add(page)
            if section != 'unknown':
                sections.add(section)

        # Формируем информацию об источниках
        sources_info = []

        if pages:
            sorted_pages = sorted(pages)
            if len(sorted_pages) <= 3:
                sources_info.append(f"Страницы: {', '.join(map(str, sorted_pages))}")
            else:
                sources_info.append(f"Страницы: {sorted_pages[0]}-{sorted_pages[-1]} и др. ({len(sorted_pages)} всего)")

        if sections and len(sections) <= 2:
            sources_info.append(f"Раздел: {', '.join(sections)}")

        # Добавляем источники к ответу
        if sources_info:
            cleaned_answer += f"\n\nИсточники: {'; '.join(sources_info)}"

        return cleaned_answer

    def _handle_error(self, error: Exception) -> str:
        """Обработка ошибок с информативными сообщениями"""
        error_str = str(error).lower()

        if "openai" in error_str or "api" in error_str:
            return "Ошибка API OpenAI. Проверьте ключ API и подключение к интернету."
        elif "pinecone" in error_str:
            return "Ошибка Pinecone. Система переключится на локальный поиск."
        elif "memory" in error_str or "out of memory" in error_str:
            return "Недостаточно памяти. Попробуйте задать более конкретный вопрос."
        elif "timeout" in error_str:
            return "Превышено время ожидания. Попробуйте еще раз."
        else:
            return f"Произошла ошибка при обработке запроса. Попробуйте переформулировать вопрос."

    def search_by_section(self, section_name: str, query: str = "") -> str:
        """Поиск информации в конкретном разделе"""
        try:
            # Нормализация названия раздела
            section_key = section_name.lower().replace(' ', '_').replace('ё', 'е')

            if section_key not in self.sections_mapping:
                available = ', '.join(sorted(self.sections_mapping.keys()))
                return f"Раздел '{section_name}' не найден.\n\nДоступные разделы:\n{available}"

            # Если запрос не указан, возвращаем общую информацию о разделе
            if not query:
                query = f"Расскажи о содержании и основных темах раздела {section_name}"

            return self.ask_question(query, section_filter=section_key)

        except Exception as e:
            logger.error(f"Ошибка поиска в разделе {section_name}: {e}")
            return f"Произошла ошибка при поиске в разделе {section_name}"

    def get_available_sections(self) -> List[str]:
        """Получение списка доступных разделов"""
        # Возвращаем только основные разделы для удобства
        main_sections = []
        seen = set()

        for section in self.sections_mapping.keys():
            # Группируем похожие названия
            base_name = section.replace('_', ' ')
            if base_name not in seen:
                main_sections.append(section)
                seen.add(base_name)

        return sorted(main_sections)

    def get_statistics(self) -> Dict:
        """Получение статистики о загруженных данных"""
        if not hasattr(self, 'chunks'):
            return {"error": "Данные не загружены"}

        stats = {
            "total_chunks": len(self.chunks),
            "total_documents": len(self.documents) if hasattr(self, 'documents') else 0,
            "sections": len(set(doc.metadata.get('section', '') for doc in self.documents)) if hasattr(self, 'documents') else 0,
            "pages": len(set(doc.metadata.get('page', 0) for doc in self.documents)) if hasattr(self, 'documents') else 0,
            "chunk_types": {}
        }

        # Статистика по типам чанков
        for chunk in self.chunks:
            chunk_type = chunk.metadata.get('chunk_type', 'unknown')
            stats["chunk_types"][chunk_type] = stats["chunk_types"].get(chunk_type, 0) + 1

        return stats

    def force_rebuild_cache(self) -> bool:
        """Принудительное пересоздание кэша"""
        try:
            logger.info("Принудительное пересоздание кэша...")

            # Очищаем кэш
            cache_files = [
                os.path.join(self.cache_dir, "processed_data.pkl"),
                os.path.join(self.cache_dir, "embeddings.pkl")
            ]

            for cache_file in cache_files:
                if os.path.exists(cache_file):
                    os.remove(cache_file)

            # Очищаем Pinecone если доступен
            if self.pinecone_available:
                try:
                    self.pinecone_index.delete(delete_all=True)
                    logger.info("Pinecone индекс очищен")
                except Exception as e:
                    logger.warning(f"Не удалось очистить Pinecone: {e}")

            # Пересоздаем данные
            self._create_fresh_data()

            # Сохраняем новый кэш
            cache_file = os.path.join(self.cache_dir, "processed_data.pkl")
            embeddings_cache = os.path.join(self.cache_dir, "embeddings.pkl")
            self._save_to_cache(cache_file, embeddings_cache)

            logger.info("Кэш успешно пересоздан")
            return True

        except Exception as e:
            logger.error(f"Ошибка при пересоздании кэша: {e}")
            return False

    def run_quality_tests(self) -> Dict:
        """Запуск тестов качества системы"""
        test_questions = [
            {
                "question": "Кто автор книги?",
                "expected_keywords": ["анвар", "халиков"],
                "section": None
            },
            {
                "question": "О чем книга Бизнес в диалоге?",
                "expected_keywords": ["бизнес", "диалог", "развитие"],
                "section": None
            },
            {
                "question": "Что такое командообразование?",
                "expected_keywords": ["команда", "сотрудник", "лидер"],
                "section": "командообразование"
            }
        ]

        results = {
            "total_tests": len(test_questions),
            "passed": 0,
            "failed": 0,
            "details": []
        }

        for i, test in enumerate(test_questions):
            try:
                answer = self.ask_question(test["question"], test["section"])
                answer_lower = answer.lower()

                # Проверяем наличие ожидаемых ключевых слов
                found_keywords = []
                for keyword in test["expected_keywords"]:
                    if keyword.lower() in answer_lower:
                        found_keywords.append(keyword)

                passed = len(found_keywords) >= len(test["expected_keywords"]) // 2

                test_result = {
                    "test_id": i + 1,
                    "question": test["question"],
                    "passed": passed,
                    "found_keywords": found_keywords,
                    "answer_length": len(answer),
                    "answer": answer[:200] + "..." if len(answer) > 200 else answer
                }

                results["details"].append(test_result)

                if passed:
                    results["passed"] += 1
                else:
                    results["failed"] += 1

            except Exception as e:
                results["failed"] += 1
                results["details"].append({
                    "test_id": i + 1,
                    "question": test["question"],
                    "passed": False,
                    "error": str(e)
                })

        results["success_rate"] = results["passed"] / results["total_tests"] * 100

        return results


# Пример использования улучшенной системы
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