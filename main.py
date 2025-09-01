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
from typing import List, Tuple, Dict
from sklearn.metrics.pairwise import cosine_similarity

# Константы безопасности
MAX_QUESTION_LENGTH = 200

# Настройка логирования
logging.basicConfig(
    level=logging.DEBUG,
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

class BookRAG:
    def __init__(self, use_sections=True, pdf_directory="book", use_multi_vector=False):
        try:
            logger.debug(f"Начало инициализации RAG с использованием разделов: {use_sections}, multi-vector: {use_multi_vector}")
            self.use_sections = use_sections
            self.pdf_directory = pdf_directory
            self.use_multi_vector = use_multi_vector
            self.multi_vector_embeddings = None
            self.sections_mapping = self._get_sections_mapping()

            # Инициализация Pinecone
            self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
            self.pinecone_environment = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")
            self.index_name = "book-rag-index"

            # Проверка Pinecone
            pinecone_available = False
            if self.pinecone_api_key:
                pinecone_available = self._init_pinecone()

            # Загрузка существующих эмбеддингов или создание новых
            if pinecone_available and self._has_existing_embeddings():
                logger.info("Найдены существующие эмбеддинги в Pinecone. Пропускаем загрузку PDF.")
                self.documents = []
                self.splits = []
                self.vectorstore = self._load_existing_vectorstore()
                # Для гибридного поиска нужно загрузить документы для BM25
                logger.info("Загружаем документы для BM25...")
                if use_sections:
                    self.documents = asyncio.run(self._load_all_sections())
                else:
                    self.documents = asyncio.run(self._load_full_book())
                self.splits = self._create_hierarchical_chunks()
            else:
                logger.info("Эмбеддинги не найдены. Загружаем PDF и создаем эмбеддинги.")
                if use_sections:
                    self.documents = asyncio.run(self._load_all_sections())
                else:
                    self.documents = asyncio.run(self._load_full_book())
                self.splits = self._create_hierarchical_chunks()
                self.vectorstore = self._create_or_load_vectorstore()

            # Инициализация моделей
            self._initialize_models()
            self.chat_history = []
            logger.info("Инициализация RAG завершена успешно")

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

    def _init_pinecone(self):
        """Инициализация Pinecone"""
        try:
            self.pc = PineconeClient(api_key=self.pinecone_api_key)
            indexes = [index.name for index in self.pc.list_indexes()]
            embedding_dimension = 3072  # Для text-embedding-3-large

            if self.index_name not in indexes:
                logger.info(f"Создание нового индекса Pinecone: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=embedding_dimension,
                    metric="cosine",
                    spec={"serverless": {"cloud": "aws", "region": "us-east-1"}}
                )
            else:
                logger.info(f"Индекс {self.index_name} уже существует")

            self.pinecone_index = self.pc.Index(self.index_name)
            logger.info("Pinecone успешно инициализирован")
            return True
        except Exception as e:
            logger.error(f"Ошибка инициализации Pinecone: {str(e)}")
            self.pinecone_index = None
            return False

    def _has_existing_embeddings(self):
        """Проверяет наличие существующих эмбеддингов в Pinecone"""
        if not hasattr(self, 'pinecone_index') or not self.pinecone_index:
            return False
        try:
            stats = self.pinecone_index.describe_index_stats()
            vector_count = stats.get('total_vector_count', 0)
            logger.info(f"Найдено векторов в Pinecone: {vector_count}")
            return vector_count > 0
        except Exception as e:
            logger.error(f"Ошибка проверки существующих эмбеддингов: {str(e)}")
            return False

    def _load_existing_vectorstore(self):
        """Загружает существующее векторное хранилище из Pinecone"""
        logger.info("Загрузка существующего векторного хранилища из Pinecone")
        class CaseInsensitiveEmbeddings(OpenAIEmbeddings):
            def __init__(self):
                super().__init__(model="text-embedding-3-large")
            def embed_query(self, text: str) -> list:
                return super().embed_query(text.lower())

        embeddings = CaseInsensitiveEmbeddings()
        vectorstore = Pinecone(
            index=self.pinecone_index,
            embedding=embeddings,
            text_key="text"
        )
        logger.info("Существующее векторное хранилище загружено")
        return vectorstore

    async def _load_and_clean_pdf_async(self, pdf_path, section_names):
        """Асинхронная загрузка и очистка PDF с PyMuPDF"""
        try:
            async with aiofiles.open(pdf_path, mode='rb') as f:
                doc = fitz.open(stream=await f.read(), filetype="pdf")
                documents = []
                primary_section = section_names[0] if isinstance(section_names, list) else section_names
                all_sections = section_names if isinstance(section_names, list) else [section_names]

                for page_num in range(doc.page_count):
                    page = doc.load_page(page_num)
                    text = page.get_text("text")
                    text = re.sub(r'\s+', ' ', text)
                    text = re.sub(r'[^\w\s.,!?:;()\[\]"-]', '', text)
                    text = re.sub(r'-\s+', '', text)
                    text = text.replace('«', '"').replace('»', '"')
                    normalized_text = text.lower()

                    for section_name in all_sections:
                        doc_metadata = {
                            "cleaned": True,
                            "length": len(text),
                            "normalized_text": normalized_text,
                            "section": section_name,
                            "primary_section": primary_section,
                            "source_file": pdf_path,
                            "all_sections": all_sections,
                            "page": page_num + 1
                        }
                        documents.append(Document(page_content=text, metadata=doc_metadata))

                logger.info(f"Загружено {len(documents)} документов из {pdf_path}")
                return documents
        except Exception as e:
            logger.error(f"Ошибка загрузки {pdf_path}: {str(e)}")
            return []

    async def _load_full_book(self):
        """Асинхронная загрузка полной книги"""
        book_path = "book/book.pdf"
        if not os.path.exists(book_path):
            raise FileNotFoundError(f"Файл {book_path} не найден")
        return await self._load_and_clean_pdf_async(book_path, ["book"])

    async def _load_all_sections(self):
        """Асинхронная загрузка всех разделов книги"""
        file_to_sections = {}
        for section_name, file_path in self.sections_mapping.items():
            if file_path not in file_to_sections:
                file_to_sections[file_path] = []
            file_to_sections[file_path].append(section_name)

        tasks = [self._load_and_clean_pdf_async(path, sections)
                 for path, sections in file_to_sections.items()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        all_documents = [doc for sublist in results if isinstance(sublist, list) for doc in sublist]

        logger.info(f"Загружено разделов: {len(set([doc.metadata['section'] for doc in all_documents]))}")
        logger.info(f"Общее количество страниц: {len(all_documents)}")
        return all_documents

    def _create_hierarchical_chunks(self):
        """Создание улучшенных семантических чанков с multiple размерами для преодоления ограничений single-vector"""
        logger.debug("Создание улучшенных семантических чанков...")

        # Попытка загрузить русскую модель spaCy с fallback
        try:
            nlp = spacy.load("ru_core_news_lg")
            logger.info("Загружена русская модель spaCy ru_core_news_lg")
        except OSError:
            logger.warning("Русская модель spaCy не найдена, попытка загрузить малую модель...")
            try:
                nlp = spacy.load("ru_core_news_sm")
                logger.info("Загружена малая русская модель spaCy ru_core_news_sm")
            except OSError:
                logger.warning("Русские модели spaCy недоступны, используем базовую обработку текста")
                nlp = None

        # Создаем чанки разных размеров для лучшего покрытия
        chunk_configs = [
            {"size": 600, "overlap": 150, "type": "small"},    # Для детальных вопросов
            {"size": 1000, "overlap": 250, "type": "medium"},  # Базовые чанки
            {"size": 1500, "overlap": 400, "type": "large"},   # Для контекстных вопросов
        ]

        all_chunks = []

        for config in chunk_configs:
            logger.debug(f"Создание чанков размера {config['size']} с перекрытием {config['overlap']}")

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=config["size"],
                chunk_overlap=config["overlap"],
                length_function=len,
                separators=["\n\n\n", "\n\n", "Глава ", "Раздел ", "Часть ", ". ", "? ", "! ", ": ", "; ", "\n"]
            )

            chunks = []
            for doc in self.documents:
                # Если spaCy доступна, используем её для разделения на предложения
                if nlp is not None:
                    try:
                        spacy_doc = nlp(doc.page_content)
                        sentences = [sent.text for sent in spacy_doc.sents]
                    except Exception as e:
                        logger.warning(f"Ошибка spaCy: {e}, используем простое разделение")
                        sentences = self._simple_sentence_split(doc.page_content)
                else:
                    # Простое разделение на предложения без spaCy
                    sentences = self._simple_sentence_split(doc.page_content)

                current_chunk = ""
                current_length = 0
                chunk_index = 0

                for sent in sentences:
                    sent_len = len(sent)
                    if current_length + sent_len > config["size"]:
                        if current_chunk.strip():  # Проверяем что чанк не пустой
                            chunk_metadata = doc.metadata.copy()
                            chunk_metadata.update({
                                "chunk_index": chunk_index,
                                "chunk_size": current_length,
                                "chunk_type": self._determine_chunk_type(chunk_index, len(sentences)),
                                "chunk_size_category": config["type"],
                                "chunk_config_size": config["size"]
                            })
                            chunks.append(Document(page_content=current_chunk.strip(), metadata=chunk_metadata))

                        # Создаем перекрытие с предыдущим чанком
                        overlap_text = self._create_overlap(current_chunk, config["overlap"])
                        current_chunk = overlap_text + " " + sent
                        current_length = len(current_chunk)
                        chunk_index += 1
                    else:
                        current_chunk += " " + sent if current_chunk else sent
                        current_length += sent_len

                # Добавляем последний чанк
                if current_chunk.strip():
                    chunk_metadata = doc.metadata.copy()
                    chunk_metadata.update({
                        "chunk_index": chunk_index,
                        "chunk_size": current_length,
                        "chunk_type": self._determine_chunk_type(chunk_index, chunk_index + 1),
                        "chunk_size_category": config["type"],
                        "chunk_config_size": config["size"]
                    })
                    chunks.append(Document(page_content=current_chunk.strip(), metadata=chunk_metadata))

            all_chunks.extend(chunks)
            logger.debug(f"Создано {len(chunks)} чанков размера {config['type']}")

        # Удаляем дубликаты на основе содержимого и метаданных
        unique_chunks = self._deduplicate_chunks(all_chunks)

        logger.info(f"Создано {len(unique_chunks)} уникальных семантических чанков ({len(all_chunks)} до дедупликации)")
        return unique_chunks

    def _determine_chunk_type(self, chunk_index: int, total_chunks: int) -> str:
        """Определяет тип чанка на основе позиции"""
        if chunk_index == 0:
            return "start"
        elif chunk_index == total_chunks - 1:
            return "end"
        else:
            return "middle"

    def _create_overlap(self, text: str, overlap_size: int) -> str:
        """Создает перекрытие заданного размера с конца текста"""
        if len(text) <= overlap_size:
            return text

        # Находим хорошую точку разрыва (конец предложения)
        sentences = text.split('. ')
        if len(sentences) > 1:
            # Берем последние предложения, которые помещаются в overlap_size
            overlap_text = ""
            for sentence in reversed(sentences[-3:]):  # Берем максимум 3 последних предложения
                candidate = sentence + ". " + overlap_text
                if len(candidate) <= overlap_size:
                    overlap_text = candidate
                else:
                    break
            return overlap_text.strip()
        else:
            # Если предложений мало, берем последние overlap_size символов
            return text[-overlap_size:].strip()

    def _deduplicate_chunks(self, chunks: List[Document]) -> List[Document]:
        """Удаляет дублирующиеся чанки на основе содержимого"""
        seen_content = set()
        unique_chunks = []

        for chunk in chunks:
            # Создаем ключ на основе содержимого и страницы
            content_key = (
                chunk.page_content[:100],  # Первые 100 символов для быстрого сравнения
                chunk.metadata.get('page', 0),
                chunk.metadata.get('section', ''),
                chunk.metadata.get('chunk_size_category', '')
            )

            if content_key not in seen_content:
                seen_content.add(content_key)
                unique_chunks.append(chunk)

        return unique_chunks

    def _simple_sentence_split(self, text):
        """Простое разделение текста на предложения без spaCy"""
        # Разделяем по основным знакам препинания
        import re
        sentences = re.split(r'[.!?]+\s+', text)
        # Фильтруем пустые строки и очень короткие предложения
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        return sentences

    def _create_or_load_vectorstore(self):
        """Создание или загрузка векторного хранилища"""
        logger.debug("Создание/загрузка векторного хранилища...")
        class CaseInsensitiveEmbeddings(OpenAIEmbeddings):
            def __init__(self):
                super().__init__(model="text-embedding-3-large")
            def embed_query(self, text: str) -> list:
                return super().embed_query(text.lower())

        embeddings = CaseInsensitiveEmbeddings()
        processed_docs = [doc for doc in self.splits if 'normalized_text' in doc.metadata or setattr(doc.metadata, 'normalized_text', doc.page_content.lower())]

        if hasattr(self, 'pinecone_index') and self.pinecone_index:
            try:
                stats = self.pinecone_index.describe_index_stats()
                if stats['total_vector_count'] > 0:
                    logger.info("Загрузка существующих эмбеддингов из Pinecone")
                    vectorstore = Pinecone(
                        index=self.pinecone_index,
                        embedding=embeddings,
                        text_key="text"
                    )
                else:
                    logger.info("Создание новых эмбеддингов в Pinecone")
                    vectorstore = self._create_pinecone_vectorstore_with_batching(processed_docs, embeddings)
                logger.info("Pinecone векторное хранилище готово")
                return vectorstore
            except Exception as e:
                logger.error(f"Ошибка работы с Pinecone: {str(e)}")
                logger.info("Переключение на FAISS...")

        logger.info("Создание FAISS векторного хранилища")
        vectorstore = self._create_faiss_vectorstore_with_batching(processed_docs, embeddings)
        logger.info("FAISS векторное хранилище создано")
        return vectorstore

    def _create_faiss_vectorstore_with_batching(self, docs, embeddings, batch_size=50):
        """Создание FAISS векторного хранилища с батчингом"""
        logger.info(f"Создание FAISS с батчингом. Размер батча: {batch_size}")
        vectorstore = None
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            logger.debug(f"Обработка батча {i//batch_size + 1}/{(len(docs) + batch_size - 1)//batch_size}")
            try:
                if vectorstore is None:
                    vectorstore = FAISS.from_documents(batch, embeddings)
                else:
                    batch_vectorstore = FAISS.from_documents(batch, embeddings)
                    vectorstore.merge_from(batch_vectorstore)
            except Exception as e:
                logger.error(f"Ошибка обработки батча {i//batch_size + 1}: {str(e)}")
                if batch_size > 10:
                    logger.info(f"Уменьшение размера батча до {batch_size//2}")
                    return self._create_faiss_vectorstore_with_batching(docs, embeddings, batch_size//2)
                else:
                    raise
        return vectorstore

    def _create_pinecone_vectorstore_with_batching(self, docs, embeddings, batch_size=50):
        """Создание Pinecone векторного хранилища с батчингом"""
        logger.info(f"Создание Pinecone с батчингом. Размер батча: {batch_size}")
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            logger.debug(f"Обработка батча {i//batch_size + 1}/{(len(docs) + batch_size - 1)//batch_size}")
            try:
                if i == 0:
                    vectorstore = Pinecone.from_documents(
                        batch,
                        embeddings,
                        index_name=self.index_name
                    )
                else:
                    Pinecone.from_documents(
                        batch,
                        embeddings,
                        index_name=self.index_name
                    )
            except Exception as e:
                logger.error(f"Ошибка обработки батча {i//batch_size + 1}: {str(e)}")
                if batch_size > 10:
                    logger.info(f"Уменьшение размера батча до {batch_size//2}")
                    return self._create_pinecone_vectorstore_with_batching(docs, embeddings, batch_size//2)
                else:
                    raise
        return Pinecone(
            index=self.pinecone_index,
            embedding=embeddings,
            text_key="text"
        )

    def _initialize_models(self):
        """Инициализация моделей"""
        logger.debug("Инициализация моделей...")
        self.llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
        self.light_llm = ChatOpenAI(model_name="gpt-4o", temperature=0.2)

        # Инициализация multi-vector подхода если включен
        if self.use_multi_vector:
            logger.info("Инициализация multi-vector подхода...")
            base_embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
            self.multi_vector_embeddings = MultiVectorEmbeddings(base_embeddings, num_vectors=3)

            # Создаем multi-vector эмбеддинги для всех документов
            if self.splits:
                texts = [doc.page_content for doc in self.splits]
                logger.info(f"Создание multi-vector эмбеддингов для {len(texts)} документов...")
                self.multi_vector_embeddings.embed_documents(texts)
                logger.info("Multi-vector эмбеддинги созданы")

        self.qa_chain = ConversationalRetrievalChain.from_llm(
            self.llm,
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5}),
            return_source_documents=True
        )
        logger.info("Модели инициализированы успешно")

    def _split_complex_question(self, question: str):
        """Разбиение сложных вопросов на подзапросы"""
        # Упрощаем - для большинства вопросов достаточно одного запроса
        # Это устранит проблемы с парсингом JSON от LLM
        try:
            # Простая проверка - если вопрос содержит "и", "а также", "плюс" - можем разделить
            if any(word in question.lower() for word in [" и ", " а также ", " плюс ", " кроме того "]):
                # Простое разделение по союзам
                parts = []
                for separator in [" и ", " а также ", " плюс ", " кроме того "]:
                    if separator in question.lower():
                        parts = question.split(separator, 1)
                        break

                if len(parts) == 2:
                    return [part.strip() for part in parts]

            # В остальных случаях возвращаем исходный вопрос
            return [question]
        except Exception as e:
            logger.warning(f"Ошибка при разделении вопроса: {e}")
            return [question]

    def ask_question(self, question: str, section_filter=None):
        """Обработка вопроса пользователя"""
        try:
            if len(question) > MAX_QUESTION_LENGTH:
                return f"Ошибка: Вопрос слишком длинный (максимум {MAX_QUESTION_LENGTH} символов)."
            if not question.strip():
                return "Ошибка: Вопрос не может быть пустым."

            suspicious_patterns = [
                "ignore previous", "ignore above", "forget everything",
                "system prompt", "system message", "you are now",
                "act as", "pretend to be", "roleplay as"
            ]
            question_lower = question.lower()
            for pattern in suspicious_patterns:
                if pattern in question_lower:
                    return "Ошибка: Обнаружен подозрительный запрос. Задайте вопрос о книге."

            logger.debug(f"Обработка вопроса: {question}")
            subquestions = self._split_complex_question(question)

            # Если один вопрос - обрабатываем напрямую
            if len(subquestions) == 1:
                docs = self._hybrid_search_with_filter(question.lower(), section_filter)
                if not docs:
                    return "Информация по данному вопросу в книге отсутствует."

                context = self._create_context_window(docs)
                logger.debug(f"Контекст для вопроса '{question}': {context[:500]}..." if len(context) > 500 else f"Контекст: {context}")
                prompt = self._create_enhanced_prompt(question, context, section_filter)
                result = self.qa_chain({"question": prompt, "chat_history": self.chat_history})
                self.chat_history.append((question, result["answer"]))
                return self._post_process_answer(result["answer"], docs)

            # Если несколько подвопросов
            else:
                answers = []
                relevant_docs = []

                for subq in subquestions:
                    docs = self._hybrid_search_with_filter(subq.lower(), section_filter)
                    if not docs:
                        answers.append(f"Информация по подзапросу '{subq}' отсутствует.")
                        continue
                    relevant_docs.extend(docs)
                    context = self._create_context_window(docs)
                    logger.debug(f"Контекст для подвопроса '{subq}': {context[:300]}..." if len(context) > 300 else f"Контекст: {context}")
                    prompt = self._create_enhanced_prompt(subq, context, section_filter)
                    result = self.qa_chain({"question": prompt, "chat_history": self.chat_history})
                    answers.append(result["answer"])
                    self.chat_history.append((subq, result["answer"]))

                final_answer = "\n".join([f"Часть {i+1}: {ans}" for i, ans in enumerate(answers)])
                return self._post_process_answer(final_answer, relevant_docs)

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Ошибка при обработке вопроса '{question}': {error_msg}", exc_info=True)

            # Более конкретные сообщения об ошибках
            if "openai" in error_msg.lower():
                return f"Ошибка OpenAI API: {error_msg}. Проверьте API ключ."
            elif "pinecone" in error_msg.lower():
                return f"Ошибка Pinecone: {error_msg}. Проверьте настройки Pinecone."
            elif "spacy" in error_msg.lower():
                return f"Ошибка spaCy: {error_msg}. Установите модель командой: python -m spacy download ru_core_news_sm"
            elif "splits" in error_msg.lower() or "list index" in error_msg.lower():
                return f"Ошибка данных: {error_msg}. Попробуйте пересоздать эмбеддинги."
            else:
                return f"Произошла ошибка: {error_msg}. Попробуйте переформулировать вопрос."

    def _hybrid_search_with_filter(self, question: str, section_filter=None):
        """Улучшенный гибридный поиск с учетом ограничений single-vector подхода"""
        try:
            # Увеличиваем количество документов для лучшего покрытия
            k_vector = 15
            k_bm25 = 20
            k_multi_vector = 12
            k_final = 15

            vector_docs = []
            multi_vector_docs = []

            # Обычный векторный поиск
            if section_filter:
                filter_dict = {"section": {"$eq": section_filter}}
                vector_docs = [(doc, 1.0) for doc in self.vectorstore.similarity_search(
                    question, k=k_vector, filter=filter_dict
                )]
            else:
                vector_docs = self.vectorstore.similarity_search_with_score(question, k=k_vector)

            # Multi-vector поиск если включен
            if self.use_multi_vector and self.multi_vector_embeddings:
                logger.debug("Выполняется multi-vector поиск...")
                # Фильтруем документы по разделу если нужно
                filtered_docs = self.splits
                if section_filter:
                    filtered_docs = [doc for doc in self.splits if doc.metadata.get('section') == section_filter]

                multi_vector_results = self.multi_vector_embeddings.similarity_search(
                    question, filtered_docs, k=k_multi_vector
                )
                multi_vector_docs = multi_vector_results

            # Проверяем, что self.splits не пустой
            if not self.splits:
                logger.warning("self.splits пустой, используем только векторный поиск")
                return [doc for doc, _ in vector_docs[:k_final]]

            # Улучшенная токенизация для русского языка
            tokenized_corpus = [self._advanced_tokenize(doc.page_content.lower()) for doc in self.splits]

            # Настройка BM25 для русского языка
            bm25 = BM25Okapi(tokenized_corpus, k1=1.5, b=0.75)
            tokenized_query = self._advanced_tokenize(question.lower())
            bm25_scores = bm25.get_scores(tokenized_query)

            keyword_docs = []
            # Сортируем по BM25 скорам и берем топ результаты
            scored_indices = [(idx, score) for idx, score in enumerate(bm25_scores) if score > 0]
            scored_indices.sort(key=lambda x: x[1], reverse=True)

            for idx, score in scored_indices[:k_bm25]:
                if not section_filter or self.splits[idx].metadata.get('section') == section_filter:
                    # Увеличиваем бонус для разных типов чанков
                    chunk_type_multiplier = {
                        'start': 1.3, 'end': 1.2, 'middle': 1.0
                    }.get(self.splits[idx].metadata.get('chunk_type', 'middle'), 1.0)

                    # Бонус для точного совпадения ключевых слов
                    exact_match_bonus = 1.0
                    doc_lower = self.splits[idx].page_content.lower()
                    query_words = tokenized_query
                    exact_matches = sum(1 for word in query_words if word in doc_lower)
                    if exact_matches > 0:
                        exact_match_bonus = 1.0 + (exact_matches / len(query_words)) * 0.5

                    final_score = score * chunk_type_multiplier * exact_match_bonus
                    keyword_docs.append((self.splits[idx], final_score))

            # Вычисляем адаптивные веса на основе характеристик запроса
            # Это критично для преодоления фундаментальных ограничений single-vector подхода
            adaptive_weights = self._calculate_adaptive_weights(question, vector_docs, keyword_docs, multi_vector_docs)

            unique_docs = {}

            # 1. Векторный поиск с адаптивным весом
            for doc, score in vector_docs:
                key = (doc.page_content, doc.metadata.get('page'), doc.metadata.get('section'))
                unique_docs[key] = (doc, score * adaptive_weights['vector'])

            # 2. Multi-vector поиск с адаптивным весом
            if self.use_multi_vector and multi_vector_docs:
                logger.debug(f"Добавляем {len(multi_vector_docs)} multi-vector результатов")
                for doc, score in multi_vector_docs:
                    key = (doc.page_content, doc.metadata.get('page'), doc.metadata.get('section'))
                    if key in unique_docs:
                        unique_docs[key] = (doc, unique_docs[key][1] + score * adaptive_weights['multi_vector'])
                    else:
                        unique_docs[key] = (doc, score * adaptive_weights['multi_vector'])

            # 3. BM25 с адаптивным весом (обычно наибольший для sparse search)
            for doc, score in keyword_docs:
                key = (doc.page_content, doc.metadata.get('page'), doc.metadata.get('section'))
                if key in unique_docs:
                    # Если документ найден несколькими методами - комбинируем скоры
                    unique_docs[key] = (doc, unique_docs[key][1] + score * adaptive_weights['bm25'])
                else:
                    unique_docs[key] = (doc, score * adaptive_weights['bm25'])

            # Дополнительное ранжирование по релевантности
            sorted_docs = sorted(unique_docs.values(), key=lambda x: x[1], reverse=True)

            # Применяем дополнительную фильтрацию по минимальному скору
            min_threshold = 0.1
            filtered_docs = [(doc, score) for doc, score in sorted_docs if score >= min_threshold]

            return [doc for doc, _ in filtered_docs[:k_final]]

        except Exception as e:
            logger.error(f"Ошибка в гибридном поиске: {str(e)}", exc_info=True)
            return [doc for doc, _ in vector_docs[:k_vector]] if vector_docs else []

    def _advanced_tokenize(self, text):
        """Улучшенная токенизация для русского языка"""
        # Удаляем знаки препинания и разделяем по пробелам
        import re
        # Сохраняем важные знаки препинания как отдельные токены
        text = re.sub(r'([.!?,:;])', r' \1 ', text)
        tokens = text.split()

        # Фильтруем слишком короткие токены и стоп-слова
        stop_words = {'и', 'в', 'на', 'с', 'по', 'для', 'от', 'до', 'при', 'о', 'об', 'что', 'как', 'так', 'это', 'то', 'же', 'но', 'или', 'а', 'да', 'нет'}
        filtered_tokens = [token for token in tokens if len(token) > 2 and token.lower() not in stop_words]

        return filtered_tokens

    def _calculate_adaptive_weights(self, question: str, vector_docs: List, keyword_docs: List, multi_vector_docs: List = None) -> Dict[str, float]:
        """Вычисляет адаптивные веса на основе характеристик запроса и результатов поиска"""

        # Анализ характеристик запроса
        question_lower = question.lower()
        question_words = self._advanced_tokenize(question_lower)

        # 1. Определяем тип запроса
        exact_match_indicators = ['точно', 'именно', 'конкретно', 'определенно', 'четко']
        conceptual_indicators = ['как', 'почему', 'зачем', 'что такое', 'объясни', 'расскажи']

        is_exact_query = any(indicator in question_lower for indicator in exact_match_indicators)
        is_conceptual_query = any(indicator in question_lower for indicator in conceptual_indicators)

        # 2. Анализируем длину запроса (короткие запросы лучше для BM25)
        query_length_factor = min(len(question_words) / 10.0, 1.0)  # Нормализуем к [0, 1]

        # 3. Анализируем пересечение результатов
        vector_content = set()
        keyword_content = set()
        multi_vector_content = set()

        for doc, _ in vector_docs:
            vector_content.add(doc.page_content[:100])  # Первые 100 символов как идентификатор

        for doc, _ in keyword_docs:
            keyword_content.add(doc.page_content[:100])

        if multi_vector_docs:
            for doc, _ in multi_vector_docs:
                multi_vector_content.add(doc.page_content[:100])

        # Вычисляем пересечения
        vector_keyword_overlap = len(vector_content & keyword_content) / max(len(vector_content), 1)

        # 4. Вычисляем адаптивные веса
        base_weights = {
            'vector': 0.3,
            'multi_vector': 0.4 if self.use_multi_vector else 0.0,
            'bm25': 0.5 if self.use_multi_vector else 0.6
        }

        # Корректировки весов
        if is_exact_query:
            # Для точных запросов увеличиваем вес BM25
            base_weights['bm25'] *= 1.3
            base_weights['vector'] *= 0.8
        elif is_conceptual_query:
            # Для концептуальных запросов увеличиваем вес vector/multi-vector
            base_weights['vector'] *= 1.2
            if self.use_multi_vector:
                base_weights['multi_vector'] *= 1.2
            base_weights['bm25'] *= 0.9

        # Корректировка на основе длины запроса
        if query_length_factor < 0.3:  # Короткий запрос
            base_weights['bm25'] *= 1.2
        elif query_length_factor > 0.7:  # Длинный запрос
            base_weights['vector'] *= 1.1
            if self.use_multi_vector:
                base_weights['multi_vector'] *= 1.1

        # Корректировка на основе пересечения результатов
        if vector_keyword_overlap > 0.7:  # Высокое пересечение - можем доверять обоим
            base_weights['vector'] *= 1.1
            base_weights['bm25'] *= 1.1
        elif vector_keyword_overlap < 0.3:  # Низкое пересечение - предпочитаем sparse
            base_weights['bm25'] *= 1.2
            base_weights['vector'] *= 0.9

        # Нормализация весов
        total_weight = sum(base_weights.values())
        normalized_weights = {k: v / total_weight for k, v in base_weights.items()}

        logger.debug(f"Адаптивные веса для запроса '{question[:50]}...': {normalized_weights}")

        return normalized_weights

    def _create_context_window(self, docs):
        """Создание контекстного окна с расширенным контекстом"""
        context_parts = []
        processed_pages = set()

        for doc in docs:
            page_num = doc.metadata.get('page')
            section = doc.metadata.get('section', '')

            # Добавляем контекст с текущей страницы
            if page_num not in processed_pages:
                page_content = [d.page_content for d in self.splits
                               if d.metadata.get('page') == page_num]
                if page_content:
                    context_parts.extend(page_content)
                    processed_pages.add(page_num)

            # Добавляем сам документ если он еще не включен
            if doc.page_content not in context_parts:
                context_parts.append(doc.page_content)

        # Ограничиваем общий размер контекста
        full_context = "\n\n".join(context_parts)
        if len(full_context) > 4000:  # Ограничение для токенов
            # Берем первые части, чтобы не превысить лимит
            truncated_parts = []
            current_length = 0
            for part in context_parts:
                if current_length + len(part) <= 4000:
                    truncated_parts.append(part)
                    current_length += len(part)
                else:
                    break
            return "\n\n".join(truncated_parts)

        return full_context

    def _create_enhanced_prompt(self, question: str, context: str, section_filter=None):
        """Создание улучшенного промпта"""
        section_info = f"\nРаздел: {section_filter}" if section_filter else ""
        prompt_template = """Ты - эксперт по книге "Бизнес в диалоге: от малого к невозможному" Анвара Халикова.

ВАЖНО: Автор книги - Анвар Халиков. Если спрашивают об авторе, упомяни это.

ИНСТРУКЦИИ:
1. Внимательно изучи предоставленный контекст
2. Найди информацию, которая отвечает на вопрос, даже если она выражена косвенно
3. Если информация есть в контексте, используй её для ответа
4. Если прямого ответа нет, но есть связанная информация - используй её
5. Только если контекст совсем не содержит релевантной информации, скажи: "Информация по этому вопросу в книге отсутствует"

Отвечай структурированно и полно на основе найденной информации.

Контекст:
{context}

{section_info}

Вопрос: {question}

Ответ:"""
        return PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question", "section_info"]
        ).format(context=context, question=question.strip(), section_info=section_info)

    def _post_process_answer(self, answer: str, source_docs: list):
        """Постобработка ответа"""
        sources = []
        sections = set()
        for doc in source_docs:
            page = doc.metadata.get('page', 0)
            section = doc.metadata.get('section', 'unknown')
            sources.append(f"Страница {page}")
            sections.add(section)

        unique_sources = sorted(set(sources))
        formatted_answer = answer.strip()
        if not formatted_answer.startswith("Ответ:"):
            formatted_answer = "Ответ: " + formatted_answer

        sections_info = ""
        if len(sections) > 1:
            sections_info = f"\nРазделы: {', '.join(sorted(sections))}"
        elif len(sections) == 1:
            section = list(sections)[0]
            if section != 'unknown':
                sections_info = f"\nРаздел: {section}"

        return f"{formatted_answer}\n\nИсточники: {', '.join(unique_sources)}{sections_info}"

    def search_by_section(self, section_name: str, query: str = ""):
        """Поиск информации в конкретном разделе"""
        try:
            if len(query) > MAX_QUESTION_LENGTH:
                return f"Ошибка: Запрос слишком длинный (максимум {MAX_QUESTION_LENGTH} символов)."
            section_key = section_name.lower().replace(' ', '_')
            if section_key not in self.sections_mapping:
                available_sections = ', '.join(self.sections_mapping.keys())
                return f"Раздел '{section_name}' не найден. Доступные разделы: {available_sections}"
            if not query:
                query = f"Расскажи о содержании раздела {section_name}"
            return self.ask_question(query, section_filter=section_key)
        except Exception as e:
            logger.error(f"Ошибка при поиске в разделе {section_name}: {str(e)}")
            return f"Произошла ошибка: {str(e)}"

    def get_available_sections(self):
        """Получение списка доступных разделов"""
        return list(self.sections_mapping.keys())

    def force_rebuild_embeddings(self):
        """Принудительное пересоздание эмбеддингов"""
        logger.info("Принудительное пересоздание эмбеддингов...")
        try:
            if hasattr(self, 'pinecone_index') and self.pinecone_index:
                self.pinecone_index.delete(delete_all=True)
                logger.info("Pinecone индекс очищен")
            if self.use_sections:
                self.documents = asyncio.run(self._load_all_sections())
            else:
                self.documents = asyncio.run(self._load_full_book())
            self.splits = self._create_hierarchical_chunks()
            self.vectorstore = self._create_or_load_vectorstore()
            self._initialize_models()
            logger.info("Эмбеддинги пересозданы")
            return True
        except Exception as e:
            logger.error(f"Ошибка при пересоздании эмбеддингов: {str(e)}")
            return False

    def run_tests(self):
        """Запуск тестов для проверки качества"""
        test_questions = [
            {"question": "О чем глава 5?", "section": "встреча5", "expected": "командообразование"},
            {"question": "Кто автор книги?", "section": None, "expected": "Анвар Халиков"}
        ]
        results = []
        for test in test_questions:
            answer = self.ask_question(test["question"], test["section"])
            passed = test["expected"].lower() in answer.lower()
            results.append({"question": test["question"], "passed": passed, "answer": answer})
            logger.info(f"Тест: {test['question']} - {'Пройден' if passed else 'Не пройден'}")
        return results


class MultiVectorEmbeddings:
    """
    Класс для реализации multi-vector подхода, преодолевающего ограничения single-vector
    Создает несколько векторов для каждого документа для улучшения покрытия семантического пространства
    """

    def __init__(self, base_embeddings: OpenAIEmbeddings, num_vectors: int = 3):
        self.base_embeddings = base_embeddings
        self.num_vectors = num_vectors
        self.doc_embeddings = {}  # {doc_id: [vector1, vector2, ...]}
        self.doc_texts = {}       # {doc_id: text}

    def embed_documents(self, texts: List[str]) -> List[List[List[float]]]:
        """Создает multiple векторы для каждого документа"""
        all_multi_embeddings = []

        for doc_idx, text in enumerate(texts):
            doc_vectors = self._create_multiple_vectors(text, doc_idx)
            all_multi_embeddings.append(doc_vectors)

        return all_multi_embeddings

    def _create_multiple_vectors(self, text: str, doc_id: int) -> List[List[float]]:
        """Создает несколько векторов для одного документа разными способами"""
        vectors = []

        # 1. Базовый эмбеддинг всего текста
        base_vector = self.base_embeddings.embed_query(text)
        vectors.append(base_vector)

        # 2. Эмбеддинг первой половины
        mid_point = len(text) // 2
        first_half = text[:mid_point]
        if len(first_half.strip()) > 50:  # Только если достаточно текста
            first_vector = self.base_embeddings.embed_query(first_half)
            vectors.append(first_vector)
        else:
            vectors.append(base_vector)  # Дублируем базовый если мало текста

        # 3. Эмбеддинг второй половины
        second_half = text[mid_point:]
        if len(second_half.strip()) > 50:
            second_vector = self.base_embeddings.embed_query(second_half)
            vectors.append(second_vector)
        else:
            vectors.append(base_vector)

        # 4. Если нужно больше векторов - создаем их через ключевые предложения
        if self.num_vectors > 3:
            key_sentences = self._extract_key_sentences(text)
            for i, sentence in enumerate(key_sentences[:self.num_vectors-3]):
                if len(sentence.strip()) > 30:
                    sent_vector = self.base_embeddings.embed_query(sentence)
                    vectors.append(sent_vector)
                else:
                    vectors.append(base_vector)

        # Сохраняем информацию о документе
        self.doc_embeddings[doc_id] = vectors
        self.doc_texts[doc_id] = text

        return vectors[:self.num_vectors]

    def _extract_key_sentences(self, text: str) -> List[str]:
        """Извлекает ключевые предложения из текста"""
        import re

        # Простое разделение на предложения
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 30]

        # Берем самые длинные предложения как наиболее информативные
        sentences.sort(key=len, reverse=True)

        return sentences[:5]  # Максимум 5 ключевых предложений

    def similarity_search(self, query: str, documents: List[Document], k: int = 10) -> List[Tuple[Document, float]]:
        """Поиск с использованием multi-vector подхода"""
        query_vector = self.base_embeddings.embed_query(query)

        doc_scores = []

        for doc_idx, doc in enumerate(documents):
            if doc_idx in self.doc_embeddings:
                # Вычисляем similarity с каждым вектором документа
                doc_vectors = self.doc_embeddings[doc_idx]
                similarities = []

                for vector in doc_vectors:
                    # Используем cosine similarity
                    sim = cosine_similarity([query_vector], [vector])[0][0]
                    similarities.append(sim)

                # Используем максимальную similarity (ColBERT-style)
                max_sim = max(similarities)

                # Альтернативно: средняя similarity
                # avg_sim = np.mean(similarities)

                # Можно также использовать взвешенную сумму
                # weights = [0.5, 0.3, 0.2]  # Больший вес базовому вектору
                # weighted_sim = sum(w * s for w, s in zip(weights, similarities))

                doc_scores.append((doc, max_sim))
            else:
                # Fallback на обычный single vector для документов без multi-vector
                single_vector = self.base_embeddings.embed_query(doc.page_content)
                sim = cosine_similarity([query_vector], [single_vector])[0][0]
                doc_scores.append((doc, sim))

        # Сортируем по убыванию similarity
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        return doc_scores[:k]


# Пример использования улучшенной системы
if __name__ == "__main__":
    print("=== Демонстрация улучшений для преодоления ограничений single-vector поиска ===")
    print()

    # Пример 1: Обычная система (только single-vector)
    print("1. Инициализация стандартной системы (single-vector):")
    try:
        rag_standard = BookRAG(use_sections=True, use_multi_vector=False)
        print("✓ Стандартная система инициализирована")
    except Exception as e:
        print(f"✗ Ошибка: {e}")

    print()

    # Пример 2: Улучшенная система с multi-vector подходом
    print("2. Инициализация улучшенной системы (multi-vector + enhanced BM25):")
    try:
        rag_enhanced = BookRAG(use_sections=True, use_multi_vector=True)
        print("✓ Улучшенная система инициализирована")
        print("  - Multi-vector подход активирован")
        print("  - Улучшенный BM25 с русской токенизацией")
        print("  - Чанки множественных размеров")
        print("  - Адаптивные веса для dense/sparse fusion")
    except Exception as e:
        print(f"✗ Ошибка: {e}")

    print()
    print("=== Ключевые улучшения согласно статье Google ===")
    print()
    print("🔍 ПРОБЛЕМА: Single-vector поиск имеет фундаментальные ограничения")
    print("   При фиксированной размерности невозможно найти все релевантные документы")
    print()
    print("🚀 РЕШЕНИЯ РЕАЛИЗОВАНЫ:")
    print()
    print("1. ГИБРИДНЫЙ ПОИСК С УВЕЛИЧЕННЫМ ВЕСОМ BM25")
    print("   ✓ BM25 (sparse) получает вес 0.5-0.6 вместо 0.3")
    print("   ✓ Векторный поиск (dense) получает вес 0.3-0.4")
    print("   ✓ Преодолевает ограничения single-vector подхода")
    print()
    print("2. MULTI-VECTOR ПОДХОД (аналог ColBERT)")
    print("   ✓ 3 вектора на документ вместо 1")
    print("   ✓ Базовый вектор + векторы частей + ключевые предложения")
    print("   ✓ MaxSim aggregation для финального скора")
    print()
    print("3. УЛУЧШЕННАЯ ТОКЕНИЗАЦИЯ BM25")
    print("   ✓ Настроенные параметры k1=1.5, b=0.75 для русского")
    print("   ✓ Фильтрация стоп-слов")
    print("   ✓ Обработка точных совпадений")
    print()
    print("4. МНОЖЕСТВЕННЫЕ РАЗМЕРЫ ЧАНКОВ")
    print("   ✓ Малые (600), средние (1000), большие (1500) чанки")
    print("   ✓ Умное перекрытие с сохранением границ предложений")
    print("   ✓ Дедупликация для оптимизации")
    print()
    print("5. АДАПТИВНЫЕ ВЕСА")
    print("   ✓ Анализ типа запроса (точный vs концептуальный)")
    print("   ✓ Учет длины запроса")
    print("   ✓ Динамическая корректировка весов")
    print()
    print("📊 ОЖИДАЕМЫЕ РЕЗУЛЬТАТЫ:")
    print("   • Значительное улучшение recall")
    print("   • Лучше находит релевантные документы")
    print("   • Устойчивость к росту базы знаний")
    print("   • Преодоление теоретических ограничений single-vector")
    print()
    print("🔬 ТЕОРЕТИЧЕСКОЕ ОБОСНОВАНИЕ:")
    print("   Google доказали: sign-rank(A) > d => single-vector не работает")
    print("   Наши решения обходят это ограничение через:")
    print("   - Sparse поиск (BM25) - не ограничен размерностью")
    print("   - Multi-vector - увеличивает effective размерность")
    print("   - Адаптивное слияние - оптимально использует оба подхода")