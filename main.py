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
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

class BookRAG:
    def __init__(self, use_sections=True, pdf_directory="book"):
        try:
            logger.debug(f"Начало инициализации RAG с использованием разделов: {use_sections}")
            self.use_sections = use_sections
            self.pdf_directory = pdf_directory
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
        """Создание семантических чанков с использованием spacy"""
        logger.debug("Создание семантических чанков...")
        nlp = spacy.load("ru_core_news_lg")  # Модель для русского языка

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n\n", "\n\n", "Глава ", "Раздел ", "Часть ", ". ", "? ", "! ", ": ", "; ", "\n"]
        )

        chunks = []
        for doc in self.documents:
            spacy_doc = nlp(doc.page_content)
            sentences = [sent.text for sent in spacy_doc.sents]
            current_chunk = ""
            current_length = 0
            chunk_index = 0

            for sent in sentences:
                sent_len = len(sent)
                if current_length + sent_len > 800:
                    chunk_metadata = doc.metadata.copy()
                    chunk_metadata.update({
                        "chunk_index": chunk_index,
                        "total_chunks": len(sentences),
                        "chunk_size": current_length,
                        "chunk_type": "middle" if chunk_index > 0 else "start"
                    })
                    chunks.append(Document(page_content=current_chunk.strip(), metadata=chunk_metadata))
                    current_chunk = sent
                    current_length = sent_len
                    chunk_index += 1
                else:
                    current_chunk += " " + sent
                    current_length += sent_len

            if current_chunk:
                chunk_metadata = doc.metadata.copy()
                chunk_metadata.update({
                    "chunk_index": chunk_index,
                    "total_chunks": chunk_index + 1,
                    "chunk_size": current_length,
                    "chunk_type": "end" if chunk_index > 0 else "start"
                })
                chunks.append(Document(page_content=current_chunk.strip(), metadata=chunk_metadata))

        logger.info(f"Создано {len(chunks)} семантических чанков")
        return chunks

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
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            self.llm,
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5}),
            return_source_documents=True
        )
        logger.info("Модели инициализированы успешно")

    def _split_complex_question(self, question: str):
        """Разбиение сложных вопросов на подзапросы"""
        split_prompt = """Разбей сложный вопрос на простые подзапросы.
        Вопрос: {question}
        Ответ в формате JSON: { "subquestions": [...] }"""
        chain = LLMChain(
            llm=self.light_llm,
            prompt=PromptTemplate(template=split_prompt, input_variables=["question"])
        )
        result = chain.run(question=question)
        try:
            return json.loads(result)["subquestions"]
        except:
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
            answers = []
            relevant_docs = []

            for subq in subquestions:
                docs = self._hybrid_search_with_filter(subq.lower(), section_filter)
                if not docs:
                    answers.append(f"Информация по подзапросу '{subq}' отсутствует.")
                    continue
                relevant_docs.extend(docs)
                context = self._create_context_window(docs)
                prompt = self._create_enhanced_prompt(subq, context, section_filter)
                result = self.qa_chain({"question": prompt, "chat_history": self.chat_history})
                answers.append(result["answer"])
                self.chat_history.append((subq, result["answer"]))

            final_answer = "\n".join([f"Подзапрос {i+1}: {ans}" for i, ans in enumerate(answers)])
            return self._post_process_answer(final_answer, relevant_docs)

        except Exception as e:
            logger.error(f"Ошибка при обработке вопроса '{question}': {str(e)}", exc_info=True)
            return f"Произошла ошибка: Не удалось обработать вопрос. Попробуйте переформулировать."

    def _hybrid_search_with_filter(self, question: str, section_filter=None):
        """Гибридный поиск с BM25"""
        try:
            k_vector = 8
            k_final = 10
            vector_docs = []
            if section_filter:
                filter_dict = {"section": {"$eq": section_filter}}
                vector_docs = [(doc, 1.0) for doc in self.vectorstore.similarity_search(
                    question, k=k_vector, filter=filter_dict
                )]
            else:
                vector_docs = self.vectorstore.similarity_search_with_score(question, k=k_vector)

            tokenized_corpus = [doc.page_content.lower().split() for doc in self.splits]
            bm25 = BM25Okapi(tokenized_corpus)
            tokenized_query = question.split()
            bm25_scores = bm25.get_scores(tokenized_query)

            keyword_docs = []
            for idx, score in enumerate(bm25_scores):
                if score > 0 and (not section_filter or self.splits[idx].metadata.get('section') == section_filter):
                    chunk_type_multiplier = {
                        'start': 1.2, 'end': 1.1, 'middle': 1.0
                    }.get(self.splits[idx].metadata.get('chunk_type', 'middle'), 1.0)
                    keyword_docs.append((self.splits[idx], score * chunk_type_multiplier))

            unique_docs = {}
            for doc, score in vector_docs:
                key = (doc.page_content, doc.metadata.get('page'), doc.metadata.get('section'))
                unique_docs[key] = (doc, score * 0.7)
            for doc, score in keyword_docs:
                key = (doc.page_content, doc.metadata.get('page'), doc.metadata.get('section'))
                if key in unique_docs:
                    unique_docs[key] = (doc, unique_docs[key][1] + score * 0.3)
                else:
                    unique_docs[key] = (doc, score * 0.3)

            sorted_docs = sorted(unique_docs.values(), key=lambda x: x[1], reverse=True)
            return [doc for doc, _ in sorted_docs[:k_final]]

        except Exception as e:
            logger.error(f"Ошибка в гибридном поиске: {str(e)}", exc_info=True)
            return [doc for doc, _ in vector_docs[:k_vector]] if vector_docs else []

    def _create_context_window(self, docs):
        """Создание контекстного окна"""
        context_parts = []
        for doc in docs:
            page_num = doc.metadata.get('page')
            context_parts.extend([d.page_content for d in self.splits
                                if d.metadata.get('page') == page_num])
        return "\n\n".join(set(context_parts))

    def _create_enhanced_prompt(self, question: str, context: str, section_filter=None):
        """Создание улучшенного промпта"""
        section_info = f"\nРаздел: {section_filter}" if section_filter else ""
        prompt_template = """Ты - эксперт по книге "Бизнес в диалоге: от малого к невозможному" Анвара Халикова.
Твоя задача - отвечать на вопросы, опираясь ТОЛЬКО на предоставленный контекст из книги.
Если контекст не содержит ответа, скажи: "Информация по этому вопросу в книге отсутствует".
Используй точные цитаты с указанием раздела и страницы, если возможно.
Отвечай кратко, структурированно и на русском языке.

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