from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.vectorstores import Pinecone
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import os
import logging
import re
import glob
from dotenv import load_dotenv
from pinecone import Pinecone as PineconeClient

# Константы безопасности
MAX_QUESTION_LENGTH = 200

load_dotenv()
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

# Установка API ключа OpenAI
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

            # Сначала проверяем Pinecone
            pinecone_available = False
            if self.pinecone_api_key:
                pinecone_available = self._init_pinecone()

            # Проверяем есть ли уже данные в Pinecone
            if pinecone_available and self._has_existing_embeddings():
                logger.info("Найдены существующие эмбеддинги в Pinecone. Пропускаем загрузку PDF.")
                # Создаем пустые списки для совместимости
                self.documents = []
                self.splits = []
                # Создаем векторное хранилище из существующих данных
                self.vectorstore = self._load_existing_vectorstore()
            else:
                logger.info("Эмбеддинги не найдены. Загружаем PDF и создаем эмбеддинги.")
                # Загрузка документов
                if use_sections:
                    self.documents = self._load_all_sections()
                else:
                    self.documents = self._load_full_book()

                # Создание чанков
                self.splits = self._create_hierarchical_chunks()

                # Создание векторного хранилища
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
            # Инициализация клиента Pinecone
            self.pc = PineconeClient(api_key=self.pinecone_api_key)

            # Проверяем существует ли индекс
            indexes = [index.name for index in self.pc.list_indexes()]

            # Определяем размерность эмбеддингов автоматически
            # text-embedding-3-large: 3072, text-embedding-ada-002: 1536
            embedding_dimension = 3072  # Обновлено для text-embedding-3-large

            if self.index_name not in indexes:
                logger.info(f"Создание нового индекса Pinecone: {self.index_name} с размерностью {embedding_dimension}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=embedding_dimension,
                    metric="cosine",
                    spec={"serverless": {"cloud": "aws", "region": "us-east-1"}}
                )
            else:
                logger.info(f"Индекс {self.index_name} уже существует")

            self.pinecone_index = self.pc.Index(self.index_name)

            # Проверяем статистику индекса
            stats = self.pinecone_index.describe_index_stats()
            logger.info(f"Статистика Pinecone индекса: {stats}")

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

        # Создаем класс для преобразования всех запросов к нижнему регистру
        class CaseInsensitiveEmbeddings(OpenAIEmbeddings):
            def __init__(self):
                # Используем text-embedding-3-large для лучшего качества
                super().__init__(model="text-embedding-3-large")

            def embed_query(self, text: str) -> list:
                return super().embed_query(text.lower())

        embeddings = CaseInsensitiveEmbeddings()

        vectorstore = Pinecone(
            index=self.pinecone_index,
            embedding=embeddings,
            text_key="text"
        )

        logger.info("Существующее векторное хранилище успешно загружено")
        return vectorstore

    def _load_full_book(self):
        """Загрузка полной книги"""
        book_path = "book/book.pdf"
        if not os.path.exists(book_path):
            raise FileNotFoundError(f"Файл {book_path} не найден")

        # Используем только название "book" для полной книги
        return self._load_and_clean_pdf(book_path, ["book"])

    def _load_all_sections(self):
        """Загрузка всех разделов книги"""
        all_documents = []

        # Создаем обратный маппинг: файл -> список названий разделов
        file_to_sections = {}
        for section_name, file_path in self.sections_mapping.items():
            if file_path not in file_to_sections:
                file_to_sections[file_path] = []
            file_to_sections[file_path].append(section_name)

        # Сначала пробуем загрузить основную книгу
        main_book_path = "book/book.pdf"
        if os.path.exists(main_book_path):
            try:
                logger.info("Загрузка основной книги...")
                # Для основной книги используем все связанные названия разделов
                main_sections = file_to_sections.get(main_book_path, ["book"])
                main_documents = self._load_and_clean_pdf(main_book_path, main_sections)
                all_documents.extend(main_documents)

                # Если основная книга большая, используем только её
                if len(main_documents) > 200:
                    logger.warning(f"Основная книга содержит {len(main_documents)} страниц. Используем только её для избежания лимитов API")
                    return all_documents

            except Exception as e:
                logger.error(f"Ошибка загрузки основной книги: {str(e)}")

        # Загружаем остальные PDF файлы из папки book (исключая основную книгу)
        pdf_files = [f for f in glob.glob("book/*.pdf") if not f.endswith("book.pdf")]

        # Ограничиваем количество дополнительных файлов
        if len(pdf_files) > 8:
            logger.warning(f"Найдено {len(pdf_files)} дополнительных файлов. Загружаем только первые 8 для избежания лимитов")
            pdf_files = pdf_files[:8]

        for pdf_path in pdf_files:
            logger.debug(f"Загрузка файла: {pdf_path}")

            try:
                # Получаем все названия разделов для этого файла
                section_names = file_to_sections.get(pdf_path, [os.path.basename(pdf_path).replace('.pdf', '')])
                logger.debug(f"Разделы для файла {pdf_path}: {section_names}")

                documents = self._load_and_clean_pdf(pdf_path, section_names)
                all_documents.extend(documents)

                # Проверяем общий размер
                if len(all_documents) > 500:
                    logger.warning(f"Достигнут лимит документов ({len(all_documents)}). Прекращаем загрузку дополнительных разделов")
                    break

            except Exception as e:
                logger.error(f"Ошибка загрузки {pdf_path}: {str(e)}")
                continue

        logger.info(f"Загружено разделов: {len(set([doc.metadata['section'] for doc in all_documents]))}")
        logger.info(f"Общее количество страниц: {len(all_documents)}")
        return all_documents

    def _load_and_clean_pdf(self, pdf_path, section_names):
        """Улучшенная загрузка и очистка PDF с нормализацией текста"""
        logger.debug(f"Загрузка и очистка PDF файла: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        # Если section_names это список, берем первый как основной
        if isinstance(section_names, list):
            primary_section = section_names[0] if section_names else "unknown"
            all_sections = section_names
        else:
            primary_section = section_names
            all_sections = [section_names]

        # Очистка и обогащение текста
        cleaned_documents = []
        for doc in documents:
            text = doc.page_content

            # Более глубокая очистка и нормализация текста
            # 1. Удаление лишних пробелов
            text = re.sub(r'\s+', ' ', text)
            # 2. Удаление спецсимволов, но сохранение пунктуации
            text = re.sub(r'[^\w\s.,!?:;()\[\]"-]', '', text)
            # 3. Нормализация переносов строк и дефисов
            text = re.sub(r'-\s+', '', text)
            # 4. Унификация кавычек
            text = text.replace('"', '"').replace('"', '"').replace('«', '"').replace('»', '"')

            # Добавляем дополнительную информацию для улучшения поиска
            normalized_text = text.lower()

            # Создаем отдельный документ для каждого названия раздела
            for section_name in all_sections:
                doc_copy = doc.copy()

                # Добавление расширенных метаданных
                doc_copy.metadata['cleaned'] = True
                doc_copy.metadata['length'] = len(text)
                doc_copy.metadata['normalized_text'] = normalized_text
                doc_copy.metadata['section'] = section_name
                doc_copy.metadata['primary_section'] = primary_section
                doc_copy.metadata['source_file'] = pdf_path
                doc_copy.metadata['all_sections'] = all_sections
                doc_copy.page_content = text
                cleaned_documents.append(doc_copy)

        logger.info(f"Загружено и очищено {len(cleaned_documents)} документов из {primary_section} (всего названий: {len(all_sections)})")
        return cleaned_documents

    def _create_hierarchical_chunks(self):
        """Создание чанков с семантическим разбиением"""
        logger.debug("Создание семантических чанков...")

        # Семантические разделители в порядке приоритета
        semantic_separators = [
            "\n\n\n",  # Разделы
            "\n\n",    # Параграфы
            "Глава ",  # Структурные элементы
            "Раздел ",
            "Часть ",
            ". ",      # Предложения (с проверкой пробела)
            "? ",      # Вопросы
            "! ",      # Восклицания
            ": ",      # Определения
            "; ",      # Части предложений
            "\n",      # Строки
        ]

        # Создаем сплиттер с семантическими правилами
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200,  # Увеличено для лучшего сохранения контекста
            length_function=len,
            separators=semantic_separators
        )

        # Создаем чанки
        chunks = []
        for doc in self.documents:
            # Разбиваем документ на чанки
            doc_chunks = splitter.split_documents([doc])

            # Обрабатываем каждый чанк
            for i, chunk in enumerate(doc_chunks):
                # Сохраняем метаданные
                chunk.metadata['chunk_index'] = i
                chunk.metadata['total_chunks'] = len(doc_chunks)
                chunk.metadata['chunk_size'] = len(chunk.page_content)

                # Определяем тип чанка
                if i == 0:
                    chunk.metadata['chunk_type'] = 'start'
                elif i == len(doc_chunks) - 1:
                    chunk.metadata['chunk_type'] = 'end'
                else:
                    chunk.metadata['chunk_type'] = 'middle'

                chunks.append(chunk)

        logger.info(f"Создано {len(chunks)} семантических чанков")
        return chunks

    def _create_or_load_vectorstore(self):
        """Создание или загрузка векторного хранилища с поддержкой Pinecone"""
        logger.debug("Создание/загрузка векторного хранилища...")

        # Создаем класс для преобразования всех запросов к нижнему регистру
        class CaseInsensitiveEmbeddings(OpenAIEmbeddings):
            def __init__(self):
                # Используем text-embedding-3-large для лучшего качества
                super().__init__(model="text-embedding-3-large")

            def embed_query(self, text: str) -> list:
                return super().embed_query(text.lower())

        embeddings = CaseInsensitiveEmbeddings()

        # Предварительная обработка документов
        processed_docs = []
        for doc in self.splits:
            if 'normalized_text' not in doc.metadata:
                doc.metadata['normalized_text'] = doc.page_content.lower()
            processed_docs.append(doc)

        logger.info(f"Всего документов для обработки: {len(processed_docs)}")

        # Логируем информацию о разделах
        sections_count = {}
        for doc in processed_docs:
            section = doc.metadata.get('section', 'unknown')
            sections_count[section] = sections_count.get(section, 0) + 1
        logger.info(f"Документы по разделам: {sections_count}")

        # Используем Pinecone если доступен, иначе FAISS
        if hasattr(self, 'pinecone_index') and self.pinecone_index:
            try:
                # Проверяем есть ли уже данные в индексе
                stats = self.pinecone_index.describe_index_stats()
                logger.info(f"Текущие данные в Pinecone: {stats}")

                if stats['total_vector_count'] > 0:
                    logger.info("Загрузка существующих эмбеддингов из Pinecone")
                    vectorstore = Pinecone(
                        index=self.pinecone_index,
                        embedding=embeddings,
                        text_key="text"
                    )
                else:
                    logger.info("Создание новых эмбеддингов в Pinecone (с батчингом)")
                    vectorstore = self._create_pinecone_vectorstore_with_batching(processed_docs, embeddings)

                # Проверяем финальную статистику
                final_stats = self.pinecone_index.describe_index_stats()
                logger.info(f"Финальная статистика Pinecone: {final_stats}")

                logger.info("Pinecone векторное хранилище готово")
                return vectorstore

            except Exception as e:
                logger.error(f"Ошибка работы с Pinecone: {str(e)}")
                logger.info("Переключение на FAISS...")

        # Fallback на FAISS с батчингом
        logger.info("Создание FAISS векторного хранилища с батчингом")
        vectorstore = self._create_faiss_vectorstore_with_batching(processed_docs, embeddings)
        logger.info("FAISS векторное хранилище создано успешно")
        return vectorstore

    def _create_faiss_vectorstore_with_batching(self, docs, embeddings, batch_size=50):
        """Создание FAISS векторного хранилища с батчингом"""
        logger.info(f"Создание FAISS с батчингом. Размер батча: {batch_size}")

        vectorstore = None

        # Обрабатываем документы батчами
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            logger.debug(f"Обработка батча {i//batch_size + 1}/{(len(docs) + batch_size - 1)//batch_size}, документов: {len(batch)}")

            try:
                if vectorstore is None:
                    # Создаем первый векторный магазин
                    vectorstore = FAISS.from_documents(batch, embeddings)
                else:
                    # Добавляем к существующему
                    batch_vectorstore = FAISS.from_documents(batch, embeddings)
                    vectorstore.merge_from(batch_vectorstore)

                logger.debug(f"Батч {i//batch_size + 1} обработан успешно")

            except Exception as e:
                logger.error(f"Ошибка обработки батча {i//batch_size + 1}: {str(e)}")
                # Уменьшаем размер батча и повторяем
                if batch_size > 10:
                    logger.info(f"Уменьшение размера батча до {batch_size//2}")
                    return self._create_faiss_vectorstore_with_batching(docs, embeddings, batch_size//2)
                else:
                    raise

        return vectorstore

    def _create_pinecone_vectorstore_with_batching(self, docs, embeddings, batch_size=50):
        """Создание Pinecone векторного хранилища с батчингом"""
        logger.info(f"Создание Pinecone с батчингом. Размер батча: {batch_size}")

        # Для Pinecone создаем батчами и добавляем в индекс
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            logger.debug(f"Обработка батча {i//batch_size + 1}/{(len(docs) + batch_size - 1)//batch_size}, документов: {len(batch)}")

            try:
                if i == 0:
                    # Создаем первый векторный магазин
                    vectorstore = Pinecone.from_documents(
                        batch,
                        embeddings,
                        index_name=self.index_name
                    )
                else:
                    # Добавляем к существующему индексу
                    Pinecone.from_documents(
                        batch,
                        embeddings,
                        index_name=self.index_name
                    )

                logger.debug(f"Батч {i//batch_size + 1} обработан успешно")

            except Exception as e:
                logger.error(f"Ошибка обработки батча {i//batch_size + 1}: {str(e)}")
                # Уменьшаем размер батча и повторяем
                if batch_size > 10:
                    logger.info(f"Уменьшение размера батча до {batch_size//2}")
                    return self._create_pinecone_vectorstore_with_batching(docs, embeddings, batch_size//2)
                else:
                    raise

        # Возвращаем векторное хранилище для работы с созданным индексом
        return Pinecone(
            index=self.pinecone_index,
            embedding=embeddings,
            text_key="text"
        )

    def _initialize_models(self):
        """Инициализация моделей разного размера для разных задач"""
        logger.debug("Инициализация моделей...")

        # Основная модель для генерации ответов
        self.llm = ChatOpenAI(
            model_name="gpt-4o",
            temperature=0
        )

        # Легкая модель для предварительного анализа
        self.light_llm = ChatOpenAI(
            model_name="gpt-4o",
            temperature=0.2
        )

        self.qa_chain = ConversationalRetrievalChain.from_llm(
            self.llm,
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5}),
            return_source_documents=True
        )
        logger.info("Модели инициализированы успешно")

    def ask_question(self, question: str, section_filter=None) -> str:
        try:
            # Валидация длины входного запроса для предотвращения промт-инжекшн
            if len(question) > MAX_QUESTION_LENGTH:
                return "Ошибка: Вопрос слишком длинный. Максимальная длина вопроса - 200 символов. Пожалуйста, сократите ваш вопрос."

            # Проверка на пустой запрос
            if not question.strip():
                return "Ошибка: Вопрос не может быть пустым."

            # Дополнительная проверка на подозрительные паттерны промт-инжекшн
            suspicious_patterns = [
                "ignore previous", "ignore above", "forget everything",
                "system prompt", "system message", "you are now",
                "act as", "pretend to be", "roleplay as",
                "ignore all previous", "disregard previous",
                "new instructions", "new rules", "new system"
            ]

            question_lower = question.lower()
            for pattern in suspicious_patterns:
                if pattern in question_lower:
                    return "Ошибка: Обнаружен подозрительный паттерн в запросе. Пожалуйста, задайте вопрос о содержании книги."

            logger.debug(f"Обработка вопроса: {question}")
            if section_filter:
                logger.debug(f"Фильтр по разделу: {section_filter}")

            # Нормализация вопроса
            normalized_question = question.lower()

            # Определение раздела из вопроса если не указан фильтр
            if not section_filter:
                section_filter = self._detect_section_from_question(normalized_question)

            # Гибридный поиск с фильтрацией по разделам
            relevant_docs = self._hybrid_search_with_filter(normalized_question, section_filter)
            if not relevant_docs:
                return "В тексте книги не найдено релевантной информации для ответа на этот вопрос."

            # Создание контекстного окна
            context = self._create_context_window(relevant_docs)

            # Формирование улучшенного промпта
            enhanced_prompt = self._create_enhanced_prompt(question, context, section_filter)

            # Получение ответа
            result = self.qa_chain({
                "question": enhanced_prompt,
                "chat_history": self.chat_history
            })

            self.chat_history.append((question, result["answer"]))

            # Постобработка ответа
            final_answer = self._post_process_answer(result["answer"], result["source_documents"])

            return final_answer

        except Exception as e:
            error_msg = f"Ошибка при обработке вопроса: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return f"Произошла ошибка: {error_msg}"

    def _detect_section_from_question(self, question: str):
        """Определение раздела из вопроса пользователя"""
        question_lower = question.lower()

        # Проверяем ключевые слова для определения раздела
        for section_key, _ in self.sections_mapping.items():
            if section_key in question_lower:
                logger.debug(f"Обнаружен раздел из вопроса: {section_key}")
                return section_key

        return None

    def _hybrid_search_with_filter(self, question: str, section_filter=None):
        """Улучшенный гибридный поиск"""
        try:
            normalized_question = question.lower()

            # Параметры поиска
            k_vector = 8
            k_final = 10

            # Векторный поиск
            vector_docs = []
            if section_filter:
                if hasattr(self.vectorstore, 'similarity_search_with_score'):
                    results = self.vectorstore.similarity_search_with_score(normalized_question, k=k_vector)
                    vector_docs = [(doc, score) for doc, score in results
                                 if doc.metadata.get('section') == section_filter][:k_vector]
                else:
                    filter_dict = {"section": {"$eq": section_filter}}
                    docs = self.vectorstore.similarity_search(
                        normalized_question,
                        k=k_vector,
                        filter=filter_dict
                    )
                    vector_docs = [(doc, 1.0) for doc in docs]
            else:
                if hasattr(self.vectorstore, 'similarity_search_with_score'):
                    vector_docs = self.vectorstore.similarity_search_with_score(normalized_question, k=k_vector)
                else:
                    vector_docs = [(doc, 1.0) for doc in self.vectorstore.similarity_search(normalized_question, k=k_vector)]

            # Ключевой поиск с улучшенным скорингом
            keyword_docs = []
            if self.splits:
                keywords = self._extract_keywords(normalized_question)

                for doc in self.splits:
                    if section_filter and doc.metadata.get('section') != section_filter:
                        continue

                    doc_content_lower = doc.page_content.lower()

                    # Подсчет совпадений с учетом позиции в тексте
                    matches = 0
                    for keyword in keywords:
                        if keyword in doc_content_lower:
                            # Если ключевое слово ближе к началу чанка, оно важнее
                            position = doc_content_lower.index(keyword) / len(doc_content_lower)
                            matches += 1 * (1 - position * 0.5)  # Вес выше для слов в начале

                    if matches > 0:
                        # Нормализуем score с учетом типа чанка
                        base_score = matches / len(keywords)
                        chunk_type_multiplier = {
                            'start': 1.2,  # Повышаем вес начальных чанков
                            'end': 1.1,    # Слегка повышаем вес конечных чанков
                            'middle': 1.0
                        }.get(doc.metadata.get('chunk_type', 'middle'), 1.0)

                        final_score = base_score * chunk_type_multiplier
                        keyword_docs.append((doc, final_score))

            # Объединение результатов с весами
            unique_docs = {}
            for doc, score in vector_docs:
                key = (doc.page_content, doc.metadata.get('page'), doc.metadata.get('section'))
                unique_docs[key] = (doc, score * 0.7)  # Вес для векторного поиска

            for doc, score in keyword_docs:
                key = (doc.page_content, doc.metadata.get('page'), doc.metadata.get('section'))
                if key in unique_docs:
                    # Комбинируем scores если документ уже есть
                    unique_docs[key] = (doc, unique_docs[key][1] + score * 0.3)
                else:
                    unique_docs[key] = (doc, score * 0.3)

            # Сортировка и возврат результатов
            sorted_docs = sorted(unique_docs.values(), key=lambda x: x[1], reverse=True)
            return [doc for doc, _ in sorted_docs[:k_final]]

        except Exception as e:
            logger.error(f"Ошибка в гибридном поиске: {str(e)}", exc_info=True)
            return [doc for doc, _ in vector_docs[:k_vector]] if 'vector_docs' in locals() else []

    def _create_context_window(self, docs):
        """Создание контекстного окна с учетом окружающих фрагментов"""
        context_parts = []
        for doc in docs:
            # Добавляем контекст из крупных чанков
            if doc.metadata.get('chunk_type') == 'large':
                context_parts.append(doc.page_content)
            else:
                # Для малых чанков добавляем соседние фрагменты (только если self.splits не пуст)
                if self.splits:
                    page_num = doc.metadata.get('page')
                    context_parts.extend([d.page_content for d in self.splits
                                       if d.metadata.get('page') == page_num])
                else:
                    # Если splits пуст, просто добавляем сам документ
                    context_parts.append(doc.page_content)

        return "\n\n".join(set(context_parts))

    def _create_enhanced_prompt(self, question: str, context: str, section_filter=None):
        """Создание улучшенного промпта с информацией о разделах"""
        normalized_question = question.strip()

        # Добавляем информацию о разделе в промпт
        section_info = ""
        if section_filter:
            section_info = f"\n\nВнимание: Этот вопрос относится к разделу '{section_filter}' книги."

        prompt_template = """Ты - специализированный ассистент по книге "Бизнес в диалоге: от малого к невозможному" Анвара Халикова.
        Твоя главная задача - помогать читателям глубже понять содержание этой книги, отвечая на их вопросы, опираясь ИСКЛЮЧИТЕЛЬНО на информацию из книги.

        КЛЮЧЕВЫЕ ПРИНЦИПЫ РАБОТЫ:
        1. Отвечай ТОЛЬКО на основе информации из предоставленного контекста книги
        2. Если информации в контексте недостаточно или она отсутствует - честно сообщи об этом
        3. Не придумывай и не добавляй информацию, которой нет в книге
        4. Структурируй ответы по пунктам, если это уместно
        5. Используй цитаты из книги для подтверждения важных моментов
        6. Сохраняй деловой и дружелюбный стиль общения
        7. ВАЖНО: Если вопрос не связан с содержанием книги, вежливо объясни, что ты специализируешься только на содержании книги "Бизнес в диалоге" и можешь отвечать только на вопросы по ней
        8. При цитировании указывай раздел книги, откуда взята информация

        КОНТЕКСТ ИЗ КНИГИ:
        {context}

        ВОПРОС:
        {question}

        ОТВЕТ НА ОСНОВЕ КНИГИ:"""

        return PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question", "section_info"]
        ).format(context=context, question=normalized_question, section_info=section_info)

    def _post_process_answer(self, answer: str, source_docs: list):
        """Постобработка ответа с информацией о разделах"""
        # Форматируем источники с указанием разделов
        sources = []
        sections = set()

        for doc in source_docs:
            page = doc.metadata.get('page', 0) + 1
            section = doc.metadata.get('section', 'unknown')
            sources.append(f"Страница {page}")
            sections.add(section)

        unique_sources = sorted(set(sources))

        # Форматируем ответ
        formatted_answer = answer.strip()
        if not formatted_answer.startswith("Ответ:"):
            formatted_answer = "Ответ: " + formatted_answer

        # Добавляем информацию о разделах
        sections_info = ""
        if len(sections) > 1:
            sections_info = f"\nРазделы: {', '.join(sorted(sections))}"
        elif len(sections) == 1:
            section = list(sections)[0]
            if section != 'unknown':
                sections_info = f"\nРаздел: {section}"

        return f"{formatted_answer}\n\nИсточники: {', '.join(unique_sources)}{sections_info}"

    def search_by_section(self, section_name: str, query: str = "") -> str:
        """Поиск информации в конкретном разделе книги"""
        try:
            # Валидация длины входного запроса для предотвращения промт-инжекшн
            if len(query) > MAX_QUESTION_LENGTH:
                return "Ошибка: Запрос слишком длинный. Максимальная длина запроса - 200 символов. Пожалуйста, сократите ваш запрос."

            # Нормализуем название раздела
            section_key = section_name.lower().replace(' ', '_')

            logger.debug(f"Поиск в разделе: '{section_name}' -> '{section_key}'")

            if section_key not in self.sections_mapping:
                available_sections = ', '.join(self.sections_mapping.keys())
                return f"Раздел '{section_name}' не найден. Доступные разделы: {available_sections}"

            # Если запрос не указан, возвращаем общую информацию о разделе
            if not query:
                query = f"Расскажи о содержании раздела {section_name}"

            # Выполняем поиск с фильтром по разделу
            return self.ask_question(query, section_filter=section_key)

        except Exception as e:
            error_msg = f"Ошибка при поиске в разделе {section_name}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg

    def get_available_sections(self) -> list:
        """Получение списка доступных разделов"""
        return list(self.sections_mapping.keys())

    def force_rebuild_embeddings(self):
        """Принудительное пересоздание эмбеддингов"""
        logger.info("Принудительное пересоздание эмбеддингов...")

        try:
            # Очищаем Pinecone индекс если он существует
            if hasattr(self, 'pinecone_index') and self.pinecone_index:
                logger.info("Очистка существующего Pinecone индекса...")
                # Удаляем все векторы из индекса
                self.pinecone_index.delete(delete_all=True)
                logger.info("Pinecone индекс очищен")

            # Загружаем документы заново
            logger.info("Загрузка документов...")
            if self.use_sections:
                self.documents = self._load_all_sections()
            else:
                self.documents = self._load_full_book()

            # Создаем чанки заново
            logger.info("Создание чанков...")
            self.splits = self._create_hierarchical_chunks()

            # Создаем векторное хранилище заново
            logger.info("Создание векторного хранилища...")
            self.vectorstore = self._create_or_load_vectorstore()

            # Переинициализируем модели
            self._initialize_models()

            logger.info("Эмбеддинги успешно пересозданы")
            return True

        except Exception as e:
            logger.error(f"Ошибка при пересоздании эмбеддингов: {str(e)}", exc_info=True)
            return False