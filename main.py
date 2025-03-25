from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import os
import logging
import re
from dotenv import load_dotenv

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
    def __init__(self, pdf_path="book.pdf"):
        try:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"Файл {pdf_path} не найден")

            logger.debug(f"Начало инициализации RAG с файлом: {pdf_path}")
            self.pdf_path = pdf_path

            # Улучшение 1: Иерархическое разделение текста
            self.documents = self._load_and_clean_pdf()

            # Улучшение 2: Многоуровневое разделение на чанки
            self.splits = self._create_hierarchical_chunks()

            # Улучшение 3: Создание эмбеддингов с метаданными
            self.vectorstore = self._create_vectorstore()

            # Улучшение 4: Инициализация двух моделей для разных задач
            self._initialize_models()

            self.chat_history = []
            logger.info("Инициализация RAG завершена успешно")

        except Exception as e:
            logger.error(f"Критическая ошибка при инициализации RAG: {str(e)}", exc_info=True)
            raise

    def _load_and_clean_pdf(self):
        """Улучшенная загрузка и очистка PDF с нормализацией текста"""
        logger.debug("Загрузка и очистка PDF файла...")
        loader = PyPDFLoader(self.pdf_path)
        documents = loader.load()

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
            normalized_text = text.lower()  # Сохраняем нормализованную версию в метаданных

            # Добавление метаданных
            doc.metadata['cleaned'] = True
            doc.metadata['length'] = len(text)
            doc.metadata['normalized_text'] = normalized_text
            doc.page_content = text
            cleaned_documents.append(doc)

        logger.info(f"Загружено и очищено {len(cleaned_documents)} страниц")
        return cleaned_documents

    def _create_hierarchical_chunks(self):
        """Создание иерархических чанков разного размера"""
        logger.debug("Создание иерархических чанков...")

        # Большие чанки для контекста
        large_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            length_function=len
        )

        # Малые чанки для точного поиска
        small_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len
        )

        large_chunks = large_splitter.split_documents(self.documents)
        small_chunks = small_splitter.split_documents(self.documents)

        # Добавляем метаданные о размере чанка
        for chunk in large_chunks:
            chunk.metadata['chunk_type'] = 'large'
        for chunk in small_chunks:
            chunk.metadata['chunk_type'] = 'small'

        all_chunks = large_chunks + small_chunks
        logger.info(f"Создано {len(all_chunks)} чанков (крупных: {len(large_chunks)}, мелких: {len(small_chunks)})")
        return all_chunks

    def _create_vectorstore(self):
        """Создание векторного хранилища с метаданными и регистронезависимым поиском"""
        logger.debug("Создание векторного хранилища...")

        # Создание класса для преобразования всех запросов к нижнему регистру
        class CaseInsensitiveEmbeddings(OpenAIEmbeddings):
            def embed_query(self, text: str) -> list:
                # Преобразуем текст запроса к нижнему регистру перед созданием эмбеддинга
                return super().embed_query(text.lower())

        # Используем наши улучшенные эмбеддинги с нормализацией регистра
        embeddings = CaseInsensitiveEmbeddings()

        # Предварительная обработка документов, чтобы в них была
        # информация как с сохранением регистра, так и с нормализацией
        processed_docs = []
        for doc in self.splits:
            # Добавляем к метаданным нормализованную версию текста, если её ещё нет
            if 'normalized_text' not in doc.metadata:
                doc.metadata['normalized_text'] = doc.page_content.lower()
            processed_docs.append(doc)

        vectorstore = FAISS.from_documents(processed_docs, embeddings)
        logger.info("Векторное хранилище с регистронезависимым поиском создано успешно")
        return vectorstore

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

    def ask_question(self, question: str) -> str:
        try:
            logger.debug(f"Обработка вопроса: {question}")

            # Нормализация вопроса (приведение к нижнему регистру для поиска)
            # Но сохраняем оригинальный вопрос для ответа
            normalized_question = question.lower()

            # Улучшение 5: Гибридный поиск с нормализованным вопросом
            relevant_docs = self._hybrid_search(normalized_question)
            if not relevant_docs:
                return "В тексте книги не найдено релевантной информации для ответа на этот вопрос."

            # Улучшение 6: Контекстное окно
            context = self._create_context_window(relevant_docs)

            # Формируем улучшенный промпт
            enhanced_prompt = self._create_enhanced_prompt(question, context)

            # Получаем ответ
            result = self.qa_chain({
                "question": enhanced_prompt,
                "chat_history": self.chat_history
            })

            self.chat_history.append((question, result["answer"]))

            # Улучшение 7: Постобработка ответа
            final_answer = self._post_process_answer(result["answer"], result["source_documents"])

            return final_answer

        except Exception as e:
            error_msg = f"Ошибка при обработке вопроса: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return f"Произошла ошибка: {error_msg}"

    def _hybrid_search(self, question: str):
        """Гибридный поиск: комбинация векторного и ключевого поиска"""
        try:
            # Подготовка текста запроса (приведение к нижнему регистру)
            normalized_question = question.lower()

            # Векторный поиск с нормализованным запросом
            vector_docs = self.vectorstore.similarity_search(normalized_question, k=3)

            # Простой ключевой поиск с улучшенной нормализацией
            keywords = self._extract_keywords(normalized_question)
            keyword_docs = []

            for doc in self.splits:
                doc_content_lower = doc.page_content.lower()
                if any(keyword in doc_content_lower for keyword in keywords):
                    keyword_docs.append(doc)

            # Объединяем результаты без использования set()
            # Используем словарь для удаления дубликатов по содержимому
            unique_docs = {}
            for doc in vector_docs + keyword_docs:
                # Используем содержимое и номер страницы как ключ
                key = (doc.page_content, doc.metadata.get('page'))
                unique_docs[key] = doc

            # Возвращаем топ-5 уникальных документов
            return list(unique_docs.values())[:5]

        except Exception as e:
            logger.error(f"Ошибка в гибридном поиске: {str(e)}", exc_info=True)
            # В случае ошибки возвращаем только результаты векторного поиска
            return vector_docs[:5]

    def _extract_keywords(self, text: str):
        """Извлечение ключевых слов с нормализацией"""
        # Удаляем стоп-слова и оставляем только существенные слова
        stop_words = {'и', 'в', 'на', 'с', 'по', 'к', 'у', 'о', 'из', 'что', 'как', 'кто', 'для', 'при'}
        words = text.lower().split()
        keywords = [word for word in words if len(word) > 3 and word not in stop_words]

        # Добавляем словоформы (простая стемминг-эмуляция для русского языка)
        stemmed_keywords = []
        for word in keywords:
            stemmed_keywords.append(word)
            # Добавляем вариант без окончания (очень упрощенный подход)
            if len(word) > 5:
                stemmed_keywords.append(word[:-1])  # Без последней буквы
                stemmed_keywords.append(word[:-2])  # Без двух последних букв

        return stemmed_keywords

    def _create_context_window(self, docs):
        """Создание контекстного окна с учетом окружающих фрагментов"""
        context_parts = []
        for doc in docs:
            # Добавляем контекст из крупных чанков
            if doc.metadata.get('chunk_type') == 'large':
                context_parts.append(doc.page_content)
            else:
                # Для малых чанков добавляем соседние фрагменты
                page_num = doc.metadata.get('page')
                context_parts.extend([d.page_content for d in self.splits
                                   if d.metadata.get('page') == page_num])

        return "\n\n".join(set(context_parts))

    def _create_enhanced_prompt(self, question: str, context: str):
        """Создание улучшенного промпта с нормализацией регистра"""
        # Сохраняем оригинальный вопрос для промпта, но проводим предварительную обработку
        normalized_question = question.strip()

        # Добавляем инструкцию по регистронезависимому поиску
        prompt_template = """Ты - эксперт по анализу бизнес-литературы.
        Твоя задача - дать подробный, структурированный ответ на вопрос, используя ТОЛЬКО предоставленный контекст.

        ПРАВИЛА:
        1. Используй ТОЛЬКО информацию из контекста
        2. Структурируй ответ по пунктам, если это уместно
        3. Цитируй важные части текста
        4. Если информации недостаточно - честно признай это
        5. Сохраняй деловой стиль общения
        6. Проводи регистронезависимый поиск (игнорируй разницу между заглавными и строчными буквами)
        7. ВАЖНО: Если вопрос не связан с содержанием книги (например, просьба написать код, приготовить рецепт,
           рассказать о погоде, курсе валют, или выполнить любую другую задачу, не имеющую отношения к анализу
           текста книги), вежливо откажись отвечать и объясни, что ты можешь помочь только с вопросами,
           связанными с содержанием книги

        КОНТЕКСТ:
        {context}

        ВОПРОС:
        {question}

        СТРУКТУРИРОВАННЫЙ ОТВЕТ:"""

        return PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        ).format(context=context, question=normalized_question)

    def _post_process_answer(self, answer: str, source_docs: list):
        """Постобработка ответа"""
        # Форматируем источники
        sources = [f"Страница {doc.metadata['page']+1}" for doc in source_docs]
        unique_sources = sorted(set(sources))

        # Форматируем ответ
        formatted_answer = answer.strip()
        if not formatted_answer.startswith("Ответ:"):
            formatted_answer = "Ответ: " + formatted_answer

        return f"{formatted_answer}\n\nИсточники: {', '.join(unique_sources)}"