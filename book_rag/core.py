import asyncio
import logging
import os
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Any

from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import HumanMessage, SystemMessage, Document

from book_rag.config import logger, MAX_QUESTION_LENGTH, SECTIONS_MAPPING
from book_rag.text_utils import is_suspicious_query, enhance_query
from book_rag.data import BookLoader
from book_rag.storage import StorageManager
from book_rag.search import SearchEngine, create_bm25_index, create_metadata_index

class ImprovedBookRAG:
    def __init__(self, use_sections=True, pdf_directory="book", cache_embeddings=True):
        try:
            logger.info(f"Инициализация улучшенной RAG системы с кэшированием: {cache_embeddings}")
            self.use_sections = use_sections
            self.pdf_directory = pdf_directory
            self.cache_embeddings = cache_embeddings
            
            # Инициализация менеджеров
            self.storage_manager = StorageManager(pinecone_api_key=os.getenv("PINECONE_API_KEY"))
            self.loader = BookLoader(pdf_directory=pdf_directory)
            
            # Данные
            self.documents = []
            self.chunks = []
            self.vectorstore = None
            self.bm25_index = None
            self.metadata_index = None
            self.search_engine = None

            # Загрузка или создание данных
            self._load_or_create_data()

            # Инициализация поиска
            self.search_engine = SearchEngine(
                vectorstore=self.vectorstore,
                bm25_index=self.bm25_index,
                chunks=self.chunks,
                pinecone_available=self.storage_manager.pinecone_available
            )

            # Инициализация моделей
            self._initialize_models()

            # История чата
            self.chat_history = []

            logger.info("Улучшенная RAG система инициализирована успешно")

        except Exception as e:
            logger.error(f"Критическая ошибка при инициализации RAG: {str(e)}", exc_info=True)
            raise

    def _load_or_create_data(self):
        """Загрузка или создание данных с кэшированием"""
        
        # Проверяем кэш
        if self.cache_embeddings:
            cached_data = self.storage_manager.load_from_cache()
            if cached_data:
                self.documents = cached_data['documents']
                self.chunks = cached_data['chunks']
                self.bm25_index = cached_data['bm25_index']
                self.metadata_index = cached_data['metadata_index']
                self.vectorstore = cached_data['vectorstore']
                return

        # Создаем данные с нуля
        logger.info("Создание данных с нуля...")
        
        # 1. Загрузка документов
        if self.use_sections:
            self.documents = asyncio.run(self.loader.load_all_sections())
        else:
            self.documents = asyncio.run(self.loader.load_full_book())

        # 2. Умные чанки
        self.chunks = self.loader.create_smart_chunks(self.documents)

        # 3. Индексы поиска
        self.bm25_index = create_bm25_index(self.chunks)
        self.metadata_index = create_metadata_index(self.chunks)

        # 4. Векторное хранилище
        self.vectorstore = self.storage_manager.create_vectorstore(self.chunks)

        # Сохраняем в кэш
        if self.cache_embeddings:
            data_to_cache = {
                'documents': self.documents,
                'chunks': self.chunks,
                'bm25_index': self.bm25_index,
                'metadata_index': self.metadata_index,
                'vectorstore': self.vectorstore
            }
            self.storage_manager.save_to_cache(data_to_cache)

    def _initialize_models(self):
        """Инициализация языковых моделей"""
        logger.info("Инициализация языковых моделей...")

        # Основная модель для ответов
        self.llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.1,
            max_tokens=2000
        )

        # Легкая модель для анализа запросов (если потребуется в будущем)
        self.light_llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0,
            max_tokens=500
        )

        logger.info("Языковые модели инициализированы")

    def ask_question(self, question: str, section_filter: Optional[str] = None) -> str:
        """Основной метод для ответа на вопросы"""
        try:
            # Валидация входных данных
            if not question or not question.strip():
                return "Ошибка: Вопрос не может быть пустым."

            if len(question) > MAX_QUESTION_LENGTH:
                return f"Ошибка: Вопрос слишком длинный (максимум {MAX_QUESTION_LENGTH} символов)."

            # Проверка на подозрительные запросы
            if is_suspicious_query(question):
                return "Ошибка: Обнаружен подозрительный запрос. Задайте вопрос о книге."

            logger.info(f"Обработка вопроса: {question[:100]}...")

            # Анализ и улучшение запроса
            enhanced_query = enhance_query(question)

            # Гибридный поиск релевантных документов
            relevant_docs = self.search_engine.hybrid_search(enhanced_query, section_filter)

            if not relevant_docs:
                return self._handle_no_results(question, section_filter)

            # Создание контекста и ответа
            context = self._create_optimized_context(relevant_docs)
            prompt = self._create_smart_prompt(question, context, section_filter)

            
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
            # Note: We handle pinecone fallback in initialization, but runtime errors might occur
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

            if section_key not in SECTIONS_MAPPING:
                available = ', '.join(self.get_available_sections())
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

        for section in SECTIONS_MAPPING.keys():
            # Группируем похожие названия
            base_name = section.replace('_', ' ')
            if base_name not in seen:
                main_sections.append(section)
                seen.add(base_name)

        return sorted(main_sections)

    def get_statistics(self) -> Dict:
        """Получение статистики о загруженных данных"""
        if not self.chunks:
            return {"error": "Данные не загружены"}

        stats = {
            "total_chunks": len(self.chunks),
            "total_documents": len(self.documents) if self.documents else 0,
            "sections": len(set(doc.metadata.get('section', '') for doc in self.documents)) if self.documents else 0,
            "pages": len(set(doc.metadata.get('page', 0) for doc in self.documents)) if self.documents else 0,
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
            self.storage_manager.clear_cache()
            
            # Пересоздаем данные
            self._load_or_create_data()

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
