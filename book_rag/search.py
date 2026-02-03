import logging
from typing import List, Tuple, Dict, Optional, Any
from collections import defaultdict
from rank_bm25 import BM25Okapi
from langchain.schema import Document

from book_rag.config import logger
from book_rag.text_utils import advanced_tokenize

def create_bm25_index(chunks: List[Document]) -> BM25Okapi:
    """Создание BM25 индекса с оптимизированными параметрами для русского"""
    logger.info("Создание BM25 индекса...")
    texts = [chunk.page_content.lower() for chunk in chunks]
    tokenized_texts = [advanced_tokenize(text) for text in texts]
    return BM25Okapi(tokenized_texts, k1=1.2, b=0.75)

def create_metadata_index(chunks: List[Document]) -> Dict:
    """Создание индекса метаданных для быстрой фильтрации"""
    index = {
        'by_section': defaultdict(list),
        'by_page': defaultdict(list),
        'by_chunk_type': defaultdict(list),
        'with_numbers': [],
        'with_questions': []
    }

    for i, chunk in enumerate(chunks):
        metadata = chunk.metadata

        index['by_section'][metadata.get('section', 'unknown')].append(i)
        index['by_page'][metadata.get('page', 0)].append(i)
        index['by_chunk_type'][metadata.get('chunk_type', 'standard')].append(i)

        if metadata.get('has_numbers', False):
            index['with_numbers'].append(i)
        if metadata.get('has_questions', False):
            index['with_questions'].append(i)

    return index

class SearchEngine:
    def __init__(self, vectorstore: Any, bm25_index: BM25Okapi, chunks: List[Document], pinecone_available: bool = False):
        self.vectorstore = vectorstore
        self.bm25_index = bm25_index
        self.chunks = chunks
        self.pinecone_available = pinecone_available

    def hybrid_search(self, query: str, section_filter: Optional[str] = None) -> List[Document]:
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
            tokenized_query = advanced_tokenize(query.lower())

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
        weights = self._calculate_adaptive_weights(query)

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

    def _calculate_adaptive_weights(self, query: str) -> Dict[str, float]:
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
