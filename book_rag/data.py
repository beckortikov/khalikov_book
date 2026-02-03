import asyncio
import aiofiles
import fitz  # PyMuPDF
import hashlib
import logging
import os
import re
import spacy
from collections import defaultdict
from typing import List, Dict, Optional

from langchain.schema import Document

from book_rag.config import logger, SECTIONS_MAPPING
from book_rag.text_utils import advanced_text_cleaning, simple_sentence_split

class BookLoader:
    def __init__(self, pdf_directory: str = "book"):
        self.pdf_directory = pdf_directory

    async def load_all_sections(self) -> List[Document]:
        """Улучшенная загрузка всех разделов"""
        file_to_sections = defaultdict(list)

        for section_name, file_path in SECTIONS_MAPPING.items():
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

    async def load_full_book(self) -> List[Document]:
        """Загрузка полной книги"""
        book_path = os.path.join(self.pdf_directory, "book.pdf")
        if not os.path.exists(book_path):
            # Fallback to checking keys in mapping if they point to a full book
             if "book/book.pdf" in SECTIONS_MAPPING.values():
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
                cleaned_text = advanced_text_cleaning(text)

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

    def create_smart_chunks(self, documents: List[Document]) -> List[Document]:
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

        for doc in documents:
            text = doc.page_content

            # Разделение на предложения
            if nlp:
                try:
                    spacy_doc = nlp(text)
                    sentences = [sent.text.strip() for sent in spacy_doc.sents]
                except Exception:
                    sentences = simple_sentence_split(text)
            else:
                sentences = simple_sentence_split(text)

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
            "sentence_count": len(simple_sentence_split(text))
        })

        return Document(page_content=text, metadata=metadata)

    def _create_smart_overlap(self, text: str, overlap_size: int) -> str:
        """Создание умного перекрытия по границам предложений"""
        if len(text) <= overlap_size:
            return text

        sentences = simple_sentence_split(text)

        # Берем последние предложения, которые помещаются в overlap
        overlap_text = ""
        for sentence in reversed(sentences):
            candidate = sentence + ". " + overlap_text
            if len(candidate) <= overlap_size:
                overlap_text = candidate
            else:
                break

        return overlap_text.strip()

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
