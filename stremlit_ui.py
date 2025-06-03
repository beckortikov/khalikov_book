import streamlit as st
from main import BookRAG
import os
import logging

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Книжный помощник с RAG",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Книжный помощник с RAG")

# Инициализация RAG при запуске приложения
if 'rag' not in st.session_state:
    try:
        with st.spinner('Инициализация системы...'):
            logger.info("Инициализация RAG при запуске...")
            st.session_state.rag = BookRAG()
            st.success('Система успешно инициализирована!')
    except FileNotFoundError:
        st.error("Файл book.pdf не найден в директории приложения")
        st.stop()
    except Exception as e:
        st.error(f"Ошибка при инициализации системы: {str(e)}")
        logger.error(f"Ошибка инициализации: {str(e)}", exc_info=True)
        st.stop()

# Создаем боковую панель
with st.sidebar:
    st.header("Настройки")

    # Получаем список доступных разделов
    available_sections = st.session_state.rag.get_available_sections()

    # Выбор раздела
    selected_section = st.selectbox(
        "Выберите раздел книги",
        ["Вся книга"] + available_sections,
        index=0
    )

    # Кнопка для пересоздания эмбеддингов
    if st.button("Пересоздать эмбеддинги"):
        with st.spinner("Пересоздание эмбеддингов..."):
            success = st.session_state.rag.force_rebuild_embeddings()
            if success:
                st.success("Эмбеддинги успешно пересозданы!")
            else:
                st.error("Ошибка при пересоздании эмбеддингов")

# Основной интерфейс
st.subheader("Задайте вопрос о книге")

# Интерфейс для вопросов
question = st.text_input("Ваш вопрос:", key="question_input")

if question:
    try:
        with st.spinner('Ищу ответ...'):
            # Если выбран конкретный раздел
            if selected_section != "Вся книга":
                response = st.session_state.rag.search_by_section(selected_section, question)
            else:
                response = st.session_state.rag.ask_question(question)

            # Разделяем ответ на основную часть и метаданные
            parts = response.split("\n\nИсточники:")
            main_answer = parts[0]
            metadata = "Источники:" + parts[1] if len(parts) > 1 else ""

            # Выводим основной ответ
            st.write(main_answer)

            # Выводим метаданные в отдельном блоке
            if metadata:
                with st.expander("Метаданные ответа"):
                    st.write(metadata)

            logger.info("Успешно получен ответ на вопрос")
    except Exception as e:
        error_msg = f"Ошибка при обработке вопроса: {str(e)}"
        logger.error(error_msg, exc_info=True)
        st.error(error_msg)

# Добавляем информацию о текущем разделе
if selected_section != "Вся книга":
    st.info(f"🔍 Поиск осуществляется в разделе: {selected_section}")

# Добавляем краткую инструкцию
with st.expander("ℹ️ Как пользоваться"):
    st.markdown("""
    1. Выберите раздел книги в боковой панели (или оставьте "Вся книга" для поиска по всему тексту)
    2. Введите ваш вопрос в текстовое поле
    3. Получите структурированный ответ с указанием источников

    **Примеры вопросов:**
    - "Что такое ключевой фактор успеха?"
    - "Расскажи о структуре команды"
    - "Как справляться с кризисом в бизнесе?"
    """)