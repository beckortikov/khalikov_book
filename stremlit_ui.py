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

# Добавляем CSS для стилизации чата
st.markdown("""
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.assistant {
    background-color: #475063
}
.chat-message .avatar {
    width: 20%;
}
.chat-message .avatar img {
    max-width: 78px;
    max-height: 78px;
    border-radius: 50%;
    object-fit: cover;
}
.chat-message .message {
    width: 80%;
    padding: 0 1.5rem;
    color: #fff;
}
</style>
""", unsafe_allow_html=True)

# Функция для отображения сообщения чата
def message(content, is_user=False, is_error=False):
    avatar = "👤" if is_user else "🤖"
    background_color = "#2b313e" if is_user else "#475063"
    if is_error:
        background_color = "#8B0000"

    st.markdown(f"""
    <div class="chat-message" style="background-color: {background_color}">
        <div class="avatar">
            <div style="font-size: 50px; text-align: center;">{avatar}</div>
        </div>
        <div class="message">{content}</div>
    </div>
    """, unsafe_allow_html=True)

# Инициализация состояния сессии
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'rag' not in st.session_state:
    try:
        with st.spinner('Инициализация системы...'):
            logger.info("Инициализация RAG при запуске...")
            st.session_state.rag = BookRAG()
            st.session_state.messages.append({
                "role": "assistant",
                "content": "👋 Здравствуйте! Я ваш книжный помощник. Задайте мне вопрос о книге, и я постараюсь помочь вам найти ответ."
            })
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

    # Кнопка для очистки истории
    if st.button("Очистить историю"):
        st.session_state.messages = [{
            "role": "assistant",
            "content": "👋 История очищена. Задайте мне новый вопрос о книге."
        }]
        st.experimental_rerun()

# Основной интерфейс
st.title("📚 Книжный помощник с RAG")

# Если выбран конкретный раздел, показываем информацию
if selected_section != "Вся книга":
    st.info(f"🔍 Поиск осуществляется в разделе: {selected_section}")

# Отображаем историю сообщений
for message_data in st.session_state.messages:
    message(
        message_data["content"],
        is_user=(message_data["role"] == "user"),
        is_error=message_data.get("is_error", False)
    )

# Поле ввода
if prompt := st.chat_input("Задайте вопрос о книге..."):
    # Добавляем вопрос пользователя в историю
    st.session_state.messages.append({"role": "user", "content": prompt})
    message(prompt, is_user=True)

    try:
        with st.spinner('Ищу ответ...'):
            # Если выбран конкретный раздел
            if selected_section != "Вся книга":
                response = st.session_state.rag.search_by_section(selected_section, prompt)
            else:
                response = st.session_state.rag.ask_question(prompt)

            # Разделяем ответ на основную часть и метаданные
            parts = response.split("\n\nИсточники:")
            main_answer = parts[0]
            metadata = "Источники:" + parts[1] if len(parts) > 1 else ""

            # Формируем полный ответ с метаданными
            full_response = f"{main_answer}\n\n<details><summary>Метаданные ответа</summary>{metadata}</details>"

            # Добавляем ответ в историю
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response
            })
            message(full_response)

            logger.info("Успешно получен ответ на вопрос")
    except Exception as e:
        error_msg = f"❌ Ошибка при обработке вопроса: {str(e)}"
        logger.error(error_msg, exc_info=True)
        st.session_state.messages.append({
            "role": "assistant",
            "content": error_msg,
            "is_error": True
        })
        message(error_msg, is_error=True)

# Добавляем краткую инструкцию
with st.sidebar:
    with st.expander("ℹ️ Как пользоваться"):
        st.markdown("""
        1. Выберите раздел книги в боковой панели (или оставьте "Вся книга" для поиска по всему тексту)
        2. Введите ваш вопрос в поле внизу экрана
        3. Получите структурированный ответ с указанием источников

        **Примеры вопросов:**
        - "Что такое ключевой фактор успеха?"
        - "Расскажи о структуре команды"
        - "Как справляться с кризисом в бизнесе?"
        """)