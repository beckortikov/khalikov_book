import streamlit as st
import logging
import sys
import os

# Добавляем путь к модулям
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import BookRAG

# Константы безопасности
MAX_QUESTION_LENGTH = 200

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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

# Инициализация настроек если их нет
if 'use_multi_vector' not in st.session_state:
    st.session_state.use_multi_vector = False
if 'use_sections' not in st.session_state:
    st.session_state.use_sections = True

if 'rag' not in st.session_state:
    try:
        with st.spinner('Инициализация системы...'):
            logger.info("Инициализация RAG при запуске...")
            st.session_state.rag = BookRAG(
                use_sections=st.session_state.use_sections,
                use_multi_vector=st.session_state.use_multi_vector
            )

            # Приветственное сообщение с информацией о режиме
            mode_info = []
            if st.session_state.use_multi_vector:
                mode_info.append("🚀 Multi-vector поиск")
            mode_info.append("🔍 Гибридный BM25+Vector поиск")
            mode_info.append("📊 Адаптивные веса")

            greeting = f"""👋 Здравствуйте! Я ваш улучшенный книжный помощник.

**Активные возможности:**
{chr(10).join(['• ' + mode for mode in mode_info])}

Задайте мне вопрос о книге, и я найду максимально релевантный ответ!"""

            st.session_state.messages.append({
                "role": "assistant",
                "content": greeting
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
    st.header("⚙️ Настройки поиска")

    # Секция режимов поиска
    st.subheader("🔍 Режимы поиска")

    # Переключатель multi-vector
    new_multi_vector = st.checkbox(
        "🚀 Multi-vector поиск",
        value=st.session_state.use_multi_vector,
        help="Включает создание нескольких векторов на документ для улучшения recall"
    )

    # Переключатель разделов
    new_use_sections = st.checkbox(
        "📖 Использовать разделы",
        value=st.session_state.use_sections,
        help="Загружает книгу по разделам для лучшей навигации"
    )

    # Проверяем, изменились ли настройки
    settings_changed = (
        new_multi_vector != st.session_state.use_multi_vector or
        new_use_sections != st.session_state.use_sections
    )

    if settings_changed:
        st.warning("⚠️ Настройки изменены. Нажмите 'Применить настройки' для перезапуска системы.")

        if st.button("🔄 Применить настройки", type="primary"):
            # Обновляем настройки
            st.session_state.use_multi_vector = new_multi_vector
            st.session_state.use_sections = new_use_sections

            # Удаляем старый RAG объект
            if 'rag' in st.session_state:
                del st.session_state.rag

            # Очищаем сообщения
            st.session_state.messages = []

            # Перезагружаем страницу
            st.rerun()

    # Информация о текущем режиме
    st.info(f"""
**Текущий режим:**
• Multi-vector: {'✅' if st.session_state.use_multi_vector else '❌'}
• Разделы: {'✅' if st.session_state.use_sections else '❌'}
• Гибридный поиск: ✅
• Адаптивные веса: ✅
    """)

    st.divider()
    st.subheader("📚 Навигация по книге")

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
        st.rerun()

    st.divider()

    # Информация о новых возможностях
    with st.expander("🚀 Новые возможности (Google Research)"):
        st.markdown("""
        **Проблема single-vector поиска:**
        При большой базе знаний обычный векторный поиск не может найти все релевантные документы из-за ограничений размерности.

        **Наши решения:**

        🔍 **Гибридный поиск** - комбинирует:
        • Dense vectors (семантическое понимание)
        • Sparse BM25 (точное совпадение ключевых слов)

        🚀 **Multi-vector** - создает:
        • 3 вектора на документ вместо 1
        • Лучшее покрытие семантического пространства

        📊 **Адаптивные веса** - автоматически:
        • Анализирует тип запроса
        • Корректирует важность каждого метода поиска

        📚 **Умный чанкинг** - создает:
        • Чанки разных размеров (600/1000/1500)
        • Умное перекрытие для лучшего контекста
        """)

    with st.expander("💡 Советы по использованию"):
        st.markdown("""
        **Для лучших результатов:**

        🎯 **Точные вопросы:**
        • "Что именно говорится о кризисе?"
        • "Конкретные шаги командообразования"

        🤔 **Концептуальные вопросы:**
        • "Как работает управление командой?"
        • "Почему важна структура бизнеса?"

        🔍 **Используйте multi-vector для:**
        • Сложных многоаспектных тем
        • Поиска скрытых связей
        • Улучшения полноты ответов
        """)

    # Статистика производительности
    if st.session_state.get('rag') and hasattr(st.session_state.rag, 'splits'):
        total_chunks = len(st.session_state.rag.splits) if st.session_state.rag.splits else 0
        st.metric("📊 Чанков в базе", total_chunks)
        if st.session_state.use_multi_vector and st.session_state.rag.multi_vector_embeddings:
            vectors_count = total_chunks * 3  # 3 вектора на чанк
            st.metric("🚀 Multi-векторов", vectors_count)

# Основной интерфейс
st.title("📚 Книжный помощник с RAG")

# Показываем активные режимы в основном интерфейсе
col1, col2, col3 = st.columns(3)

with col1:
    if st.session_state.use_multi_vector:
        st.success("🚀 Multi-vector активен")
    else:
        st.info("🔍 Single-vector режим")

with col2:
    st.success("📊 Адаптивные веса")

with col3:
    st.success("🔍 Гибридный BM25+Vector")

# Если выбран конкретный раздел, показываем информацию
if selected_section != "Вся книга":
    st.info(f"📖 Поиск в разделе: **{selected_section}**")

# Отображаем историю сообщений
for message_data in st.session_state.messages:
    message(
        message_data["content"],
        is_user=(message_data["role"] == "user"),
        is_error=message_data.get("is_error", False)
    )

# Поле ввода
if prompt := st.chat_input(f"Задайте вопрос о книге (максимум {MAX_QUESTION_LENGTH} символов)..."):
    # Валидация длины входного запроса для предотвращения промт-инжекшн
    if len(prompt) > MAX_QUESTION_LENGTH:
        st.error(f"❌ Вопрос слишком длинный ({len(prompt)} символов). Максимальная длина: {MAX_QUESTION_LENGTH} символов.")
        st.info("💡 Пожалуйста, сократите ваш вопрос.")
    else:
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
                if metadata:
                    full_response = f"{main_answer}\n\n**Метаданные ответа:**\n{metadata}"
                else:
                    full_response = main_answer

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
        **Быстрый старт:**
        1. 🚀 Включите Multi-vector для лучшего поиска
        2. 📖 Выберите раздел книги или "Вся книга"
        3. ❓ Задайте вопрос в поле внизу
        4. 📋 Получите улучшенный ответ с источниками

        **Примеры вопросов:**

        *Точные запросы:*
        - "Что именно говорится о ключевом факторе успеха?"
        - "Конкретные этапы командообразования"

        *Концептуальные запросы:*
        - "Как работает управление в кризисе?"
        - "Почему важна организационная структура?"

        **💡 Совет:** Multi-vector режим особенно эффективен для сложных вопросов!
        """)