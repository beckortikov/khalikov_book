from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from pydantic import BaseModel
from main import BookRAG
import os
import logging
import uvicorn
import shutil
from typing import Optional, List, Dict, Any
from fastapi.middleware.cors import CORSMiddleware

# Настройка логирования
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log', mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Книжный помощник API",
              description="API для получения ответов на вопросы по книге с использованием RAG",
              version="1.0.0")

# Добавляем CORS middleware для разрешения запросов с разных источников
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене рекомендуется указать конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Модели данных
class QuestionRequest(BaseModel):
    question: str
    section_filter: Optional[str] = None

class SectionSearchRequest(BaseModel):
    section_name: str
    query: Optional[str] = ""

class QuestionResponse(BaseModel):
    answer: str

class SectionsResponse(BaseModel):
    sections: List[str]

class StatusResponse(BaseModel):
    status: str
    message: str

# Глобальная переменная для хранения инициализированного RAG
rag_instance = None
# Флаг, указывающий, что система в процессе инициализации
initializing = False

@app.on_event("startup")
async def startup_event():
    """Инициализация RAG при запуске сервера"""
    global rag_instance, initializing

    initializing = True

    try:
        logger.info("Инициализация RAG при запуске сервера...")
        # Используем новую систему с поддержкой разделов
        rag_instance = BookRAG(use_sections=True, pdf_directory="book")
        logger.info("Система успешно инициализирована!")
    except FileNotFoundError as e:
        logger.error(f"Файл не найден: {str(e)}", exc_info=True)
        # Сервер запустится, но API будет возвращать ошибки до тех пор, пока RAG не будет инициализирован
    except Exception as e:
        logger.error(f"Ошибка при инициализации системы: {str(e)}", exc_info=True)
        # Сервер запустится, но API будет возвращать ошибки до тех пор, пока RAG не будет инициализирован
    finally:
        initializing = False

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Конечная точка для ответа на вопросы о книге"""
    global rag_instance

    if rag_instance is None:
        logger.error("RAG не инициализирован")
        raise HTTPException(status_code=503, detail="Система не готова. Пожалуйста, повторите запрос позже.")

    try:
        logger.debug(f"Получен вопрос: {request.question}")
        if request.section_filter:
            logger.debug(f"Фильтр по разделу: {request.section_filter}")

        # Проверка релевантности вопроса
        if not is_book_related_question(request.question):
            logger.info(f"Получен нерелевантный вопрос: {request.question}")
            return {"answer": "Я могу отвечать только на вопросы о содержании книги. Пожалуйста, задайте вопрос, связанный с текстом книги."}

        # Получаем ответ от RAG с учетом фильтра по разделу
        response = rag_instance.ask_question(request.question, section_filter=request.section_filter)
        logger.info("Успешно получен ответ на вопрос")

        return {"answer": response}
    except Exception as e:
        error_msg = f"Ошибка при обработке вопроса: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)

def is_book_related_question(question: str) -> bool:
    """
    Проверяет, относится ли вопрос к содержанию книги.

    Функция анализирует вопрос и определяет, связан ли он с книгой
    или это нерелевантный запрос (например, просьба написать код,
    рецепт, выполнить задачу, не связанную с содержанием книги).

    Args:
        question: Текст вопроса

    Returns:
        bool: True если вопрос относится к книге, False в противном случае
    """
    # Приводим вопрос к нижнему регистру для упрощения проверки
    question_lower = question.lower()

    # Ключевые слова, указывающие на нерелевантные запросы
    irrelevant_patterns = [
        "напиши код", "напиши программу", "создай программу",
        "сделай", "приготовь", "рецепт", "как готовить",
        "погода", "курс валют", "прогноз погоды",
        "как построить", "создай сайт", "напиши сайт",
        "напиши песню", "напиши стих", "напиши рассказ",
        "какой сегодня день", "который час"
    ]

    # Проверяем наличие нерелевантных паттернов в вопросе
    for pattern in irrelevant_patterns:
        if pattern in question_lower:
            return False

    # По умолчанию считаем вопрос релевантным
    return True

@app.post("/upload", response_model=StatusResponse)
async def upload_book(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Загрузка новой книги и реинициализация RAG"""
    global rag_instance, initializing

    # Проверка, что файл это PDF
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Только PDF файлы разрешены")

    if initializing:
        raise HTTPException(status_code=409, detail="Система уже находится в процессе инициализации")

    try:
        # Сохраняем файл
        file_path = "book.pdf"  # Используем стандартное имя файла

        # Создаем временный файл и затем перемещаем его
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"Файл {file.filename} успешно загружен и сохранен как {file_path}")

        # Запускаем реинициализацию в фоне
        background_tasks.add_task(reinitialize_rag, file_path)

        return {"status": "processing", "message": "Файл успешно загружен. Началась инициализация системы."}

    except Exception as e:
        error_msg = f"Ошибка при загрузке файла: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)

async def reinitialize_rag(file_path: str):
    """Функция для реинициализации RAG в фоновом режиме"""
    global rag_instance, initializing

    initializing = True

    try:
        logger.info(f"Начало реинициализации RAG с новым файлом: {file_path}")

        # Создаем новый экземпляр BookRAG с новой системой
        # Если загружен новый файл, используем его как основную книгу
        if file_path == "book.pdf":
            # Копируем в папку book для консистентности
            if not os.path.exists("book"):
                os.makedirs("book")
            if not os.path.exists("book/book.pdf"):
                shutil.copy(file_path, "book/book.pdf")

        new_rag = BookRAG(use_sections=True, pdf_directory="book")

        # Заменяем старый экземпляр новым
        rag_instance = new_rag

        logger.info("Система успешно реинициализирована с новым файлом")

    except Exception as e:
        logger.error(f"Ошибка при реинициализации системы: {str(e)}", exc_info=True)

    finally:
        initializing = False

@app.get("/health", response_model=StatusResponse)
async def health_check():
    """Проверка работоспособности API"""
    global rag_instance, initializing

    if initializing:
        return {"status": "initializing", "message": "RAG система инициализируется"}

    if rag_instance is None:
        return {"status": "not_ready", "message": "RAG система не инициализирована"}

    return {"status": "ok", "message": "Система работает нормально"}

@app.post("/search_section", response_model=QuestionResponse)
async def search_in_section(request: SectionSearchRequest):
    """Поиск в конкретном разделе книги"""
    global rag_instance

    if rag_instance is None:
        logger.error("RAG не инициализирован")
        raise HTTPException(status_code=503, detail="Система не готова. Пожалуйста, повторите запрос позже.")

    try:
        logger.debug(f"Поиск в разделе: {request.section_name}, запрос: {request.query}")

        response = rag_instance.search_by_section(request.section_name, request.query)
        logger.info(f"Успешно выполнен поиск в разделе {request.section_name}")

        return {"answer": response}
    except Exception as e:
        error_msg = f"Ошибка при поиске в разделе: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/sections", response_model=SectionsResponse)
async def get_sections():
    """Получение списка доступных разделов книги"""
    global rag_instance

    if rag_instance is None:
        logger.error("RAG не инициализирован")
        raise HTTPException(status_code=503, detail="Система не готова. Пожалуйста, повторите запрос позже.")

    try:
        sections = rag_instance.get_available_sections()
        logger.info("Успешно получен список разделов")
        return {"sections": sections}
    except Exception as e:
        error_msg = f"Ошибка при получении списка разделов: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/rebuild_embeddings", response_model=StatusResponse)
async def rebuild_embeddings(background_tasks: BackgroundTasks):
    """Принудительное пересоздание эмбеддингов"""
    global rag_instance, initializing

    if rag_instance is None:
        logger.error("RAG не инициализирован")
        raise HTTPException(status_code=503, detail="Система не готова. Пожалуйста, повторите запрос позже.")

    if initializing:
        raise HTTPException(status_code=409, detail="Система уже находится в процессе инициализации")

    try:
        logger.info("Запуск принудительного пересоздания эмбеддингов...")

        # Запускаем пересоздание в фоне
        background_tasks.add_task(force_rebuild_embeddings_task)

        return {"status": "processing", "message": "Началось пересоздание эмбеддингов. Это может занять несколько минут."}

    except Exception as e:
        error_msg = f"Ошибка при запуске пересоздания эмбеддингов: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)

async def force_rebuild_embeddings_task():
    """Фоновая задача для пересоздания эмбеддингов"""
    global rag_instance, initializing

    initializing = True

    try:
        logger.info("Начало принудительного пересоздания эмбеддингов...")

        if rag_instance:
            success = rag_instance.force_rebuild_embeddings()
            if success:
                logger.info("Эмбеддинги успешно пересозданы")
            else:
                logger.error("Ошибка при пересоздании эмбеддингов")
        else:
            logger.error("RAG instance не найден")

    except Exception as e:
        logger.error(f"Критическая ошибка при пересоздании эмбеддингов: {str(e)}", exc_info=True)

    finally:
        initializing = False

if __name__ == "__main__":
    uvicorn.run("api:app", host="localhost", port=8001, reload=False)