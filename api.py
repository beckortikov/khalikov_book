from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from pydantic import BaseModel
from main import BookRAG
import os
import logging
import uvicorn
from typing import Optional, List, Dict, Any
from fastapi.middleware.cors import CORSMiddleware
import shutil

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

class QuestionResponse(BaseModel):
    answer: str

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
        rag_instance = BookRAG()  # Используем файл по умолчанию (book.pdf)
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

        # Получаем ответ от RAG
        response = rag_instance.ask_question(request.question)
        logger.info("Успешно получен ответ на вопрос")

        return {"answer": response}
    except Exception as e:
        error_msg = f"Ошибка при обработке вопроса: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)

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

        # Создаем новый экземпляр BookRAG с новым файлом
        new_rag = BookRAG(pdf_path=file_path)

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

if __name__ == "__main__":
    uvicorn.run("api:app", host="localhost", port=8001, reload=False)