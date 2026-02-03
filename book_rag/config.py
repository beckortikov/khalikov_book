import os
import logging
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()

# Константы безопасности
MAX_QUESTION_LENGTH = 200

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag.log', mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("book_rag")

def get_openai_api_key():
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key
    return None

# Маппинг разделов книги к файлам
SECTIONS_MAPPING = {
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
