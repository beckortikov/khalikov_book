import requests
import argparse
import sys
import time
import json
import os

# Константы
API_URL = "http://localhost:8000"
MAX_QUESTION_LENGTH = 200

def check_health():
    """Проверка статуса API"""
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            print("✅ API готово к работе")
            return True
        else:
            print(f"❌ API недоступно. Статус: {response.status_code}")
            return False
    except requests.RequestException as e:
        print(f"❌ Ошибка подключения к API: {str(e)}")
        return False

def upload_book(file_path):
    """Загрузка новой книги"""
    if not os.path.exists(file_path):
        print(f"Файл {file_path} не найден")
        return False

    try:
        with open(file_path, 'rb') as file:
            files = {'file': (os.path.basename(file_path), file, 'application/pdf')}
            response = requests.post(f"{API_URL}/upload", files=files)
            response.raise_for_status()
            data = response.json()
            print(f"Статус загрузки: {data['status']}")
            print(f"Сообщение: {data['message']}")
            return True
    except requests.RequestException as e:
        print(f"Ошибка при загрузке книги: {str(e)}")
        if hasattr(e, 'response') and e.response:
            try:
                error_data = e.response.json()
                print(f"Детали ошибки: {error_data.get('detail', 'Нет деталей')}")
            except ValueError:
                print(f"Статус код: {e.response.status_code}")
        return False

def ask_question(question):
    """Задать вопрос API"""
    try:
        # Валидация длины входного запроса для предотвращения промт-инжекшн
        if len(question) > MAX_QUESTION_LENGTH:
            print("❌ Ошибка: Вопрос слишком длинный. Максимальная длина вопроса - 200 символов.")
            print(f"   Текущая длина: {len(question)} символов")
            return False

        # Проверка на пустой запрос
        if not question.strip():
            print("❌ Ошибка: Вопрос не может быть пустым.")
            return False

        response = requests.post(
            f"{API_URL}/ask",
            json={"question": question}
        )
        response.raise_for_status()
        data = response.json()
        print("\nОтвет на ваш вопрос:")
        print("-" * 80)
        print(data['answer'])
        print("-" * 80)

        # Проверка на нерелевантный вопрос
        if "Я могу отвечать только на вопросы о содержании книги" in data['answer']:
            print("\nПодсказка: Пожалуйста, задавайте вопросы, связанные с содержанием загруженной книги.")

        return True
    except requests.RequestException as e:
        print(f"Ошибка при отправке вопроса: {str(e)}")
        if hasattr(e, 'response') and e.response:
            try:
                error_data = e.response.json()
                print(f"Детали ошибки: {error_data.get('detail', 'Нет деталей')}")
            except ValueError:
                print(f"Статус код: {e.response.status_code}")
        return False

def interactive_mode():
    """Интерактивный режим для задания вопросов"""
    print("Интерактивный режим (для выхода введите 'exit' или 'quit')")
    print("⚠️  Максимальная длина вопроса: 200 символов")

    # Проверяем статус API
    if not check_health():
        print("API не готово или недоступно. Попробуйте позже.")
        return

    while True:
        question = input("\nВведите ваш вопрос о книге: ")
        if question.lower() in ('exit', 'quit'):
            break

        if not question.strip():
            continue

        # Валидация длины в интерактивном режиме
        if len(question) > MAX_QUESTION_LENGTH:
            print(f"❌ Вопрос слишком длинный ({len(question)} символов). Максимум: 200 символов.")
            print("   Пожалуйста, сократите ваш вопрос.")
            continue

        ask_question(question)

def main():

    global API_URL

    parser = argparse.ArgumentParser(description="Клиент для API книжного помощника")

    parser.add_argument('--url', default=None, help=f"URL API (по умолчанию: {API_URL})")

    subparsers = parser.add_subparsers(dest='command', help='Команды')

    # Проверка статуса
    subparsers.add_parser('status', help='Проверить статус API')

    # Загрузка книги
    upload_parser = subparsers.add_parser('upload', help='Загрузить новую книгу')
    upload_parser.add_argument('file', help='Путь к PDF файлу')

    # Задать вопрос
    ask_parser = subparsers.add_parser('ask', help='Задать один вопрос')
    ask_parser.add_argument('question', help='Ваш вопрос о книге')

    # Интерактивный режим
    subparsers.add_parser('interactive', help='Интерактивный режим для задания вопросов')

    args = parser.parse_args()

    # Обновляем URL API, если указан
    if args.url:
        API_URL = args.url

    if args.command == 'status':
        check_health()
    elif args.command == 'upload':
        upload_book(args.file)
    elif args.command == 'ask':
        if check_health():
            ask_question(args.question)
    elif args.command == 'interactive':
        interactive_mode()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()