import requests
import argparse
import sys
import time
import json
import os

API_URL = "http://localhost:8000"  # Адрес API по умолчанию

def check_health():
    """Проверка статуса API"""
    try:
        response = requests.get(f"{API_URL}/health")
        response.raise_for_status()
        data = response.json()
        print(f"Статус API: {data['status']}")
        print(f"Сообщение: {data['message']}")
        return data['status'] == 'ok'
    except requests.RequestException as e:
        print(f"Ошибка при проверке статуса API: {str(e)}")
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

        ask_question(question)

def main():
    parser = argparse.ArgumentParser(description="Клиент для API книжного помощника")

    parser.add_argument('--url', default=API_URL, help=f"URL API (по умолчанию: {API_URL})")

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
    global API_URL
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