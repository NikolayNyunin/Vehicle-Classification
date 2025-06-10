# bot.py
import logging
import os
import requests
import cv2
from telegram import Update
from telegram.ext import (
    Updater,
    MessageHandler,
    Filters,
    CallbackContext,
    CommandHandler
)

TOKEN = "TOKEN"

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

WELCOME_MESSAGE = (
    "Привет! "
    "Я бот для обнаружения и классификации автомобилей по типу кузова.\n\n"
    "Я могу определить 9 типов машин:\n"
    "- Sedan (Седан)\n"
    "- SUV (Внедорожник)\n"
    "- Coupe (Купе)\n"
    "- Convertible (Кабриолет)\n"
    "- Hatchback (Хэтчбек)\n"
    "- Minivan (Минивэн)\n"
    "- Van (Автобус)\n"
    "- Truck (Грузовик)\n"
    "- Other (Другой тип транспорта)\n\n"
)

COMMANDS_MESSAGE = (
    "Доступные команды:\n\n"
    "- /models: Показать доступные модели (на сервере)\n"
    "- /set <имя_модели>: Загрузить выбранную модель\n"
    "- /predict: Переключиться в режим обработки фото\n"
    "- /stop: Остановить обработку и сбросить настройки\n"
)

WAITING_MESSAGE = "Ожидаю фотографию автомобиля..."
STOPPED_MESSAGE = "Бот остановлен. Фото не обрабатываются."
NOT_PREDICT_MESSAGE = "Вы не в режиме /predict. Фото не обрабатывается."

SERVER_URL = "http://127.0.0.1:5000"  # адрес нашего FastAPI-сервера


def start_command(update: Update, context: CallbackContext):
    """Отправляет приветствие, список команд,
    и третьим сообщением – текущая модель."""
    update.message.reply_text(WELCOME_MESSAGE)
    update.message.reply_text(COMMANDS_MESSAGE)

    try:
        resp = requests.get(f"{SERVER_URL}/current")
        resp.raise_for_status()
        data = resp.json()
        current_model = data.get("current_model", "неизвестно")
        update.message.reply_text(
            f"Сейчас используется модель: {current_model}"
        )
    except Exception as e:
        update.message.reply_text(f"Не удалось узнать текущую модель: {e}")

    context.user_data.clear()


def predict_command(update: Update, context: CallbackContext):
    context.user_data["predict_mode"] = True
    update.message.reply_text(WAITING_MESSAGE)


def stop_command(update: Update, context: CallbackContext):
    context.user_data["predict_mode"] = False
    update.message.reply_text(STOPPED_MESSAGE)


def models_command(update: Update, context: CallbackContext):
    """GET /models -> список моделей + текущая."""
    try:
        resp = requests.get(f"{SERVER_URL}/models")
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            update.message.reply_text(f"Ошибка: {data['error']}")
            return

        pt_files = data.get("models", [])
        current_model = data.get("current", "неизвестно")

        if not pt_files:
            update.message.reply_text(
                "На сервере не найдено моделей (.pt-файлов)."
            )
        else:
            msg = "Доступные модели:\n"
            for m in pt_files:
                if m == current_model:
                    msg += f"- {m} (ТЕКУЩАЯ)\n"
                else:
                    msg += f"- {m}\n"
            update.message.reply_text(msg)
    except requests.RequestException as e:
        update.message.reply_text(f"Ошибка при запросе /models:\n{e}")


def set_command(update: Update, context: CallbackContext):
    """
    /set <имя_файла.pt> -> POST /set?model=<имя>.
    """
    text_parts = update.message.text.strip().split()
    if len(text_parts) < 2:
        update.message.reply_text("Использование: /set <имя_модели.pt>")
        return

    desired_model = text_parts[1]

    try:
        url = f"{SERVER_URL}/set"
        resp = requests.post(url, params={"model": desired_model})
        if resp.status_code == 200:
            data = resp.json()
            if "status" in data:
                update.message.reply_text(data["status"])
            else:
                update.message.reply_text(f"Ответ сервера: {data}")
        else:
            update.message.reply_text(
                f"Ошибка при загрузке модели: {resp.status_code}\n{resp.text}"
            )
    except requests.RequestException as e:
        update.message.reply_text(f"Ошибка при запросе /set:\n{e}")


def call_inference_api(image_path: str):
    """Шлём POST /predict с файлом."""
    with open(image_path, "rb") as f:
        resp = requests.post(f"{SERVER_URL}/predict", files={"image": f})
    resp.raise_for_status()
    return resp.json()


def draw_boxes_opencv(image_path: str, boxes: list):
    image = cv2.imread(image_path)
    for box in boxes:
        x1 = int(box["x1"])
        y1 = int(box["y1"])
        x2 = int(box["x2"])
        y2 = int(box["y2"])
        label_text = box["label"]

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            image,
            label_text,
            (x1, max(y1 - 5, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

    annotated_path = os.path.splitext(image_path)[0] + "_annotated.jpg"
    cv2.imwrite(annotated_path, image)
    return annotated_path


def handle_photo(update: Update, context: CallbackContext):
    if not context.user_data.get("predict_mode", False):
        update.message.reply_text(NOT_PREDICT_MESSAGE)
        return

    photo = update.message.photo[-1]
    local_image_path = "temp.jpg"
    photo_file = photo.get_file()
    photo_file.download(custom_path=local_image_path)

    try:
        result_json = call_inference_api(local_image_path)
        boxes = result_json.get("boxes", [])
        annotated_path = draw_boxes_opencv(local_image_path, boxes)

        with open(annotated_path, "rb") as f:
            update.message.reply_photo(photo=f)
    except requests.RequestException as e:
        update.message.reply_text(f"Ошибка при запросе /predict:\n{e}")


def main():
    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start_command))
    dp.add_handler(CommandHandler("predict", predict_command))
    dp.add_handler(CommandHandler("stop", stop_command))
    dp.add_handler(CommandHandler("models", models_command))
    dp.add_handler(CommandHandler("set", set_command))

    dp.add_handler(MessageHandler(Filters.photo, handle_photo))

    updater.start_polling()
    updater.idle()


if __name__ == "__main__":
    main()
