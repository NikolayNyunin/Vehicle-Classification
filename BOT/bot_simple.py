import logging
import os
import cv2
import torch
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    MessageHandler,
    ContextTypes,
    filters
)

TOKEN = "TOKEN"

model = torch.hub.load(
    './yolov5',
    'custom',
    path='yolov5/model_3_ft.82.pt',
    source='local'
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)


def draw_boxes_opencv(image_path, results):
    """Draw bounding boxes using OpenCV (for YOLOv5)."""
    image = cv2.imread(image_path)

    for x1, y1, x2, y2, conf, cls in results.xyxy[0].cpu().numpy():
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        label_text = model.names[int(cls)]

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


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles incoming photo, runs YOLO inference,
    and sends back annotated image."""
    if not update.message.photo:
        return

    photo = update.message.photo[-1]
    file_id = photo.file_id
    new_file = await context.bot.get_file(file_id)
    local_image_path = "temp.jpg"
    await new_file.download_to_drive(local_image_path)

    results = model(local_image_path)
    annotated_path = draw_boxes_opencv(local_image_path, results)

    await update.message.reply_photo(photo=open(annotated_path, "rb"))


def main():
    """Start the bot using ApplicationBuilder (python-telegram-bot v20+)."""
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    app.run_polling()


if __name__ == "__main__":
    main()
