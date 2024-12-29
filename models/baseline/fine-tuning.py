from models.baseline.train import CarsDataset, Config, CustomResNet18, DataLoader, load_checkpoint, train_loop

import pandas as pd
import torch

NUM_CLASSES = 10
IMG_SIZE = 224
BATCH_SIZE = 16


def fine_tune_model(model: torch.nn.Module,
                    checkpoint_path: str,
                    csv_path: str,
                    image_folder: str,
                    config: Config = None,
                    device='cuda'):
    """Функция для дообучения модели на пользовательских данных."""

    data = pd.read_csv(csv_path)
    data['file_path'] = data['image_name'].apply(lambda x: f"{image_folder}/{x}")
    data['type'] -= 1
    train_dataset = CarsDataset(data=data, mode='train', img_size=IMG_SIZE)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False
    )

    try:
        model.to(device)
        loaded_checkpoint = load_checkpoint(model, checkpoint_path=checkpoint_path)
        model = loaded_checkpoint['model']
        print("Загружен предыдущий чекпоинт для дообучения.")
        train_loop(
            model=model,
            train_loader=train_loader,
            val_loader=None,
            checkpoint=loaded_checkpoint,
            config=config
        )
        print("Дообучение завершено.")

    except FileNotFoundError:
        print("Чекпоинт не найден.")


if __name__ == '__main__':
    fine_tune_model(
        model=CustomResNet18(num_classes=NUM_CLASSES),
        csv_path='data.csv',
        image_folder=r"C:\Users\Max\Downloads\data\folder",
        checkpoint_path="../../data/best_checkpoint_val_p_0.8277_r_0.7873_f1_0.8043.pt"
    )
