import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchmetrics.classification import Precision, Recall, F1Score
from torchvision.models import resnet18

from typing import Callable, Type, Optional
from dataclasses import dataclass
from collections import Counter

CUDA = torch.cuda.is_available()

# if not train_on_gpu:
#     print('CUDA is not available')
# else:
#     print('CUDA is available!')

SEED = 42
NUM_CLASSES = 10
IMG_SIZE = 224
BATCH_SIZE = 256


class AlbumentationsTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        img = np.array(img)
        augmented = self.transform(image=img)
        return augmented['image']


class CarsDataset(Dataset):
    """
    Датасет с картинками, который параллельно подгружает их из папок,
    производит масштабирование и превращение в тензоры.
    """
    def __init__(self, data, mode, img_size):
        super().__init__()
        self.data = data
        self.mode = mode
        self.img_size = img_size

    def __len__(self):
        return len(self.data)

    def test_transforms(self):
        transforms = A.Compose([
            A.Resize(height=self.img_size, width=self.img_size, interpolation=cv2.INTER_LINEAR),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255),
            ToTensorV2()
        ])
        return transforms

    def train_transforms(self):
        transforms = A.Compose([
            A.Resize(height=self.img_size, width=self.img_size, interpolation=cv2.INTER_LINEAR),

            # Горизонтальное отражение
            A.HorizontalFlip(p=0.3),

            # Вращение от -40 до +40 градусов
            A.Rotate(limit=40, p=0.3),

            # Случайное изменение яркости и контраста
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),

            # Эффект размытия
            A.Blur(blur_limit=(3, 7), p=0.3),

            # Случайное зануление прямоугольных областей
            A.CoarseDropout(max_holes=3, max_height=40, max_width=40, p=0.3),

            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255),
            ToTensorV2()
        ])

        return transforms

    def __getitem__(self, index):
        row = self.data.iloc[index]
        img_path = row["file_path"]
        label = row["type"]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.mode == 'train':
            transforms = self.train_transforms()
            transform = AlbumentationsTransform(transform=transforms)
            image = transform(image)
        else:
            transforms = self.test_transforms()
            transform = AlbumentationsTransform(transform=transforms)
            image = transform(image)
        return image, torch.tensor(label, dtype=torch.long)


def make_weighted_loader(dataset, labels, batch_size, drop_last):
    """Функция для сэмплирования с учетом дисбаланса классов."""

    label_counts = Counter(labels)
    label_weights = {label: 1.0 / count for label, count in label_counts.items()}
    sample_weights = np.array([label_weights[label] for label in labels])
    num_samples = 10 * max(label_counts.values())
    sampler = WeightedRandomSampler(sample_weights, num_samples=num_samples, replacement=True)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=drop_last
    )
    return train_loader


def save_checkpoint(
    model,
    optimizer,
    scheduler=None,
    step=None,
    best_f1=None,
    config=None,
    checkpoint_path='checkpoint.pt'
):
    """Сохранение чекпоинта обучения."""

    checkpoint = {
        'optimizer_class': optimizer.__class__.__name__,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'step': step,
        'best_f1': best_f1,
        'config': Config,
    }
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(
    model,
    checkpoint_path='checkpoint.pt'
):
    """Загрузка чекпоинта обучения."""

    checkpoint = torch.load(checkpoint_path)
    config = checkpoint.get('config', None)
    print(config)

    if config is None:
        raise ValueError("Config not found in the checkpoint. Please provide a valid checkpoint.")

    optimizer_class_name = checkpoint.get('optimizer_class', None)
    optimizer_class = getattr(optim, optimizer_class_name)
    optimizer = optimizer_class(model.parameters(), lr=config.lr)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    scheduler = None
    if config.scheduler:
        scheduler = config.scheduler(
            optimizer, 
            step_size=config.scheduler_step_size, 
            gamma=config.scheduler_gamma
        )

        # Загрузка состояния планировщика, если оно есть в чекпоинте
        if checkpoint['scheduler_state_dict'] is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return {
        'model': model,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'step': checkpoint.get('step', 0),
        'best_f1': checkpoint.get('best_f1', None),
        'config': config,
    }


@dataclass
class Config:
    """Конфигурация."""

    # Общие параметры
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name: Optional[str] = 'default_name'

    # Обучение
    batch_size: int = 256
    n_epochs: int = 30
    eval_every: int = 100
    lr: float = 1e-3

    optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam
    loss_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = nn.CrossEntropyLoss()

    scheduler: Optional[Type[torch.optim.lr_scheduler._LRScheduler]] = torch.optim.lr_scheduler.StepLR
    scheduler_step_size: int = 1
    scheduler_gamma: float = 0.5


def train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Config | None = None,
    checkpoint: dict | None = None
):
    """Обучение модели."""

    if config is None and checkpoint is None:
        raise ValueError("Config cannot be None when initializing without a checkpoint!")

    if checkpoint is not None:
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        scheduler = checkpoint['scheduler']
        step = checkpoint['step']
        best_f1 = checkpoint['best_f1']
        config = checkpoint['config']
    else:
        optimizer = config.optimizer(model.parameters(), lr=config.lr)
        scheduler_step_size = config.scheduler_step_size
        scheduler_gamma = config.scheduler_gamma
        scheduler = config.scheduler(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
        step = 0
        best_f1 = -float('inf')

    loss_func = config.loss_func
    device = config.device

    # Метрики
    precision_metric = Precision(task='multiclass', num_classes=NUM_CLASSES, average='macro').to(device)
    recall_metric = Recall(task='multiclass', num_classes=NUM_CLASSES, average='macro').to(device)
    f1_metric = F1Score(task='multiclass', num_classes=NUM_CLASSES, average='macro').to(device)

    model.to(device)
    # Отправляем оптимизатор на устройство
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    model.train()
    for epoch in range(config.n_epochs):
        print(f"Epoch #{epoch + 1}/#{config.n_epochs}")

        for i, (img_batch, true_labels) in enumerate(tqdm(train_loader)):
            step += 1
            img_batch, true_labels = img_batch.to(device), true_labels.to(device)
            true_labels = true_labels.to(torch.long)
            
            optimizer.zero_grad()
            pred_labels = model(img_batch)
            loss_train = loss_func(pred_labels, true_labels)
            loss_train.backward()
            optimizer.step()

            if (i + 1) % config.eval_every == 0:
                # Метрики на валидации
                model.eval()
                val_loss_total = 0.0
                precision_metric.reset()
                recall_metric.reset()
                f1_metric.reset()

                with torch.no_grad():
                    for j, (img_batch_val, true_labels_val) in enumerate(val_loader):
                        img_batch_val, true_labels_val = img_batch_val.to(device), true_labels_val.to(device)
                        outputs = model(img_batch_val)
                        loss_val = loss_func(outputs, true_labels_val)
                        val_loss_total += loss_val.item()

                        # Обновляем метрики
                        predictions = torch.argmax(outputs, dim=1)
                        precision_metric.update(predictions, true_labels_val)
                        recall_metric.update(predictions, true_labels_val)
                        f1_metric.update(predictions, true_labels_val)

                torch.cuda.empty_cache()

                avg_val_loss = val_loss_total / len(val_loader)
                precision = precision_metric.compute().item()
                recall = recall_metric.compute().item()
                f1 = f1_metric.compute().item()

                print(f"Step {step}: Val Loss={avg_val_loss:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
                if f1 > best_f1:
                    best_f1 = f1
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        step=step,
                        best_f1=best_f1,
                        config=vars(config),
                        checkpoint_path=f"best_checkpoint_val_p_{precision:.4f}_r_{recall:.4f}_f1_{f1:.4f}.pt"
                    )
                    print(f"New best model saved with F1={f1:.4f}")
                model.train()

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                step=step,
                best_f1=best_f1,
                config=vars(config),
                checkpoint_path='cur_checkpoint.pt'
            )
        if scheduler is not None:
            scheduler.step()
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            step=step,
            best_f1=best_f1,
            config=vars(config),
            checkpoint_path='epoch_checkpoint.pt'
        )


class CustomResNet18(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        self.resnet = resnet18(weights='IMAGENET1K_V1')
        # Убрали последний линейный слой
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        # Сказали, что замораживаем всю сеть
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Размораживаем последний sequential block
        for param in list(self.resnet[-2].parameters()):
            param.requires_grad = True

        self.flatten = nn.Flatten()
        self.out = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.resnet(x)
        x = self.flatten(x)
        x = self.out(x)
        return x


def load_model(checkpoint_path: str) -> torch.nn.Module:
    """Загрузка модели из чекпоинта."""

    model = CustomResNet18(num_classes=NUM_CLASSES)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda' if CUDA else 'cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])

    return model


if __name__ == '__main__':
    data = pd.read_csv('data.csv', index_col=0).reset_index(drop=True)
    data['type'] -= 1

    train_val_df, test_df = train_test_split(
        data,
        test_size=0.2,
        stratify=data['type'],
        random_state=SEED
    )

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=0.2,
        stratify=train_val_df['type'],
        random_state=SEED
    )

    print(f"Train size: {len(train_df)}")
    print(f"Validation size: {len(val_df)}")
    print(f"Test size: {len(test_df)}")

    dataset = CarsDataset(data=data, mode='train', img_size=IMG_SIZE)

    train_dataset = CarsDataset(data=train_df, mode='train', img_size=IMG_SIZE)
    val_dataset = CarsDataset(data=val_df, mode='val', img_size=IMG_SIZE)
    test_dataset = CarsDataset(data=test_df, mode='test', img_size=IMG_SIZE)
    train_labels = train_dataset.data['type']

    print(f'Размер датасета train: {len(train_dataset)}')
    print(f'Размер датасета test: {len(test_dataset)}')
    print(f'Размер датасета val: {len(val_dataset)}')

    train_loader = make_weighted_loader(
        train_dataset,
        train_labels,
        batch_size=BATCH_SIZE,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=True
    )

    model = CustomResNet18(num_classes=NUM_CLASSES)
    print(model)

    config_res18_last_seq = Config(
        model_name = 'try_start_train',
        batch_size = 256,
        n_epochs = 2,
        eval_every = 200,
        lr = 1e-3
    )

    # Инициализация модели
    model = CustomResNet18(num_classes=NUM_CLASSES)

    # Запуск обучения с нуля
    res = train_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config_res18_last_seq
    )

    # Продолжение обучения
    model = CustomResNet18(num_classes=NUM_CLASSES)
    checkpoint = load_checkpoint(model, checkpoint_path='cur_checkpoint.pt')

    # Запуск дообучения
    res = train_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint=checkpoint
    )

    device = torch.device("cuda" if CUDA else "cpu")
    model = CustomResNet18(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load('best_model_val_p_0.7876_r_0,7865_f1_0,7856.pt'))
    model.to(device)
    model.eval()

    f1_metric = F1Score(task='multiclass', num_classes=NUM_CLASSES, average='macro').to(device)

    with torch.no_grad():
        for i, (img_batch_val, true_labels_val) in enumerate(test_loader):
            img_batch_val, true_labels_val = img_batch_val.to(device), true_labels_val.to(device)
            outputs = model(img_batch_val)
            predictions = torch.argmax(outputs, dim=1)
            f1_metric.update(predictions, true_labels_val)

    final_f1 = f1_metric.compute().item()
    print(f"Final Average F1 Score: {final_f1:.4f}")
