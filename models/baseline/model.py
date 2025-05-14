import torch
from torch import nn
from torchvision.models import resnet18

CUDA = torch.cuda.is_available()
NUM_CLASSES = 10


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


def load_model(checkpoint_path: str) -> nn.Module:
    """Загрузка модели из чекпоинта."""

    model = CustomResNet18(num_classes=NUM_CLASSES)
    checkpoint = torch.load(
        checkpoint_path, map_location=torch.device('cuda' if CUDA else 'cpu')
    )
    model.load_state_dict(checkpoint['model_state_dict'])

    return model
