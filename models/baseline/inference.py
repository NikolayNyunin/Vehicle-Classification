from models.baseline.train import CustomResNet18, NUM_CLASSES, IMG_SIZE, load_checkpoint

import numpy as np
import cv2
import torch
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

import os

TYPE_DICT = {0: 'minivan',
             2: 'sedan',
             5: 'fastback',
             3: 'hatchback',
             9: 'sports',
             1: 'SUV',
             8: 'convertible',
             6: 'estate',
             4: 'minibus',
             7: 'pickup'}


def inference_from_folder(folder_path, model, device='cuda'):
    """Инференс для всех изображений в указанной папке."""

    model.eval()
    model.to(device)
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            continue
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transform = Compose([
            Resize(height=IMG_SIZE, width=IMG_SIZE, interpolation=cv2.INTER_LINEAR),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255),
            ToTensorV2()
        ])
        transformed = transform(image=image)
        input_tensor = transformed["image"].unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_class].item()
        predicted_label = TYPE_DICT.get(pred_class, "Unknown")

        plt.figure(figsize=(8, 6))
        plt.imshow(image)
        plt.title(f"Predicted class: {predicted_label}\nConfidence: {confidence:.2f}")
        plt.axis('off')
        plt.show()


def inference_one_file(model: torch.nn.Module,
                       image_data: bytes,
                       device='cuda') -> dict:
    """Инференс для одного изображения."""

    model.eval()
    model.to(device)

    image_np = np.frombuffer(image_data, dtype=np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform = Compose([
        Resize(height=IMG_SIZE, width=IMG_SIZE, interpolation=cv2.INTER_LINEAR),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255),
        ToTensorV2()
    ])

    transformed = transform(image=image)
    input_tensor = transformed["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, predicted_class].item()
    predicted_label = TYPE_DICT.get(predicted_class, "Unknown")

    return {'class_id': predicted_class, 'class_name': predicted_label, 'confidence': confidence}


if __name__ == '__main__':
    checkpoint_path = "best_checkpoint.pt"
    folder_path = r"C:\Users\Max\Downloads"
    model = CustomResNet18(num_classes=NUM_CLASSES)
    loaded_data = load_checkpoint(model, checkpoint_path=checkpoint_path)
    model = loaded_data['model']
    inference_from_folder(folder_path, model)
