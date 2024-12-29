import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
from train import *

def inference_from_folder(folder_path, model, device='cuda'):
    """ Выполняет инференс для всех изображений в указанной папке. """
    type_dct = {0: 'minivan',
                2: 'sedan',
                5: 'fastback',
                3: 'hatchback',
                9: 'sports',
                1: 'SUV',
                8: 'convertible',
                6: 'estate',
                4: 'minibus',
                7: 'pickup'}
    model.eval()
    model.to(device)
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            continue
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transform = Compose([
            Resize(height=224, width=224, interpolation=cv2.INTER_LINEAR),
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
        predicted_label = type_dct.get(pred_class, "Unknown")

        plt.figure(figsize=(8, 6))
        plt.imshow(image)
        plt.title(f"Predicted class: {predicted_label}\nConfidence: {confidence:.2f}")
        plt.axis('off')
        plt.show()
checkpoint_path = "best_checkpoint_val_p_0.8277_r_0.7873_f1_0.8043.pt"
folder_path = r"C:\Users\Max\Downloads"
model = CustomResNet18(num_classes=NUM_CLASSES)
loaded_data = load_checkpoint(model, checkpoint_path=checkpoint_path)
model = loaded_data['model']
inference_from_folder(folder_path, model)