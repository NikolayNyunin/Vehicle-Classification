import os
import pandas as pd
from PIL import Image

#Faster RCNN
#Функция для маппинга широкого спектра категорий на более общие
def map_to_general_category(class_name, general_categories):
    for category, keywords in general_categories.items():
        if any(keyword in class_name for keyword in keywords):
            return category
    return 'Other'

#Функция для загрузки и обработки таблицы с разметкой (CSV → DataFrame)
def load_and_process(path, general_categories):
    df = pd.read_csv(path)
    df.rename(columns={
        'x1': 'boxx_1',
        'y1': 'boxy_1',
        'x2': 'boxx_2',
        'y2': 'boxy_2',
        'image': 'file_name',
        'ture_class_name': 'class_name'
    }, inplace=True)
    df['general_class_name'] = df['class_name'].apply(lambda x: map_to_general_category(x, general_categories))
    df['general_class'] = pd.factorize(df['general_class_name'])[0] + 1
    return df

#YOLO
#Функция для преобразования координат в YOLO формат
def convert_to_yolo_format(row, image_width, image_height):
    x_center = (row['boxx_1'] + row['boxx_2']) / 2 / image_width
    y_center = (row['boxy_1'] + row['boxy_2']) / 2 / image_height
    width = (row['boxx_2'] - row['boxx_1']) / image_width
    height = (row['boxy_2'] - row['boxy_1']) / image_height
    return f"{class_mappings[row['general_class_name']]} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

#Функция для обработки CSV и сохранения .txt файлов
def process_csv_to_yolo(csv_path, labels_dir, images_dir):
    df = pd.read_csv(csv_path)
    df['general_class_name'] = df['class_name'].apply(map_to_general_category)
    
    for _, row in df.iterrows():
        image_name = row['file_name']
        image_path = os.path.join(images_dir, image_name)

        with Image.open(image_path) as img:
            image_width, image_height = img.size

        yolo_annotation = convert_to_yolo_format(row, image_width, image_height)
        label_file = os.path.join(labels_dir, image_name.replace('.jpg', '.txt'))

        with open(label_file, 'w') as f:
            f.write(yolo_annotation + '\n')

# Function to validate class IDs in YOLO label files
def validate_label_files(labels_dir, num_classes):
    invalid_files = []
    for label_file in os.listdir(labels_dir):
        if label_file.endswith('.txt'):
            with open(os.path.join(labels_dir, label_file), 'r') as f:
                for line in f:
                    class_id = int(line.split()[0])  # Extract class ID
                    if class_id >= num_classes:  # Check if class ID is out of range
                        invalid_files.append((label_file, class_id))
    return invalid_files
