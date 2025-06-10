import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

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

#Функция для построения столбчатых графиков распределения по классам
def plot_counts(dfs, titles):
    fig, axes = plt.subplots(1, len(dfs), figsize=(16, 6), sharey=True)
    for df, title, ax in zip(dfs, titles, axes):
        counts = df['general_class_name'].value_counts().sort_index()
        ax.bar(counts.index, counts.values)
        ax.set_title(f'{title} distribution')
        ax.set_xlabel('class')
        ax.tick_params(axis='x', rotation=45)
    axes[0].set_ylabel('count')
    plt.tight_layout()
    plt.show()

#Функция для построения гистограмм ширины и высоты объектов
def plot_sizes(df, title):
    widths = df['boxx_2'] - df['boxx_1']
    heights = df['boxy_2'] - df['boxy_1']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.hist(widths, bins=30, edgecolor='black')
    ax1.set_title(f'{title} width distribution')
    ax1.set_xlabel('pixels')
    ax2.hist(heights, bins=30, edgecolor='black')
    ax2.set_title(f'{title} height distribution')
    ax2.set_xlabel('pixels')
    plt.tight_layout()
    plt.show()

#Функция для получения полного пути к изображению по имени файла
def get_image_path(filename, dataset='train', train_path=None, test_path=None):
    if dataset == 'train':
        return os.path.join(train_path, filename)
    else:
        return os.path.join(test_path, filename)

#Функция для отображения изображения с нарисованной рамкой и подписью
def show_image_with_annotation(idx, df, images_path):
    row = df.iloc[idx]
    img_path = os.path.join(images_path, row['file_name'])
    img = Image.open(img_path)
    draw = ImageDraw.Draw(img)
    bbox = (row['boxx_1'], row['boxy_1'], row['boxx_2'], row['boxy_2'])
    draw.rectangle(bbox, outline="red", width=3)
    draw.text((row['boxx_1'], row['boxy_1'] - 10),
              row['general_class_name'], fill="red")
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.axis('off')
    plt.show()
