# Описание собранных данных

## Ссылка на собранные данные: [Yandex Disk](https://disk.yandex.ru/d/gXOv80Fo-0QGQA)

## 1. Датасет: [CompCars dataset](https://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/index.html)

### Классы изображений транспортных средств: 12 -> 10
| Оригинальный класс  | Новый класс |
|:-------------------:|:-----------:|
|         MPV         |   minivan   |
|         SUV         |     SUV     |
|        sedan        |    sedan    |
|      hatchback      |  hatchback  |
|       minibus       |   minibus   |
|      fastback       |  fastback   |
|       estate        |   estate    |
|       pickup        |   pickup    |
| hardtop convertible | convertible |
|       sports        |   sports    |
|      crossover      |     SUV     |
|     convertible     | convertible |

### Дополнительное описание данных:
Помимо метки и названия класса, для каждого изображения из набора данных имеется bounding box, заданный в виде
`(x1, y1, x2, y2)` в пикселях и описывающий точное положение транспортного средства на фотографии.
Координаты удовлетворяют следующим свойствам: `1 <= x1 < x2 <= image_width` и `1 <= y1 < y2 <= image_height`.

## 2. Датасет: Stanford Cars Dataset

### Описание:
Набор данных Stanford Cars Dataset содержит 16185 изображений автомобилей 196 классов.
Данные разделены на 8144 обучающих изображения и 8041 тестовое изображение,
где каждый класс разделен примерно в соотношении 50 на 50.
Классы, как правило, указаны на уровне марки, модели, года выпуска, например,
Tesla Model S 2012 года выпуска или BMW M3 coupe 2012 года выпуска.

### Классы: 196 -> 9
- Convertible
- Coupe
- Hatchback
- Minivan
- Other
- SUV
- Sedan
- Truck
- Van

## 3. Датасет: [MIO-TCD Camera dataset](https://tcd.miovision.com/)
