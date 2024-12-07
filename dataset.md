# Описание собранных данных

## Ветка 'nikolay':

### Использованный датасет: [CompCars dataset](https://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/index.html)

### Ссылка на собранные данные: [Yandex Disk](https://disk.yandex.ru/d/gXOv80Fo-0QGQA)

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
