# server.py
import os
import torch
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
import uvicorn

app = FastAPI()

# Храним имя текущей модели в переменной:
current_model_name = "model_3_f1.82.pt"

# Грузим модель по умолчанию
model = torch.hub.load(
    './yolov5',
    'custom',
    path=f"available_models/{current_model_name}",
    source='local'
)


@app.get("/models")
def list_models():
    models_dir = "available_models"
    if not os.path.isdir(models_dir):
        return {"error": "Папка available_models не найдена."}
    pt_files = [f for f in os.listdir(models_dir) if f.endswith(".pt")]
    if not pt_files:
        return {"error": "В папке нет моделей (.pt-файлов)."}
    return {"models": pt_files, "current": current_model_name}


@app.post("/set")
def set_model(
    model_param: str = Query(
        ..., alias="model",
        description="Имя pt-файла, например model_1_f1.29.pt"
    )
):
    """
    Загрузка новой модели по имени.
    Вызываем: POST /set?model=<имя>.
    """
    global model, current_model_name
    models_dir = "available_models"
    model_path = os.path.join(models_dir, model_param)

    if not os.path.isfile(model_path):
        raise HTTPException(status_code=404,
                            detail=f"Файл {model_param} не найден.")

    try:
        model = torch.hub.load(
            './yolov5',
            'custom',
            path=model_path,
            source='local'
        )
        current_model_name = model_param
        return {"status": f"Модель '{model_param}' успешно загружена!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/current")
def get_current():
    """Возвращает, какая модель сейчас загружена."""
    return {"current_model": current_model_name}


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    file_path = "temp.jpg"
    with open(file_path, "wb") as f:
        f.write(await image.read())

    results = model(file_path)

    boxes = []
    for x1, y1, x2, y2, conf, cls in results.xyxy[0].cpu().numpy():
        boxes.append({
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2),
            "confidence": float(conf),
            "class": int(cls),
            "label": model.names[int(cls)]
        })
    return {"boxes": boxes}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000)
