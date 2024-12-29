from models.baseline.train import load_model, Config, CUDA
from models.baseline.inference import inference_one_file

from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.encoders import jsonable_encoder

from typing import List
from http import HTTPStatus
from contextlib import asynccontextmanager

# Сохранённые модели
models = {}

# Активная модель
active_model = {}


@asynccontextmanager
async def lifespan(router: APIRouter):
    """Функция жизненного цикла приложения."""

    # Загрузка бейзлайн модели
    name = 'CustomResNet18 (BASELINE)'
    description = '...'
    model = load_model('../data/best_checkpoint_val_p_0.8277_r_0.7873_f1_0.8043.pt')
    models[0] = {'name': name, 'description': description, 'model': model}

    yield

    # Удаление всех моделей
    models.clear()


router = APIRouter(lifespan=lifespan)


class FitRequest(BaseModel):
    """Запрос обучения модели."""

    name: str  # Название модели
    description: str  # Описание модели
    batch_size: int  # Гиперпараметр - размер батча
    pass


class FineTuneRequest(BaseModel):
    """Запрос дообучения модели."""

    id: int  # ID модели
    batch_size: int  # Гиперпараметр - размер батча
    pass


class SetRequest(BaseModel):
    """Запрос выбора активной модели."""

    id: int  # ID модели


class PredictResponse(BaseModel):
    """Ответ предсказания модели."""

    class_id: int  # ID предсказанного класса
    class_name: str  # Название предсказанного класса
    confidence: float  # Уверенность в предсказанном классе


class ModelInfo(BaseModel):
    """Информация о модели."""

    id: int  # ID модели
    name: str  # Название модели
    description: str  # Описание модели


class MessageResponse(BaseModel):
    """Стандартный ответ API."""

    message: str  # Сообщение


@router.post('/fit', response_model=MessageResponse, status_code=HTTPStatus.CREATED)
async def fit(request: FitRequest):
    """Обучение и сохранение модели."""

    json_request = jsonable_encoder(request)

    try:
        pass
        # TODO: implement

    except Exception as e:
        raise HTTPException(status_code=HTTPStatus.UNPROCESSABLE_ENTITY, detail=str(e).capitalize())

    return {'message': 'Model trained and saved'}


@router.post('/fine_tune', response_model=MessageResponse)
async def fine_tune(request: FineTuneRequest):
    """Дообучение модели."""

    json_request = jsonable_encoder(request)

    try:
        pass
        # TODO: implement

    except Exception as e:
        raise HTTPException(status_code=HTTPStatus.UNPROCESSABLE_ENTITY, detail=str(e).capitalize())

    return {'message': f'Model with id {request.id} fine-tuned'}


@router.post('/predict', response_model=PredictResponse)
async def predict(file: UploadFile):
    """Получение предсказаний при помощи модели."""

    if not file.filename.endswith('.jpg'):
        raise HTTPException(status_code=HTTPStatus.UNPROCESSABLE_ENTITY, detail='Only JPG images are supported')

    if not active_model:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail='No active model, call /set first')

    try:
        return inference_one_file(active_model['model_dict']['model'], file.file.read(), 'cuda' if CUDA else 'cpu')

    except Exception as e:
        raise HTTPException(status_code=HTTPStatus.UNPROCESSABLE_ENTITY, detail=str(e).capitalize())


@router.get('/models', response_model=List[ModelInfo])
async def list_models():
    """Получение списка моделей."""

    return [
        {
            'id': model_id,
            'name': model_dict['name'],
            'description': model_dict['description']
         }
        for model_id, model_dict in models.items()
    ]


@router.post('/set', response_model=MessageResponse)
async def set_model(request: SetRequest):
    """Выбор активной модели."""

    if request.id not in models:
        raise HTTPException(HTTPStatus.NOT_FOUND, f'Model with id {request.id} not found')

    active_model['id'] = request.id
    active_model['model_dict'] = models[request.id]

    return {'message': f'Model with id {request.id} successfully set active'}
