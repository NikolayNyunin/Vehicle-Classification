from typing import List
from http import HTTPStatus

from pydantic import BaseModel
from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder

# Сохранённые модели
models = {}

# Активная модель
active_model = {}

router = APIRouter()


class FitRequest(BaseModel):
    pass


class FineTuneRequest(BaseModel):
    id: int
    pass


class PredictRequest(BaseModel):
    pass


class SetRequest(BaseModel):
    id: int


class PredictResponse(BaseModel):
    pass


class ModelInfo(BaseModel):
    id: int
    name: str
    description: str


class MessageResponse(BaseModel):
    message: str



@router.post('/fit', response_model=MessageResponse, status_code=HTTPStatus.CREATED)
async def fit(request: FitRequest):
    """Обучение и сохранение модели."""

    # TODO: implement

    return {'message': 'Model trained and saved'}


@router.post('/fine_tune', response_model=MessageResponse)
async def fine_tune(request: FineTuneRequest):
    """Дообучение модели."""

    # TODO: implement

    return {'message': f'Model with id {request.id} fine-tuned'}


@router.post('/predict', response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Получение предсказаний при помощи модели."""

    pass


@router.get('/models', response_model=List[ModelInfo])
async def list_models():
    """Получение списка моделей."""

    pass


@router.post('/set', response_model=MessageResponse)
async def set_model(request: SetRequest):
    """Выбор активной модели."""

    if request.id not in models:
        raise HTTPException(HTTPStatus.NOT_FOUND, f'Model with id {request.id} not found')

    active_model['id'] = request.id
    active_model['model'] = models[request.id]

    return {'message': f'Model with id {request.id} successfully set active'}
