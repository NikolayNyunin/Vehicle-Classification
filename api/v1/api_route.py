from models.baseline.train import load_model, CUDA
from models.baseline.inference import inference_one_file

from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.encoders import jsonable_encoder
from loguru import logger

from typing import List
from http import HTTPStatus
from contextlib import asynccontextmanager

# Добавление файла логов с ротацией
logger.add('logs/backend.log', rotation='100 MB')

# Сохранённые модели
saved_models = {}

# Активная модель
active_model = {}


def load_baseline_model() -> dict:
    """Загрузка бейзлайн-модели."""

    name = 'CustomResNet18 (BASELINE)'
    description = '...'
    model = load_model('models/baseline/best_checkpoint.pt')
    return {'name': name, 'description': description, 'model': model}


@asynccontextmanager
async def lifespan(router: APIRouter):
    """Функция жизненного цикла приложения."""

    # Загрузка бейзлайн-модели
    saved_models[0] = load_baseline_model()

    logger.info('Baseline model loaded')

    yield

    # Удаление всех моделей
    saved_models.clear()

    logger.info('Saved models cleared')


router = APIRouter(lifespan=lifespan)


class Hyperparameters(BaseModel):
    """Гиперпараметры для обучения/дообучения."""

    batch_size: int  # Размер батча
    n_epochs: int  # Число эпох обучения
    eval_every: int  # Частота оценки


class FitRequest(BaseModel):
    """Запрос обучения новой модели."""

    name: str  # Название новой модели
    description: str  # Описание новой модели
    hyperparameters: Hyperparameters  # Гиперпараметры для обучения


class FineTuneRequest(BaseModel):
    """Запрос дообучения существующей модели."""

    id: int  # ID модели для дообучения
    name: str  # Название новой модели
    description: str  # Описание новой модели
    hyperparameters: Hyperparameters  # Гиперпараметры для дообучения


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
    """Обучение и сохранение новой модели."""

    logger.info('"/fit" requested')

    json_request = jsonable_encoder(request)
    name = json_request['name']
    description = json_request['description']
    hyperparameters = json_request['hyperparameters']

    try:
        # TODO: add model training
        # new_model = train_model(hyperparameters)
        new_model = load_baseline_model()['model']

    except Exception as e:
        logger.error(f'"/fit" failed: {str(e).capitalize()}')

        raise HTTPException(status_code=HTTPStatus.UNPROCESSABLE_ENTITY, detail=str(e).capitalize())

    else:
        new_id = max(saved_models.keys()) + 1
        saved_models[new_id] = {'name': name, 'description': description, 'model': new_model}

        logger.success('"/fit" responded successfully')

        return {'message': f'New model trained and saved with id {new_id}'}


@router.post('/fine_tune', response_model=MessageResponse, status_code=HTTPStatus.CREATED)
async def fine_tune(request: FineTuneRequest):
    """Дообучение существующей модели."""

    logger.info('"/fine_tune" requested')

    json_request = jsonable_encoder(request)
    model_id = json_request['id']
    name = json_request['name']
    description = json_request['description']
    hyperparameters = json_request['hyperparameters']

    if model_id not in saved_models:
        logger.error(f'"/fine_tune" failed: Model with id {model_id} not found')

        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail=f'Model with id {model_id} not found')

    try:
        # TODO: add model fine tuning
        # new_model = fine_tune_model(saved_models[model_id], hyperparameters)
        new_model = load_baseline_model()['model']

    except Exception as e:
        logger.error(f'"/fine_tune" failed: {str(e).capitalize()}')

        raise HTTPException(status_code=HTTPStatus.UNPROCESSABLE_ENTITY, detail=str(e).capitalize())

    else:
        new_id = max(saved_models.keys()) + 1
        saved_models[new_id] = {'name': name, 'description': description, 'model': new_model}

        logger.success('"/fine_tune" responded successfully')

        return {'message': f'Model with id {model_id} fine-tuned and saved with id {new_id}'}


@router.post('/predict', response_model=List[PredictResponse])
async def predict(files: List[UploadFile]):
    """Получение предсказаний при помощи выбранной модели."""

    logger.info('"/predict" requested')

    if not active_model:
        logger.error('"/predict" failed: No active model')

        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail='No active model, call /set first')

    predictions = []

    for file in files:
        if not file.filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            logger.error(f'"/predict" failed: File type not supported')

            raise HTTPException(status_code=HTTPStatus.UNPROCESSABLE_ENTITY, detail='File type not supported')

        try:
            prediction = inference_one_file(active_model['model_dict']['model'],
                                            file.file.read(), 'cuda' if CUDA else 'cpu')
            predictions.append(prediction)

        except Exception as e:
            logger.error(f'"/predict" failed: {str(e).capitalize()}')

            raise HTTPException(status_code=HTTPStatus.UNPROCESSABLE_ENTITY, detail=str(e).capitalize())

    logger.success('"/predict" responded successfully')

    return predictions


@router.get('/models', response_model=List[ModelInfo])
async def list_models():
    """Получение списка сохранённых моделей."""

    logger.info('"/models" requested')
    logger.success('"/models" responded successfully')

    return [
        {
            'id': model_id,
            'name': model_dict['name'],
            'description': model_dict['description']
         }
        for model_id, model_dict in saved_models.items()
    ]


@router.post('/set', response_model=MessageResponse)
async def set_model(request: SetRequest):
    """Выбор активной модели."""

    logger.info('"/set" requested')

    if request.id not in saved_models:
        logger.error(f'"/set" failed: Model with id {request.id} not found')

        raise HTTPException(HTTPStatus.NOT_FOUND, f'Model with id {request.id} not found')

    active_model['id'] = request.id
    active_model['model_dict'] = saved_models[request.id]

    logger.success('"/set" responded successfully')

    return {'message': f'Model with id {request.id} successfully set active'}
