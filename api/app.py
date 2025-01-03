from api.v1.api_route import router

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from loguru import logger

app = FastAPI(title='Vehicle Classification API')


class StatusResponse(BaseModel):
    status: str


@app.get('/', response_model=StatusResponse)
async def root():
    """Получение информации о статусе сервиса."""

    logger.info('"/" requested')
    logger.success('"/" responded successfully')

    return {'status': 'OK'}


@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    """Получение иконки вкладки."""

    logger.info('"/favicon.ico" requested')
    logger.success('"/favicon.ico" responded successfully')

    return FileResponse('api/img/favicon.png')


app.include_router(router, prefix='/api/v1')

if __name__ == '__main__':
    uvicorn.run('app:app', host='127.0.0.1', port=8000, reload=True)

    logger.info('Uvicorn server started')
