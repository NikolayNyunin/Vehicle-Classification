# Используем базовый образ Python
FROM python:3.10-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файл с зависимостями
COPY requirements.txt .

# Устанавливаем зависимости из requirements.txt
RUN python -m pip install --upgrade pip && python -m pip install --no-cache-dir -r requirements.txt

# Копируем весь проект в контейнер
COPY . .

# Открываем порт для Streamlit
EXPOSE 8501

# Запускаем приложение
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]