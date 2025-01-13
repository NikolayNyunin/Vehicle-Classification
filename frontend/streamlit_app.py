import streamlit as st
import requests
import pandas as pd
from loguru import logger

import os

# -----------------------------------------------------------------------------
# НАСТРОЙКА ЛОГИРОВАНИЯ
# -----------------------------------------------------------------------------
logger.add("logs/frontend.log", rotation="100 MB", retention=5, encoding="utf-8", level="INFO",
           format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")

# -----------------------------------------------------------------------------
# КОНФИГ БЭКЕНДА
# -----------------------------------------------------------------------------
# Адрес, на котором крутится ваш FastAPI. Если вы запускаете локально:
API_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000") + "/api/v1"

# -----------------------------------------------------------------------------
# ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ ПРИЛОЖЕНИЯ
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Vehicle Classification", page_icon="🚗", layout="wide")
if "dataset" not in st.session_state:
    st.session_state["dataset"] = None  # Чтобы помнить о загруженном датасете

# -----------------------------------------------------------------------------
# ФУНКЦИИ ВЗАИМОДЕЙСТВИЯ С API
# -----------------------------------------------------------------------------
def list_models():
    """
    Получаем список сохранённых моделей с бэкенда.
    """
    resp = requests.get(f"{API_URL}/models")
    if resp.status_code == 200:
        return resp.json()
    else:
        st.error(f"Ошибка при получении списка моделей: {resp.text}")
        logger.error(f"Ошибка при /models: {resp.text}")
        return []

def set_active_model(model_id: int):
    """
    Устанавливаем активную модель по ID.
    """
    payload = {"id": model_id}
    resp = requests.post(f"{API_URL}/set", json=payload)
    if resp.status_code == 200:
        return resp.json()
    else:
        st.error(f"Ошибка при выборе модели: {resp.text}")
        logger.error(f"Ошибка при /set: {resp.text}")
        return None

def train_new_model(name: str, description: str, batch_size: int, n_epochs: int, eval_every: int):
    """
    Создаём новую модель (обучение с нуля) через эндпоинт /fit.
    """
    payload = {
        "name": name,
        "description": description,
        "hyperparameters": {
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "eval_every": eval_every
        }
    }
    resp = requests.post(f"{API_URL}/fit", json=payload)
    if resp.status_code == 201:
        return resp.json()
    else:
        st.error(f"Ошибка при обучении модели: {resp.text}")
        logger.error(f"Ошибка при /fit: {resp.text}")
        return None

def fine_tune_model(model_id: int, name: str, description: str, batch_size: int, n_epochs: int, eval_every: int):
    """
    Дообучаем (fine-tune) существующую модель через эндпоинт /fine_tune.
    """
    payload = {
        "id": model_id,
        "name": name,
        "description": description,
        "hyperparameters": {
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "eval_every": eval_every
        }
    }
    resp = requests.post(f"{API_URL}/fine_tune", json=payload)
    if resp.status_code == 201:
        return resp.json()
    else:
        st.error(f"Ошибка при дообучении модели: {resp.text}")
        logger.error(f"Ошибка при /fine_tune: {resp.text}")
        return None

def predict(files):
    """
    Запрос инференса у модели. Принимает список файлов (из ст.file_uploader).
    """
    multiple_files = []
    for f in files:
        multiple_files.append(("files", (f.name, f, f.type)))

    resp = requests.post(f"{API_URL}/predict", files=multiple_files)
    if resp.status_code == 200:
        return resp.json()
    else:
        st.error(f"Ошибка при инференсе: {resp.text}")
        logger.error(f"Ошибка при /predict: {resp.text}")
        return None

# -----------------------------------------------------------------------------
# СТРАНИЦА "ЗАГРУЗКА ДАТАСЕТА" И EDA
# -----------------------------------------------------------------------------
def page_upload_and_eda():
    st.header("Загрузка датасета и базовый EDA")
    uploaded_file = st.file_uploader("Загрузите CSV-файл с данными", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state["dataset"] = df  # Сохраняем в session_state
            st.success("Датасет успешно загружен!")
            logger.info("Пользователь загрузил датасет.")
        except Exception as e:
            st.error(f"Не удалось прочитать CSV: {e}")
            logger.exception("Ошибка чтения CSV")

    if st.session_state["dataset"] is not None:
        st.subheader("Просмотр первых строк датасета:")
        st.dataframe(st.session_state["dataset"].head(10))

        st.subheader("Основные статистики:")
        st.write(st.session_state["dataset"].describe(include='all'))

# -----------------------------------------------------------------------------
# СТРАНИЦА "ОБУЧЕНИЕ / ДООБУЧЕНИЕ"
# -----------------------------------------------------------------------------
def page_training():
    st.header("Создание и дообучение модели")

    st.subheader("Создать новую модель")
    with st.form("new_model_form"):
        new_model_name = st.text_input("Имя новой модели", value="MyNewModel")
        new_model_desc = st.text_area("Описание новой модели", value="...")
        batch_size = st.number_input("Batch size", value=16, min_value=1)
        n_epochs = st.number_input("Epochs", value=2, min_value=1)
        eval_every = st.number_input("Eval every (кол-во итераций)", value=100, min_value=1)

        submitted_new = st.form_submit_button("Начать обучение")
        if submitted_new:
            result = train_new_model(
                name=new_model_name,
                description=new_model_desc,
                batch_size=batch_size,
                n_epochs=n_epochs,
                eval_every=eval_every
            )
            if result:
                st.success(result.get("message", "Обучение завершено"))
                logger.info(f"Новая модель создана: {result}")

    st.markdown("---")
    st.subheader("Дообучить существующую модель")

    models_list = list_models()
    model_options = {f"{m['name']} (ID={m['id']})": m['id'] for m in models_list}
    if model_options:
        chosen_model = st.selectbox("Выберите модель для дообучения", list(model_options.keys()))
    else:
        st.warning("Моделей пока нет")
        chosen_model = None

    with st.form("fine_tune_form"):
        new_ft_name = st.text_input("Имя (новая модель после дообучения)", value="MyFineTunedModel")
        new_ft_desc = st.text_area("Описание", value="...")
        ft_batch_size = st.number_input("Batch size", value=16, min_value=1, key="ft_bs")
        ft_n_epochs = st.number_input("Epochs", value=2, min_value=1, key="ft_ep")
        ft_eval_every = st.number_input("Eval every (кол-во итераций)", value=100, min_value=1, key="ft_ev")

        submitted_ft = st.form_submit_button("Дообучить модель")
        if submitted_ft and chosen_model is not None:
            ft_result = fine_tune_model(
                model_options[chosen_model],
                name=new_ft_name,
                description=new_ft_desc,
                batch_size=ft_batch_size,
                n_epochs=ft_n_epochs,
                eval_every=ft_eval_every
            )
            if ft_result:
                st.success(ft_result.get("message", "Дообучение завершено"))
                logger.info(f"Дообучение модели: {ft_result}")

# -----------------------------------------------------------------------------
# СТРАНИЦА "МОДЕЛИ" — СПИСОК И УСТАНОВКА АКТИВНОЙ МОДЕЛИ
# -----------------------------------------------------------------------------
def page_models():
    st.header("Список моделей и выбор активной")

    models_list = list_models()
    if not models_list:
        st.warning("Нет ни одной сохранённой модели.")
        return

    df_models = pd.DataFrame(models_list)
    st.dataframe(df_models)

    st.subheader("Установить активную модель")
    model_options = {f"{m['name']} (ID={m['id']})": m['id'] for m in models_list}
    chosen_model = st.selectbox("Выберите модель", list(model_options.keys()))
    if st.button("Сделать выбранную модель активной"):
        resp = set_active_model(model_options[chosen_model])
        if resp:
            st.success(resp.get("message", "Модель установлена"))
            logger.info(f"Активная модель выбрана: {resp}")

# -----------------------------------------------------------------------------
# СТРАНИЦА "ИНФЕРЕНС"
# -----------------------------------------------------------------------------
def page_inference():
    st.header("Инференс (предсказание класса)")

    st.write("Выберите одну или несколько картинок для предсказания")
    uploaded_images = st.file_uploader("Загрузите изображения", type=["png", "jpg", "jpeg", "bmp", "tiff"],
                                       accept_multiple_files=True)

    if st.button("Получить предсказания"):
        if uploaded_images:
            predictions = predict(uploaded_images)
            if predictions:
                for i, p in enumerate(predictions):
                    st.write(f"**Файл**: {uploaded_images[i].name}")
                    st.write(f"**Предсказанный класс**: {p['class_name']}")
                    st.write(f"**Уверенность**: {p['confidence']:.4f}")
                    st.write("---")
        else:
            st.warning("Сначала загрузите хотя бы одно изображение!")

# -----------------------------------------------------------------------------
# ОСНОВНАЯ СТРАНИЦА (НАВИГАЦИЯ)
# -----------------------------------------------------------------------------
def main():
    st.title("Vehicle Classification (Group 47)")

    menu = ["Загрузка датасета и EDA", "Обучение/Дообучение", "Модели", "Инференс"]
    choice = st.sidebar.selectbox("Навигация", menu)

    if choice == "Загрузка датасета и EDA":
        page_upload_and_eda()
    elif choice == "Обучение/Дообучение":
        page_training()
    elif choice == "Модели":
        page_models()
    elif choice == "Инференс":
        page_inference()

if __name__ == "__main__":
    main()
