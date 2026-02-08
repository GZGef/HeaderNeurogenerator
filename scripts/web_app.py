#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Веб-приложение для генерации заголовков через API.

Запуск:
    python scripts/web_app.py
    или
    uvicorn scripts.web_app:app --host 0.0.0.0 --port 8000
"""

import sys
from pathlib import Path

# Добавляем корневую директорию в PYTHONPATH
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import logging

from src.config import Config
from src.model import ModelLoader
from src.inference import Inference
from src.utils import setup_logging


# Модели запроса и ответа
class NewsRequest(BaseModel):
    """Модель запроса на генерацию заголовков."""
    text: str
    num_titles: Optional[int] = 3


class NewsResponse(BaseModel):
    """Модель ответа с сгенерированными заголовками."""
    titles: List[str]
    model_name: str
    device: str


class HealthResponse(BaseModel):
    """Модель ответа для проверки здоровья."""
    status: str
    model_loaded: bool
    model_name: Optional[str]
    device: Optional[str]


# Глобальные переменные для хранения модели
app: FastAPI
inference: Optional[Inference] = None
config: Optional[Config] = None
logger: logging.Logger


def create_app():
    """
    Создание FastAPI приложения.
    
    Returns:
        FastAPI приложение
    """
    global app, inference, config, logger
    
    # Настройка логирования
    logger = setup_logging(
        log_file="logs/web_app.log",
        level="INFO"
    )
    
    logger.info("=" * 80)
    logger.info("ЗАПУСК ВЕБ-ПРИЛОЖЕНИЯ")
    logger.info("=" * 80)
    
    # Загрузка конфигурации
    logger.info("Загрузка конфигурации...")
    config = Config("config.yaml")
    logger.info(f"Конфигурация загружена: {config}")
    
    # Загрузка модели
    logger.info("Загрузка модели...")
    try:
        model_loader = ModelLoader(config)
        
        # Попытка загрузить обученную модель
        model_dir = config.get_final_model_path()
        if model_dir.exists():
            logger.info(f"Загрузка обученной модели из {model_dir}...")
            model, tokenizer = model_loader.load_model_from_path(model_dir)
        else:
            logger.warning("Обученная модель не найдена, загружается стандартная модель...")
            model, tokenizer, _ = model_loader.prepare_model_for_training()
        
        # Создание инференса
        inference = Inference(config, model, tokenizer)
        
        logger.info("Модель успешно загружена")
        
    except Exception as e:
        logger.error(f"Ошибка при загрузке модели: {e}", exc_info=True)
        raise
    
    # Создание FastAPI приложения
    app = FastAPI(
        title=config.get_web_title(),
        description=config.get_web_description(),
        version=config.get_web_version(),
    )
    
    # Добавление обработчиков
    add_handlers()
    
    return app


def add_handlers():
    """
    Добавление обработчиков маршрутов.
    """
    
    @app.get("/", response_model=HealthResponse)
    async def root():
        """Корневой маршрут."""
        return {
            "status": "running",
            "model_loaded": inference is not None,
            "model_name": inference.model.config._name_or_path if inference else None,
            "device": inference.device if inference else None,
        }
    
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Проверка здоровья сервиса."""
        return {
            "status": "healthy",
            "model_loaded": inference is not None,
            "model_name": inference.model.config._name_or_path if inference else None,
            "device": inference.device if inference else None,
        }
    
    @app.post("/generate", response_model=NewsResponse)
    async def generate_titles(request: NewsRequest):
        """
        Генерация заголовков к новости.
        
        Пример запроса:
        ```json
        {
            "text": "На Красной площади прошёл праздничный парад в честь Дня города...",
            "num_titles": 3
        }
        ```
        """
        try:
            if inference is None:
                raise HTTPException(status_code=500, detail="Модель не загружена")
            
            if not request.text.strip():
                raise HTTPException(status_code=400, detail="Текст не может быть пустым")
            
            logger.info(f"Запрос на генерацию заголовков (num_titles={request.num_titles})")
            
            # Генерация заголовков
            titles = inference.generate_titles(
                request.text,
                num_return_sequences=request.num_titles
            )
            
            logger.info(f"Сгенерировано {len(titles)} заголовков")
            
            return NewsResponse(
                titles=titles,
                model_name=inference.model.config._name_or_path,
                device=inference.device,
            )
            
        except Exception as e:
            logger.error(f"Ошибка при генерации заголовков: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/info")
    async def get_info():
        """Получение информации о модели."""
        if inference is None:
            raise HTTPException(status_code=500, detail="Модель не загружена")
        
        gen_info = inference.get_generation_info()
        
        return {
            "model_name": inference.model.config._name_or_path,
            "device": inference.device,
            "generation_params": gen_info,
            "vocab_size": inference.tokenizer.vocab_size,
            "max_length": config.get_max_length(),
            "max_new_tokens": config.get_max_new_tokens(),
        }
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Обработчик выключения приложения."""
        logger.info("Завершение работы веб-приложения...")
        
        if inference is not None:
            inference.cleanup()
        
        logger.info("Веб-приложение завершило работу")


# Создание приложения
app = create_app()


if __name__ == "__main__":
    # Запуск веб-сервера
    uvicorn.run(
        "scripts.web_app:app",
        host=config.get_web_host(),
        port=config.get_web_port(),
        reload=False,
        log_level="info",
    )