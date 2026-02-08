"""
Пакет для обучения и использования модели генерации заголовков.

Модули:
    config: Конфигурация проекта
    dataset: Загрузка и подготовка данных
    model: Загрузка и настройка модели
    trainer: Обучение модели
    inference: Генерация заголовков
    utils: Вспомогательные функции
"""

__version__ = "1.0.0"
__author__ = "AI Team"
__email__ = "team@example.com"

from src.config import Config
from src.dataset import DatasetLoader
from src.model import ModelLoader
from src.trainer import Trainer
from src.inference import Inference
from src.utils import setup_logging, save_json, load_json

__all__ = [
    "Config",
    "DatasetLoader",
    "ModelLoader",
    "Trainer",
    "Inference",
    "setup_logging",
    "save_json",
    "load_json",
]