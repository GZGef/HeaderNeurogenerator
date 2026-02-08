#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Скрипт для обучения модели генерации заголовков.

Запуск:
    python scripts/train.py
"""

import sys
from pathlib import Path

# Добавляем корневую директорию в PYTHONPATH
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

import logging
from src.config import Config
from src.dataset import DatasetLoader
from src.model import ModelLoader
from src.trainer import Trainer
from src.utils import setup_logging, save_json


def main():
    """
    Основная функция обучения модели.
    """
    # Настройка логирования
    logger = setup_logging(
        log_file="logs/training.log",
        level="INFO"
    )
    
    logger.info("=" * 80)
    logger.info("ЗАПУСК ОБУЧЕНИЯ МОДЕЛИ ГЕНЕРАЦИИ ЗАГОЛОВКОВ")
    logger.info("=" * 80)
    
    try:
        # 1. Загрузка конфигурации
        logger.info("1. Загрузка конфигурации...")
        config = Config("config.yaml")
        logger.info(f"Конфигурация загружена: {config}")
        
        # 2. Загрузка и подготовка данных
        logger.info("2. Загрузка и подготовка данных...")
        dataset_loader = DatasetLoader(config)
        
        # Сначала загрузим модель и токенизатор для подготовки датасета
        logger.info("3. Загрузка модели...")
        model_loader = ModelLoader(config)
        model, tokenizer, data_collator = model_loader.prepare_model_for_training()
        
        # Получение информации о модели
        model_info = model_loader.get_model_info()
        logger.info(f"Модель загружена: {model_info['name']}")
        logger.info(f"Количество параметров: {model_info['parameters']:,}")
        
        # Подготовка датасета с токенизатором
        train_dataset = dataset_loader.prepare_dataset(tokenizer=tokenizer)
        
        # Получение информации о датасете
        dataset_info = dataset_loader.get_dataset_info(train_dataset)
        logger.info(f"Датасет загружен: {dataset_info['length']} примеров")
        
        # Сохранение информации о датасете
        save_json(dataset_info, config.get_dataset_stats_path())
        logger.info(f"Информация о датасете сохранена в {config.get_dataset_stats_path()}")
        
        # 4. Создание тренера
        logger.info("4. Создание тренера...")
        trainer = Trainer(config, model, tokenizer, data_collator, train_dataset)
        
        # Настройка аргументов обучения
        trainer.setup_training_args()
        
        # Настройка оптимизатора
        trainer.setup_optimizer()
        
        # 5. Обучение модели
        logger.info("5. Обучение модели...")
        trainer.train()
        
        # 6. Сохранение модели
        logger.info("6. Сохранение модели...")
        trainer.save_model()
        
        # 7. Очистка ресурсов
        logger.info("7. Очистка ресурсов...")
        trainer.cleanup()
        model_loader.cleanup()
        dataset_loader.cleanup()
        
        logger.info("=" * 80)
        logger.info("ОБУЧЕНИЕ УСПЕШНО ЗАВЕРШЕНО")
        logger.info("=" * 80)
        logger.info(f"Модель сохранена в: {config.get_final_model_path()}")
        logger.info(f"Информация о датасете: {config.get_dataset_stats_path()}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Критическая ошибка при обучении: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)