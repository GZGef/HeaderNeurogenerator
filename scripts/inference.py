#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Скрипт для генерации заголовков с использованием обученной модели.

Запуск:
    python scripts/inference.py --interactive
    python scripts/inference.py --input scripts/example_data.json --output results/generated_titles.json
"""

import sys
from pathlib import Path

# Добавляем корневую директорию в PYTHONPATH
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

import argparse
import json
import logging
from typing import List, Dict, Any
from src.config import Config
from src.model import ModelLoader
from src.inference import Inference
from src.utils import setup_logging, load_json, save_json


def load_model_for_inference(config: Config):
    """
    Загрузка модели для инференса.
    
    Args:
        config: Конфигурация проекта
    
    Returns:
        Кортеж (model, tokenizer, inference)
    """
    logger = logging.getLogger(__name__)
    
    # Попытка загрузить обученную модель
    model_dir = config.get_final_model_path()
    
    if model_dir.exists():
        logger.info(f"Загрузка обученной модели из {model_dir}...")
        model_loader = ModelLoader(config)
        model, tokenizer = model_loader.load_model_from_path(model_dir)
    else:
        logger.warning("Обученная модель не найдена, загружается стандартная модель...")
        logger.warning("Результаты могут быть неоптимальными")
        model_loader = ModelLoader(config)
        model, tokenizer, _ = model_loader.prepare_model_for_training()
    
    # Создание объекта инференса
    inference = Inference(config, model, tokenizer)
    
    return model, tokenizer, inference


def interactive_mode(config: Config):
    """
    Интерактивный режим генерации заголовков.
    
    Args:
        config: Конфигурация проекта
    """
    logger = logging.getLogger(__name__)
    
    print("=" * 80)
    print("ИНТЕРАКТИВНЫЙ РЕЖИМ ГЕНЕРАЦИИ ЗАГОЛОВКОВ")
    print("=" * 80)
    
    # Загрузка модели
    model, tokenizer, inference = load_model_for_inference(config)
    
    print("\nВведите текст новости (или 'exit' для выхода):")
    
    while True:
        try:
            text = input("\nТекст новости: ").strip()
            
            if text.lower() in ['exit', 'выход', 'quit']:
                print("\nВыход из интерактивного режима")
                break
            
            if not text:
                print("⚠️  Текст не может быть пустым")
                continue
            
            # Генерация заголовков
            print("\n⏳ Генерация заголовков...")
            titles = inference.generate_titles(text)
            
            # Вывод результатов
            print("\n" + "=" * 80)
            print("СГЕНЕРИРОВАННЫЕ ЗАГОЛОВКИ:")
            print("=" * 80)
            for i, title in enumerate(titles, 1):
                print(f"{i}. {title}")
            print("=" * 80)
            
        except KeyboardInterrupt:
            logger.info("\nПрервано пользователем")
            break
        except Exception as e:
            logger.error(f"Ошибка: {e}")
            continue
    
    # Очистка ресурсов
    inference.cleanup()
    model = None
    tokenizer = None


def generate_from_file(config: Config, input_path: str, output_path: str):
    """
    Генерация заголовков из файла.
    
    Args:
        config: Конфигурация проекта
        input_path: Путь к входному файлу
        output_path: Путь к выходному файлу
    """
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("ГЕНЕРАЦИЯ ЗАГОЛОВКОВ ИЗ ФАЙЛА")
    logger.info("=" * 80)
    
    # Загрузка данных
    logger.info(f"Загрузка данных из {input_path}...")
    data = load_json(Path(input_path))
    
    if not isinstance(data, list):
        logger.error("Данные должны быть списком объектов")
        return
    
    logger.info(f"Загружено {len(data)} примеров")
    
    # Загрузка модели
    model, tokenizer, inference = load_model_for_inference(config)
    
    # Генерация заголовков
    results = []
    for i, item in enumerate(data, 1):
        try:
            text = item.get("text", "")
            reference_title = item.get("title", "")
            
            if not text:
                logger.warning(f"Пример {i}: пустой текст, пропускаем")
                continue
            
            logger.info(f"Обработка примера {i}/{len(data)}")
            
            # Генерация заголовков
            titles = inference.generate_titles(text)
            
            # Сохранение результата
            result = {
                "text": text,
                "generated_titles": titles,
                "reference_title": reference_title,
            }
            
            results.append(result)
            
            # Вывод прогресса
            if i % 5 == 0:
                logger.info(f"Обработано {i}/{len(data)} примеров")
                
        except Exception as e:
            logger.error(f"Ошибка при обработке примера {i}: {e}")
            continue
    
    # Сохранение результатов
    logger.info(f"Сохранение результатов в {output_path}...")
    save_json(results, Path(output_path))
    
    logger.info(f"Результаты сохранены: {len(results)} примеров")
    
    # Очистка ресурсов
    inference.cleanup()
    model = None
    tokenizer = None


def main():
    """
    Основная функция скрипта.
    """
    # Парсинг аргументов
    parser = argparse.ArgumentParser(
        description="Генерация заголовков к новостным текстам",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  Интерактивный режим:
    python scripts/inference.py --interactive
  
  Генерация из файла:
    python scripts/inference.py --input scripts/example_data.json --output results/generated_titles.json
        """
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Запустить интерактивный режим"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        help="Путь к входному JSON файлу с текстами новостей"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="results/generated_titles.json",
        help="Путь к выходному JSON файлу (по умолчанию: results/generated_titles.json)"
    )
    
    args = parser.parse_args()
    
    # Настройка логирования
    logger = setup_logging(
        log_file="logs/inference.log",
        level="INFO"
    )
    
    logger.info("=" * 80)
    logger.info("ЗАПУСК СКРИПТА ГЕНЕРАЦИИ ЗАГОЛОВКОВ")
    logger.info("=" * 80)
    
    try:
        # Загрузка конфигурации
        logger.info("Загрузка конфигурации...")
        config = Config("config.yaml")
        logger.info(f"Конфигурация загружена: {config}")
        
        # Выбор режима работы
        if args.interactive:
            interactive_mode(config)
        elif args.input:
            generate_from_file(config, args.input, args.output)
        else:
            logger.error("Не указан режим работы. Используйте --interactive или --input")
            parser.print_help()
            return 1
        
        logger.info("=" * 80)
        logger.info("РАБОТА СКРИПТА ЗАВЕРШЕНА")
        logger.info("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)