#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Главная точка входа в проект "Нейрогенератор заголовков".

Этот скрипт предоставляет удобный интерфейс для работы с системой:
- Обучение модели
- Генерация заголовков
- Запуск веб-API
- Визуализация результатов

Запуск:
    python main.py --help
    python main.py train
    python main.py inference --interactive
    python main.py web
    python main.py visualize
"""

import sys
from pathlib import Path

# Добавляем корневую директорию в PYTHONPATH
root_dir = Path(__file__).parent
sys.path.insert(0, str(root_dir))

import argparse
import logging
from src.utils import setup_logging


def train_command(args):
    """Обучение модели."""
    from scripts.train import main as train_main
    
    logger = logging.getLogger(__name__)
    logger.info("Запуск обучения модели...")
    
    exit_code = train_main()
    
    if exit_code == 0:
        logger.info("✅ Обучение завершено успешно!")
    else:
        logger.error("❌ Ошибка при обучении")
    
    return exit_code


def inference_command(args):
    """Генерация заголовков."""
    from scripts.inference import main as inference_main
    
    logger = logging.getLogger(__name__)
    logger.info("Запуск генерации заголовков...")
    
    # Передаем аргументы
    sys.argv = ['scripts/inference.py']
    if args.interactive:
        sys.argv.append('--interactive')
    if args.input:
        sys.argv.extend(['--input', args.input])
    if args.output:
        sys.argv.extend(['--output', args.output])
    
    exit_code = inference_main()
    
    if exit_code == 0:
        logger.info("✅ Генерация завершена успешно!")
    else:
        logger.error("❌ Ошибка при генерации")
    
    return exit_code


def web_command(args):
    """Запуск веб-API."""
    from scripts.web_app import main as web_main
    
    logger = logging.getLogger(__name__)
    logger.info("Запуск веб-API...")
    
    exit_code = web_main()
    
    if exit_code == 0:
        logger.info("✅ Веб-API запущен!")
    else:
        logger.error("❌ Ошибка при запуске веб-API")
    
    return exit_code


def visualize_command(args):
    """Визуализация результатов."""
    from scripts.visualize_results import main as visualize_main
    
    logger = logging.getLogger(__name__)
    logger.info("Запуск визуализации результатов...")
    
    exit_code = visualize_main()
    
    if exit_code == 0:
        logger.info("✅ Визуализация завершена!")
    else:
        logger.error("❌ Ошибка при визуализации")
    
    return exit_code


def info_command(args):
    """Вывод информации о проекте."""
    from src.utils import get_project_info, check_gpu_availability
    
    logger = logging.getLogger(__name__)
    
    # Информация о проекте
    project_info = get_project_info()
    
    print("\n" + "=" * 80)
    print("ИНФОРМАЦИЯ О ПРОЕКТЕ")
    print("=" * 80)
    print(f"Название: {project_info['name']}")
    print(f"Версия: {project_info['version']}")
    print(f"Описание: {project_info['description']}")
    print(f"Автор: {project_info['author']}")
    print(f"Email: {project_info['email']}")
    print()
    
    # Информация о GPU
    gpu_info = check_gpu_availability()
    
    print("ИНФОРМАЦИЯ О СИСТЕМЕ")
    print("=" * 80)
    print(f"GPU доступен: {'✅ Да' if gpu_info['available'] else '❌ Нет'}")
    
    if gpu_info['available']:
        print(f"Количество GPU: {gpu_info['device_count']}")
        print(f"Название GPU: {gpu_info['device_name']}")
        print(f"Память GPU: {gpu_info['total_memory_gb']:.2f} GB")
    else:
        print(f"Ошибка: {gpu_info.get('error', 'Неизвестно')}")
    
    print()
    
    # Информация о конфигурации
    try:
        from src.config import Config
        config = Config("config.yaml")
        
        print("КОНФИГУРАЦИЯ")
        print("=" * 80)
        print(f"Модель: {config.get_model_name()}")
        print(f"Устройство: {config.get_device()}")
        print(f"Количество примеров: {config.get_num_examples()}")
        print(f"Эпохи обучения: {config.training.get('epochs', 4)}")
        print(f"Размер батча: {config.training.get('batch_size', 16)}")
        print()
        
    except Exception as e:
        logger.warning(f"Не удалось загрузить конфигурацию: {e}")
    
    print("=" * 80)
    
    return 0


def main():
    """
    Основная функция парсинга аргументов и запуска команд.
    """
    parser = argparse.ArgumentParser(
        description="Нейрогенератор заголовков - система для обучения GPT модели на русскоязычных новостях",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  Обучение модели:
    python main.py train
  
  Генерация заголовков (интерактивный режим):
    python main.py inference --interactive
  
  Генерация из файла:
    python main.py inference --input scripts/example_data.json --output results/output.json
  
  Запуск веб-API:
    python main.py web
  
  Визуализация результатов:
    python main.py visualize
  
  Информация о системе:
    python main.py info
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Доступные команды')
    
    # Команда обучения
    parser_train = subparsers.add_parser('train', help='Обучение модели')
    
    # Команда инференса
    parser_inference = subparsers.add_parser('inference', help='Генерация заголовков')
    parser_inference.add_argument('--interactive', action='store_true', help='Интерактивный режим')
    parser_inference.add_argument('--input', type=str, help='Путь к входному JSON файлу')
    parser_inference.add_argument('--output', type=str, default='results/generated_titles.json', help='Путь к выходному файлу')
    
    # Команда веб-API
    parser_web = subparsers.add_parser('web', help='Запуск веб-API')
    
    # Команда визуализации
    parser_visualize = subparsers.add_parser('visualize', help='Визуализация результатов')
    
    # Команда информации
    parser_info = subparsers.add_parser('info', help='Информация о системе')
    
    args = parser.parse_args()
    
    # Настройка логирования
    logger = setup_logging(
        log_file="logs/main.log",
        level="INFO"
    )
    
    logger.info("=" * 80)
    logger.info("ЗАПУСК НЕЙРОГЕНЕРАТОРА ЗАГОЛОВКОВ")
    logger.info("=" * 80)
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Выполнение команды
    try:
        if args.command == 'train':
            return train_command(args)
        elif args.command == 'inference':
            return inference_command(args)
        elif args.command == 'web':
            return web_command(args)
        elif args.command == 'visualize':
            return visualize_command(args)
        elif args.command == 'info':
            return info_command(args)
        else:
            parser.print_help()
            return 1
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)