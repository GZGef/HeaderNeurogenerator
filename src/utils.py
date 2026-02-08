"""
Вспомогательные утилиты для проекта.

Модуль содержит функции для логирования, сохранения/загрузки данных,
и других вспомогательных операций.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import sys


def setup_logging(
    log_file: Optional[str] = None,
    level: str = "INFO",
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
) -> logging.Logger:
    """
    Настройка логирования для проекта.
    
    Args:
        log_file: Путь к файлу лога (если None, логи только в консоль)
        level: Уровень логирования (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format: Формат сообщений лога
    
    Returns:
        Настроенный логгер
    """
    # Создание логгера
    logger = logging.getLogger("news_title_generator")
    
    # Установка уровня логирования
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    logger.setLevel(level_map.get(level.upper(), logging.INFO))
    
    # Удаление существующих обработчиков
    logger.handlers.clear()
    
    # Форматтер
    formatter = logging.Formatter(format)
    
    # Обработчик для консоли
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Обработчик для файла (если указан)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def save_json(data: Any, filepath: Path, ensure_ascii: bool = False, indent: int = 2) -> None:
    """
    Сохранение данных в JSON файл.
    
    Args:
        data: Данные для сохранения
        filepath: Путь к файлу
        ensure_ascii: Использовать ли только ASCII символы
        indent: Отступ для форматирования
    """
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=ensure_ascii, indent=indent)
    except Exception as e:
        raise IOError(f"Ошибка сохранения JSON файла {filepath}: {e}")


def load_json(filepath: Path, encoding: str = "utf-8") -> Any:
    """
    Загрузка данных из JSON файла.
    
    Args:
        filepath: Путь к файлу
        encoding: Кодировка файла
    
    Returns:
        Загруженные данные
    """
    try:
        with open(filepath, 'r', encoding=encoding) as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Файл не найден: {filepath}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Ошибка парсинга JSON файла {filepath}: {e}")


def save_text(text: str, filepath: Path, encoding: str = "utf-8") -> None:
    """
    Сохранение текста в файл.
    
    Args:
        text: Текст для сохранения
        filepath: Путь к файлу
        encoding: Кодировка файла
    """
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding=encoding) as f:
            f.write(text)
    except Exception as e:
        raise IOError(f"Ошибка сохранения текстового файла {filepath}: {e}")


def load_text(filepath: Path, encoding: str = "utf-8") -> str:
    """
    Загрузка текста из файла.
    
    Args:
        filepath: Путь к файлу
        encoding: Кодировка файла
    
    Returns:
        Загруженный текст
    """
    try:
        with open(filepath, 'r', encoding=encoding) as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Файл не найден: {filepath}")


def format_number(num: float, decimals: int = 2) -> str:
    """
    Форматирование числа с разделителями.
    
    Args:
        num: Число для форматирования
        decimals: Количество знаков после запятой
    
    Returns:
        Отформатированная строка
    """
    return f"{num:,.{decimals}f}"


def format_percentage(num: float, decimals: int = 1) -> str:
    """
    Форматирование числа как процентов.
    
    Args:
        num: Число для форматирования (0-100)
        decimals: Количество знаков после запятой
    
    Returns:
        Отформатированная строка с процентом
    """
    return f"{num:.{decimals}f}%"


def create_markdown_table(headers: List[str], rows: List[List[Any]]) -> str:
    """
    Создание markdown таблицы.
    
    Args:
        headers: Заголовки столбцов
        rows: Строки таблицы
    
    Returns:
        Строка с markdown таблицей
    """
    if not rows:
        return ""
    
    # Вычисляем ширину каждого столбца
    col_widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))
    
    # Создаем разделитель
    separator = "| " + " | ".join(["-" * w for w in col_widths]) + " |"
    
    # Создаем заголовок
    header_row = "| " + " | ".join([str(h).ljust(w) for h, w in zip(headers, col_widths)]) + " |"
    
    # Создаем строки
    data_rows = []
    for row in rows:
        data_row = "| " + " | ".join([str(c).ljust(w) for c, w in zip(row, col_widths)]) + " |"
        data_rows.append(data_row)
    
    # Собираем таблицу
    table = "\n".join([header_row, separator] + data_rows)
    return table


def check_gpu_availability() -> Dict[str, Any]:
    """
    Проверка доступности GPU.
    
    Returns:
        Словарь с информацией о GPU
    """
    try:
        import torch
        
        info = {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "current_device": None,
            "device_name": None,
            "total_memory_gb": None,
        }
        
        if torch.cuda.is_available():
            device_id = torch.cuda.current_device()
            info["current_device"] = device_id
            info["device_name"] = torch.cuda.get_device_name(device_id)
            info["total_memory_gb"] = torch.cuda.get_device_properties(device_id).total_memory / 1e9
        
        return info
    except ImportError:
        return {"available": False, "error": "PyTorch not installed"}


def validate_config(config: Dict[str, Any]) -> List[str]:
    """
    Валидация конфигурации.
    
    Args:
        config: Словарь с конфигурацией
    
    Returns:
        Список ошибок валидации (пустой, если все ок)
    """
    errors = []
    
    # Проверка обязательных секций
    required_sections = ["model", "training", "generation", "data", "paths"]
    for section in required_sections:
        if section not in config:
            errors.append(f"Отсутствует обязательная секция: {section}")
    
    # Проверка параметров модели
    if "model" in config:
        model = config["model"]
        if "name" not in model:
            errors.append("В секции model отсутствует параметр 'name'")
        if "max_length" not in model:
            errors.append("В секции model отсутствует параметр 'max_length'")
    
    # Проверка параметров обучения
    if "training" in config:
        training = config["training"]
        if "epochs" not in training:
            errors.append("В секции training отсутствует параметр 'epochs'")
        if "batch_size" not in training:
            errors.append("В секции training отсутствует параметр 'batch_size'")
    
    # Проверка параметров данных
    if "data" in config:
        data = config["data"]
        if "dataset_url" not in data:
            errors.append("В секции data отсутствует параметр 'dataset_url'")
    
    return errors


def get_project_info() -> Dict[str, str]:
    """
    Получение информации о проекте.
    
    Returns:
        Словарь с информацией о проекте
    """
    return {
        "name": "Нейрогенератор заголовков",
        "version": "1.0.0",
        "description": "Система для обучения GPT модели на русскоязычных новостях и генерации заголовков",
        "author": "AI Team",
        "email": "team@example.com",
    }