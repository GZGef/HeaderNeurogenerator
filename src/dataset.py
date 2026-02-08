"""
Модуль для загрузки и подготовки данных.

Содержит классы для загрузки датасета новостей, предобработки данных
и подготовки их для обучения модели.
"""

import pandas as pd
import numpy as np
import re
import zipfile
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datasets import Dataset
import logging

from src.config import Config
from src.utils import setup_logging


class DatasetLoader:
    """
    Класс для загрузки и подготовки датасета новостей.
    
    Загружает датасет с удалённого источника, выполняет предобработку
    и готовит данные для обучения модели.
    """
    
    def __init__(self, config: Config):
        """
        Инициализация загрузчика датасета.
        
        Args:
            config: Конфигурация проекта
        """
        self.config = config
        self.logger = setup_logging(
            log_file=str(config.get_training_log_path()),
            level=config.logging.get("level", "INFO")
        )
        
        self.dataset_url = config.get_dataset_url()
        self.dataset_path = Path(config.get_dataset_path())
        self.num_examples = config.get_num_examples()
        
        self.logger.info(f"Инициализация загрузчика датасета")
        self.logger.info(f"URL датасета: {self.dataset_url}")
        self.logger.info(f"Путь к датасету: {self.dataset_path}")
        self.logger.info(f"Количество примеров: {self.num_examples}")
    
    def download_dataset(self) -> bool:
        """
        Загрузка датасета с удалённого источника.
        
        Returns:
            True если загрузка успешна, иначе False
        """
        self.logger.info("Начало загрузки датасета...")
        
        try:
            # Создание директории для датасета
            self.dataset_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Скачивание архива
            zip_path = self.dataset_path.parent / "russiannews_dataset.zip"
            
            self.logger.info(f"Скачивание архива в {zip_path}")
            response = requests.get(self.dataset_url, stream=True)
            response.raise_for_status()
            
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            self.logger.info(f"Архив скачан, размер: {zip_path.stat().st_size / 1e6:.2f} MB")
            
            # Распаковка архива
            self.logger.info(f"Распаковка архива в {self.dataset_path.parent}")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.dataset_path.parent)
            
            self.logger.info("Датасет успешно распакован")
            
            # Удаление архива
            zip_path.unlink()
            self.logger.info("Архив удалён")
            
            return True
            
        except requests.RequestException as e:
            self.logger.error(f"Ошибка при скачивании датасета: {e}")
            return False
        except zipfile.BadZipFile as e:
            self.logger.error(f"Ошибка при распаковке архива: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Неизвестная ошибка при загрузке датасета: {e}")
            return False
    
    def load_dataframe(self) -> pd.DataFrame:
        """
        Загрузка датасета в DataFrame.
        
        Returns:
            DataFrame с данными
        """
        self.logger.info("Загрузка данных в DataFrame...")
        
        # Проверка существования файла
        if not self.dataset_path.exists():
            self.logger.warning(f"Файл датасета не найден: {self.dataset_path}")
            self.logger.info("Попытка загрузить датасет...")
            if not self.download_dataset():
                raise FileNotFoundError(f"Не удалось загрузить датасет: {self.dataset_path}")
        
        try:
            # Загрузка CSV
            df = pd.read_csv(
                self.dataset_path,
                sep=self.config.data.get("separator", ","),
                encoding=self.config.data.get("encoding", "utf-8")
            )
            
            self.logger.info(f"Датасет загружен: {len(df)} строк, {len(df.columns)} колонок")
            self.logger.info(f"Колонки: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке датасета: {e}")
            raise
    
    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Предобработка данных в DataFrame.
        
        Args:
            df: Исходный DataFrame
        
        Returns:
            Предобработанный DataFrame
        """
        self.logger.info("Начало предобработки данных...")
        
        # Сохраняем исходный размер
        original_size = len(df)
        
        # Выбор колонок
        columns = self.config.data.get("columns", ["title", "text"])
        self.logger.info(f"Выбор колонок: {columns}")
        
        if not all(col in df.columns for col in columns):
            available = list(df.columns)
            raise ValueError(f"Указанные колонки {columns} не найдены. Доступные: {available}")
        
        df = df[columns].copy()
        
        # Удаление пустых значений
        self.logger.info("Удаление пустых значений...")
        df = df.dropna()
        self.logger.info(f"Удалено пустых значений: {original_size - len(df)}")
        
        # Удаление дубликатов
        original_size = len(df)
        self.logger.info("Удаление дубликатов...")
        df = df.drop_duplicates()
        self.logger.info(f"Удалено дубликатов: {original_size - len(df)}")
        
        # Очистка текста
        self.logger.info("Очистка текста...")
        df['text'] = df['text'].apply(self._clean_text)
        df['title'] = df['title'].apply(self._clean_text)
        
        # Удаление пустых строк после очистки
        df = df.dropna()
        
        # Ограничение размера
        if self.num_examples > 0 and len(df) > self.num_examples:
            self.logger.info(f"Ограничение размера до {self.num_examples} примеров")
            df = df.iloc[:self.num_examples]
        
        self.logger.info(f"Предобработка завершена: {len(df)} примеров")
        
        return df
    
    def _clean_text(self, text: str) -> Optional[str]:
        """
        Очистка текста от лишних символов.
        
        Args:
            text: Исходный текст
        
        Returns:
            Очищенный текст
        """
        if pd.isna(text):
            return None
        
        text = str(text)
        
        # Удаление лишних пробелов
        text = re.sub(r'\s+', ' ', text)
        
        # Удаление специальных символов (оставляем кириллицу, латиницу, цифры, знаки препинания)
        text = re.sub(r'[^\w\s.,!?;:()\-–—]', '', text)
        
        # Удаление лишних пробелов в начале и конце
        text = text.strip()
        
        # Проверка, что текст не пустой
        if len(text) < 5:
            return None
        
        return text
    
    def prepare_dataset(self, tokenizer=None) -> Dataset:
        """
        Подготовка датасета для обучения.
        
        Args:
            tokenizer: Токенизатор для токенизации текста (опционально)
        
        Returns:
            Dataset из библиотеки datasets
        """
        self.logger.info("Подготовка датасета для обучения...")
        
        # Загрузка и предобработка
        df = self.load_dataframe()
        df = self.preprocess_dataframe(df)
        
        # Конвертация в Dataset
        dataset = Dataset.from_pandas(df)
        
        # Подготовка данных для обучения (форматирование для GPT)
        def format_for_training(example):
            """Форматирование примера для обучения GPT."""
            text = example['text']
            title = example['title']
            
            # Формат: <text> -> <title>
            formatted_text = f"Текст: {text}\nЗаголовок: {title}"
            
            # Если токенизатор предоставлен, токенизируем текст
            if tokenizer is not None:
                # Токенизация текста с паддингом до max_length
                tokenized = tokenizer(
                    formatted_text,
                    truncation=True,
                    max_length=self.config.get_max_length(),
                    padding='max_length',  # Паддинг до max_length для всех примеров
                    return_tensors=None
                )
                
                # Для causal language modeling, input_ids и labels совпадают
                # tokenized["input_ids"] это список токенов
                input_ids = tokenized["input_ids"]
                
                return {
                    "input_ids": input_ids,
                    "labels": input_ids
                }
            else:
                # Если токенизатор не предоставлен, возвращаем текст
                return {"text": formatted_text, "labels": formatted_text}
        
        # Применяем форматирование
        dataset = dataset.map(format_for_training, remove_columns=['title', 'text'])
        
        # Фильтрация пустых примеров (только если токенизатор был предоставлен)
        if tokenizer is not None:
            # Добавим отладочную информацию
            self.logger.info("Проверка структуры данных после токенизации...")
            if len(dataset) > 0:
                first_example = dataset[0]
                self.logger.info(f"Первый пример: {first_example}")
                self.logger.info(f"Тип input_ids: {type(first_example.get('input_ids'))}")
                self.logger.info(f"Значение input_ids: {first_example.get('input_ids')}")
            
            dataset = dataset.filter(lambda x: isinstance(x["input_ids"], list) and len(x["input_ids"]) > 0)
        
        self.logger.info(f"Датасет подготовлен: {len(dataset)} примеров")
        
        # Проверка, что датасет не пустой
        if len(dataset) == 0:
            self.logger.error("Датасет пустой после подготовки!")
            raise ValueError("Датасет пустой. Проверьте данные и токенизацию.")
        
        return dataset
    
    def get_dataset_info(self, dataset: Dataset) -> Dict[str, Any]:
        """
        Получение информации о датасете.
        
        Args:
            dataset: Датасет
        
        Returns:
            Словарь с информацией
        """
        info = {
            "length": len(dataset),
            "columns": dataset.column_names,
            "features": {col: str(dataset.features[col]) for col in dataset.column_names},
        }
        
        return info
    
    def split_dataset(self, dataset: Dataset, test_size: float = 0.2) -> Tuple[Dataset, Dataset]:
        """
        Разделение датасета на обучающую и тестовую выборки.
        
        Args:
            dataset: Исходный датасет
            test_size: Доля тестовой выборки
        
        Returns:
            Кортеж (train_dataset, test_dataset)
        """
        self.logger.info(f"Разделение датасета на train/test (test_size={test_size})")
        
        # Разделение
        split = dataset.train_test_split(test_size=test_size, seed=self.config.training.get("seed", 42))
        
        train_dataset = split["train"]
        test_dataset = split["test"]
        
        self.logger.info(f"Train: {len(train_dataset)} примеров")
        self.logger.info(f"Test: {len(test_dataset)} примеров")
        
        return train_dataset, test_dataset
    
    def get_statistical_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Получение статистической информации о датасете.
        
        Args:
            df: DataFrame с данными
        
        Returns:
            Словарь со статистикой
        """
        stats = {
            "total_examples": len(df),
            "mean_text_length": float(df['text'].str.len().mean()),
            "mean_title_length": float(df['title'].str.len().mean()),
            "mean_text_words": float(df['text'].str.split().str.len().mean()),
            "mean_title_words": float(df['title'].str.split().str.len().mean()),
            "std_text_length": float(df['text'].str.len().std()),
            "std_title_length": float(df['title'].str.len().std()),
            "min_text_length": int(df['text'].str.len().min()),
            "max_text_length": int(df['text'].str.len().max()),
            "min_title_length": int(df['title'].str.len().min()),
            "max_title_length": int(df['title'].str.len().max()),
        }
        
        return stats
    
    def cleanup(self):
        """Очистка ресурсов."""
        self.logger.info("Очистка ресурсов загрузчика датасета")