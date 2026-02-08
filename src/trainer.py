"""
Модуль для обучения модели.

Содержит класс для обучения модели GPT на датасете новостей.
"""

import torch
from pathlib import Path
from typing import Dict, Any, Optional
from transformers import (
    TrainingArguments,
    Trainer as TransformersTrainer,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset
import logging

from src.config import Config
from src.utils import setup_logging, save_json


class Trainer:
    """
    Класс для обучения модели GPT.
    
    Управляет процессом обучения, логированием и сохранением модели.
    """
    
    def __init__(
        self,
        config: Config,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        data_collator: DataCollatorForLanguageModeling,
        train_dataset: Dataset,
    ):
        """
        Инициализация тренера.
        
        Args:
            config: Конфигурация проекта
            model: Модель для обучения
            tokenizer: Токенизатор
            data_collator: Коллатор данных
            train_dataset: Датасет для обучения
        """
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.data_collator = data_collator
        self.train_dataset = train_dataset
        
        self.logger = setup_logging(
            log_file=str(config.get_training_log_path()),
            level=config.logging.get("level", "INFO")
        )
        
        self.training_args = None
        self.trainer = None
        self.optimizer = None
        self.scheduler = None
        
        self.logger.info("Инициализация тренера")
        self.logger.info(f"Размер датасета: {len(train_dataset)} примеров")
    
    def setup_training_args(self):
        """
        Настройка аргументов обучения.
        """
        self.logger.info("Настройка аргументов обучения...")
        
        training_params = self.config.get_training_params()
        
        # Определяем устройство
        device = self.config.get_device()
        self.logger.info(f"Используемое устройство: {device}")
        
        # Проверяем доступность CUDA
        import torch
        use_cuda = torch.cuda.is_available()
        self.logger.info(f"CUDA доступно: {use_cuda}")
        
        if use_cuda:
            self.logger.info(f"Название GPU: {torch.cuda.get_device_name(0)}")
            self.logger.info(f"Память GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        self.training_args = TrainingArguments(
            output_dir=str(self.config.get_results_dir()),
            num_train_epochs=training_params.get("epochs", 4),
            per_device_train_batch_size=training_params.get("batch_size", 16),
            learning_rate=training_params.get("learning_rate", 2e-5),  # Добавлен learning_rate
            warmup_ratio=training_params.get("warmup_ratio", 0.1),
            gradient_accumulation_steps=training_params.get("gradient_accumulation_steps", 2),
            push_to_hub=False,
            seed=training_params.get("seed", 42),
            fp16=training_params.get("fp16", True) and use_cuda,  # FP16 только если CUDA доступно
            weight_decay=training_params.get("weight_decay", 0.01),
            logging_strategy="steps",
            logging_steps=training_params.get("logging_steps", 500),  # Уменьшена частота логирования
            logging_dir=str(self.config.get_logs_dir()),
            report_to="none",
            save_strategy="no",  # Отключаем автосохранение во время обучения
            dataloader_num_workers=0,
        )
        
        self.logger.info("Аргументы обучения настроены")
        self.logger.info(f"  Эпохи: {self.training_args.num_train_epochs}")
        self.logger.info(f"  Batch size: {self.training_args.per_device_train_batch_size}")
        self.logger.info(f"  Learning rate: {training_params.get('learning_rate', 2e-5)}")
        self.logger.info(f"  Warmup ratio: {self.training_args.warmup_ratio}")
        self.logger.info(f"  Gradient accumulation: {self.training_args.gradient_accumulation_steps}")
        self.logger.info(f"  FP16: {self.training_args.fp16}")
    
    def setup_optimizer(self):
        """
        Настройка оптимизатора (Trainer создаст его автоматически для совместимости с FP16).
        """
        self.logger.info("Оптимизатор будет создан автоматически Trainer'ом")
        self.logger.info(f"  Тип: AdamW (по умолчанию)")
        self.logger.info(f"  Learning rate: {self.config.training.get('learning_rate', 2e-5)}")
        self.logger.info(f"  Weight decay: {self.config.training.get('weight_decay', 0.01)}")
        
        # Не создаем собственный оптимизатор - пусть Trainer создаст его
        # для правильной работы с FP16
        self.optimizer = None
        self.scheduler = None
    
    def train(self):
        """
        Запуск обучения модели.
        """
        self.logger.info("=" * 80)
        self.logger.info("НАЧАЛО ОБУЧЕНИЯ")
        self.logger.info("=" * 80)
        
        try:
            # Создание тренера (без custom optimizer для совместимости с FP16)
            self.trainer = TransformersTrainer(
                model=self.model,
                args=self.training_args,
                data_collator=self.data_collator,
                train_dataset=self.train_dataset,
            )
            
            # Запуск обучения
            self.trainer.train()
            
            self.logger.info("=" * 80)
            self.logger.info("ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО")
            self.logger.info("=" * 80)
            
        except Exception as e:
            self.logger.error(f"Ошибка при обучении: {e}", exc_info=True)
            raise
    
    def save_model(self):
        """
        Сохранение обученной модели.
        """
        self.logger.info("Сохранение модели...")
        
        try:
            # Сохранение модели
            output_dir = self.config.get_final_model_path()
            self.trainer.save_model(str(output_dir))
            
            # Сохранение токенизатора
            self.tokenizer.save_pretrained(output_dir)
            
            self.logger.info(f"Модель сохранена в {output_dir}")
            
            # Сохранение информации о тренировке
            training_info = self.get_training_info()
            save_json(training_info, output_dir / "training_info.json")
            
            self.logger.info("Информация о тренировке сохранена")
            
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении модели: {e}")
            raise
    
    def get_training_info(self) -> Dict[str, Any]:
        """
        Получение информации о тренировке.
        
        Returns:
            Словарь с информацией
        """
        if self.trainer is None:
            return {}
        
        # Получение логов тренировки
        logs = self.trainer.state.log_history
        
        # Извлечение информации
        training_info = {
            "total_steps": self.trainer.state.global_step,
            "total_epochs": self.training_args.num_train_epochs,
            "train_batch_size": self.training_args.per_device_train_batch_size,
            "learning_rate": self.config.training.get("learning_rate", 2e-5),
            "gradient_accumulation_steps": self.training_args.gradient_accumulation_steps,
            "warmup_ratio": self.training_args.warmup_ratio,
            "weight_decay": self.training_args.weight_decay,
            "fp16": self.training_args.fp16,
            "seed": self.training_args.seed,
            "dataset_size": len(self.train_dataset),
            "final_loss": logs[-1].get("loss", None) if logs else None,
        }
        
        return training_info
    
    def cleanup(self):
        """Очистка ресурсов."""
        self.logger.info("Очистка ресурсов тренера")
        
        if self.trainer is not None:
            self.trainer = None
        
        if self.optimizer is not None:
            del self.optimizer
            self.optimizer = None
        
        if self.scheduler is not None:
            del self.scheduler
            self.scheduler = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        
        self.logger.info("Ресурсы очищены")