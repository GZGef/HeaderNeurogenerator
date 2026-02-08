"""
Модуль для загрузки и настройки модели.

Содержит классы для загрузки модели GPT и токенизатора,
подготовки модели к обучению и инференсу.
"""

import torch
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
)
import logging

from src.config import Config
from src.utils import setup_logging


class ModelLoader:
    """
    Класс для загрузки и настройки модели GPT.
    
    Загружает модель и токенизатор, готовит их к обучению
    и настраивает параметры.
    """
    
    def __init__(self, config: Config):
        """
        Инициализация загрузчика модели.
        
        Args:
            config: Конфигурация проекта
        """
        self.config = config
        self.logger = setup_logging(
            log_file=str(config.get_training_log_path()),
            level=config.logging.get("level", "INFO")
        )
        
        self.model_name = config.get_model_name()
        self.device = config.get_device()
        self.hf_token = config.get_hf_token()
        
        self.model = None
        self.tokenizer = None
        self.data_collator = None
        
        self.logger.info(f"Инициализация загрузчика модели")
        self.logger.info(f"Название модели: {self.model_name}")
        self.logger.info(f"Устройство: {self.device}")
    
    def load_model(self) -> AutoModelForCausalLM:
        """
        Загрузка модели GPT.
        
        Returns:
            Загруженная модель
        """
        self.logger.info(f"Загрузка модели {self.model_name}...")
        
        try:
            # Проверяем доступность CUDA
            import torch
            use_cuda = torch.cuda.is_available()
            
            if use_cuda:
                self.logger.info(f"CUDA доступно: {use_cuda}")
                self.logger.info(f"Название GPU: {torch.cuda.get_device_name(0)}")
                self.logger.info(f"Память GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            
            # Загрузка модели
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                dtype=torch.float16 if self.config.training.get("fp16", True) and use_cuda else torch.float32,
                device_map="auto" if use_cuda else None,
                token=self.hf_token,
                use_safetensors=True,  # Безопасная загрузка для предотвращения уязвимостей
            )
            
            # Перемещение на устройство
            if not use_cuda:
                model = model.to(self.device)
            
            self.logger.info(f"Модель загружена: {self.model_name}")
            self.logger.info(f"Количество параметров: {model.num_parameters():,}")
            
            return model
            
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке модели: {e}")
            raise
    
    def load_tokenizer(self) -> AutoTokenizer:
        """
        Загрузка токенизатора.
        
        Returns:
            Загруженный токенизатор
        """
        self.logger.info(f"Загрузка токенизатора {self.model_name}...")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=self.hf_token)
            
            # Установка специальных токенов
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            self.logger.info(f"Токенизатор загружен")
            self.logger.info(f"Размер словаря: {tokenizer.vocab_size}")
            self.logger.info(f"Токен начала последовательности: {tokenizer.bos_token}")
            self.logger.info(f"Токен конца последовательности: {tokenizer.eos_token}")
            self.logger.info(f"Токен паддинга: {tokenizer.pad_token}")
            
            return tokenizer
            
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке токенизатора: {e}")
            raise
    
    def prepare_model_for_training(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling]:
        """
        Подготовка модели к обучению.
        
        Returns:
            Кортеж (модель, токенизатор, коллатор данных)
        """
        self.logger.info("Подготовка модели к обучению...")
        
        # Загрузка модели
        self.model = self.load_model()
        
        # Загрузка токенизатора
        self.tokenizer = self.load_tokenizer()
        
        # Создание коллатора данных
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Для causal language modeling
            pad_to_multiple_of=8  # Для оптимизации на GPU
        )
        
        self.logger.info("Модель готова к обучению")
        
        return self.model, self.tokenizer, self.data_collator
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Получение информации о модели.
        
        Returns:
            Словарь с информацией о модели
        """
        if self.model is None:
            raise ValueError("Модель не загружена. Сначала вызовите prepare_model_for_training()")
        
        # Подсчет обучаемых параметров
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = self.model.num_parameters()
        
        info = {
            "name": self.model_name,
            "parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": self.device,
            "vocab_size": self.tokenizer.vocab_size if self.tokenizer else None,
            "max_length": self.config.get_max_length(),
            "max_new_tokens": self.config.get_max_new_tokens(),
        }
        
        return info
    
    def freeze_layers(self, num_layers_to_freeze: int = 0):
        """
        Заморозка слоев модели для fine-tuning.
        
        Args:
            num_layers_to_freeze: Количество слоев для заморозки (0 - не замораживать)
        """
        if self.model is None:
            raise ValueError("Модель не загружена")
        
        if num_layers_to_freeze > 0:
            self.logger.info(f"Заморозка {num_layers_to_freeze} слоев модели...")
            
            # Заморозка слоев
            for i, layer in enumerate(self.model.transformer.h):
                if i < num_layers_to_freeze:
                    for param in layer.parameters():
                        param.requires_grad = False
            
            self.logger.info(f"Заморожено {num_layers_to_freeze} слоев")
    
    def get_optimizer(self, lr: Optional[float] = None):
        """
        Создание оптимизатора для обучения.
        
        Args:
            lr: Скорость обучения (если None, берется из конфига)
        
        Returns:
            Оптимизатор
        """
        if self.model is None:
            raise ValueError("Модель не загружена")
        
        if lr is None:
            lr = self.config.training.get("learning_rate", 2e-5)
        
        # Фильтрация параметров для оптимизации
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=lr,
            weight_decay=self.config.training.get("weight_decay", 0.01)
        )
        
        self.logger.info(f"Оптимизатор создан: AdamW, lr={lr}")
        
        return optimizer
    
    def get_scheduler(self, optimizer, total_steps: int):
        """
        Создание планировщика обучения.
        
        Args:
            optimizer: Оптимизатор
            total_steps: Общее количество шагов обучения
        
        Returns:
            Планировщик
        """
        from transformers import get_linear_schedule_with_warmup
        
        warmup_steps = int(total_steps * self.config.training.get("warmup_ratio", 0.1))
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        self.logger.info(f"Планировщик создан: warmup={warmup_steps}, total={total_steps}")
        
        return scheduler
    
    def save_model(self, output_dir: Path):
        """
        Сохранение модели и токенизатора.
        
        Args:
            output_dir: Директория для сохранения
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Модель или токенизатор не загружены")
        
        self.logger.info(f"Сохранение модели в {output_dir}...")
        
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Сохранение модели
            self.model.save_pretrained(output_dir)
            
            # Сохранение токенизатора
            self.tokenizer.save_pretrained(output_dir)
            
            self.logger.info(f"Модель успешно сохранена в {output_dir}")
            
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении модели: {e}")
            raise
    
    def load_model_from_path(self, model_dir: Path) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Загрузка модели из директории.
        
        Args:
            model_dir: Директория с моделью
        
        Returns:
            Кортеж (модель, токенизатор)
        """
        self.logger.info(f"Загрузка модели из {model_dir}...")
        
        try:
            # Проверяем доступность CUDA
            import torch
            use_cuda = torch.cuda.is_available()
            
            if use_cuda:
                self.logger.info(f"CUDA доступно: {use_cuda}")
                self.logger.info(f"Название GPU: {torch.cuda.get_device_name(0)}")
            
            # Загрузка модели
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                dtype=torch.float16 if self.config.training.get("fp16", True) and use_cuda else torch.float32,
                device_map="auto" if use_cuda else None,
                token=self.hf_token,
                use_safetensors=True,  # Безопасная загрузка для предотвращения уязвимостей
            )
            
            # Перемещение на устройство
            if not use_cuda:
                model = model.to(self.device)
            
            # Загрузка токенизатора
            tokenizer = AutoTokenizer.from_pretrained(model_dir, token=self.hf_token)
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            self.logger.info(f"Модель загружена из {model_dir}")
            
            return model, tokenizer
            
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке модели из {model_dir}: {e}")
            raise
    
    def cleanup(self):
        """Очистка ресурсов."""
        self.logger.info("Очистка ресурсов загрузчика модели")
        
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        
        self.logger.info("Ресурсы очищены")