"""
Модуль для генерации заголовков.

Содержит класс для генерации заголовков к новостным текстам
с использованием обученной модели.
"""

import torch
from pathlib import Path
from typing import List, Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

from src.config import Config
from src.utils import setup_logging


class Inference:
    """
    Класс для генерации заголовков с использованием обученной модели.
    
    Предоставляет методы для генерации заголовков к новостным текстам.
    """
    
    def __init__(
        self,
        config: Config,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
    ):
        """
        Инициализация инференса.
        
        Args:
            config: Конфигурация проекта
            model: Модель для генерации
            tokenizer: Токенизатор
        """
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        
        self.logger = setup_logging(
            log_file=str(config.get_inference_log_path()),
            level=config.logging.get("level", "INFO")
        )
        
        self.device = config.get_device()
        
        self.logger.info("Инициализация инференса")
        self.logger.info(f"Устройство: {self.device}")
        self.logger.info(f"Модель: {model.config._name_or_path}")
    
    def generate_titles(self, text: str, num_return_sequences: Optional[int] = None) -> List[str]:
        """
        Генерация заголовков к тексту новости.
        
        Args:
            text: Текст новости
            num_return_sequences: Количество вариантов заголовков
        
        Returns:
            Список сгенерированных заголовков
        """
        self.logger.info("Начало генерации заголовков...")
        
        try:
            # Подготовка текста
            text_tokens = self.tokenizer.encode(text)[:300]
            truncated_text = self.tokenizer.decode(text_tokens)
            
            # Формирование входного текста
            input_text = f"{self.tokenizer.bos_token} {truncated_text} {self.tokenizer.eos_token}"
            
            # Токенизация
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.get_max_length(),
            )
            
            # Перемещение на устройство
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Параметры генерации
            gen_params = self.config.get_generation_params()
            
            if num_return_sequences is None:
                num_return_sequences = gen_params.get("num_return_sequences", 3)
            
            # Генерация
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=gen_params.get("max_new_tokens", 128),
                    num_beams=gen_params.get("num_beams", 1),
                    no_repeat_ngram_size=gen_params.get("no_repeat_ngram_size", 2),
                    do_sample=True,
                    top_k=gen_params.get("top_k", 20),
                    top_p=gen_params.get("top_p", 0.9),
                    temperature=gen_params.get("temperature", 0.7),
                    num_return_sequences=num_return_sequences,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            # Извлечение заголовков
            titles = []
            for i, output in enumerate(outputs):
                generated_tokens = output[inputs['input_ids'].shape[-1]:]
                generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=False)
                
                # Отделяем заголовок от текста
                parts = generated_text.split('<s>')
                title = parts[0].strip()
                
                # Очистка заголовка
                title = self._clean_title(title)
                
                if title:
                    titles.append(title)
                    self.logger.info(f"Вариант {i+1}: {title}")
            
            self.logger.info(f"Сгенерировано {len(titles)} заголовков")
            
            return titles
            
        except Exception as e:
            self.logger.error(f"Ошибка при генерации заголовков: {e}", exc_info=True)
            raise
    
    def _clean_title(self, title: str) -> str:
        """
        Очистка сгенерированного заголовка.
        
        Args:
            title: Сгенерированный заголовок
        
        Returns:
            Очищенный заголовок
        """
        # Удаление лишних пробелов
        title = ' '.join(title.split())
        
        # Удаление специальных символов в начале и конце
        title = title.strip('.,!?;:()[]{}')
        
        # Удаление пустых скобок
        title = title.replace('()', '').replace('[]', '').replace('{}', '')
        
        # Удаление лишних пробелов
        title = ' '.join(title.split())
        
        return title
    
    def generate_title(self, text: str) -> str:
        """
        Генерация одного заголовка к тексту новости.
        
        Args:
            text: Текст новости
        
        Returns:
            Сгенерированный заголовок
        """
        titles = self.generate_titles(text, num_return_sequences=1)
        return titles[0] if titles else ""
    
    def batch_generate_titles(self, texts: List[str]) -> List[List[str]]:
        """
        Пакетная генерация заголовков.
        
        Args:
            texts: Список текстов новостей
        
        Returns:
            Список списков сгенерированных заголовков
        """
        self.logger.info(f"Пакетная генерация для {len(texts)} текстов...")
        
        all_titles = []
        for i, text in enumerate(texts):
            self.logger.info(f"Обработка текста {i+1}/{len(texts)}")
            titles = self.generate_titles(text)
            all_titles.append(titles)
        
        self.logger.info("Пакетная генерация завершена")
        return all_titles
    
    def get_generation_info(self) -> Dict[str, Any]:
        """
        Получение информации о параметрах генерации.
        
        Returns:
            Словарь с параметрами генерации
        """
        gen_params = self.config.get_generation_params()
        
        info = {
            "max_new_tokens": gen_params.get("max_new_tokens", 128),
            "num_beams": gen_params.get("num_beams", 1),
            "no_repeat_ngram_size": gen_params.get("no_repeat_ngram_size", 2),
            "num_return_sequences": gen_params.get("num_return_sequences", 3),
            "temperature": gen_params.get("temperature", 0.7),
            "top_k": gen_params.get("top_k", 20),
            "top_p": gen_params.get("top_p", 0.9),
            "device": self.device,
        }
        
        return info
    
    def evaluate_titles(self, generated_titles: List[str], reference_titles: List[str]) -> Dict[str, float]:
        """
        Оценка качества сгенерированных заголовков.
        
        Args:
            generated_titles: Сгенерированные заголовки
            reference_titles: Референсные заголовки
        
        Returns:
            Словарь с метриками качества
        """
        try:
            from rouge_score import rouge_scorer
            
            scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'],
                use_stemmer=True
            )
            
            scores = []
            for gen, ref in zip(generated_titles, reference_titles):
                score = scorer.score(ref, gen)
                scores.append(score)
            
            # Средние значения
            avg_scores = {
                "rouge1": float(sum(s['rouge1'].fmeasure for s in scores) / len(scores)),
                "rouge2": float(sum(s['rouge2'].fmeasure for s in scores) / len(scores)),
                "rougeL": float(sum(s['rougeL'].fmeasure for s in scores) / len(scores)),
            }
            
            self.logger.info(f"Метрики качества: {avg_scores}")
            
            return avg_scores
            
        except ImportError:
            self.logger.warning("Библиотека rouge-score не установлена")
            return {}
        except Exception as e:
            self.logger.error(f"Ошибка при оценке качества: {e}")
            return {}
    
    def cleanup(self):
        """Очистка ресурсов."""
        self.logger.info("Очистка ресурсов инференса")
        
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