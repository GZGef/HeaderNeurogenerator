"""
Модуль конфигурации проекта.

Загружает и предоставляет доступ к параметрам конфигурации из YAML файла.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import os

# Загрузка переменных окружения из .env файла
load_dotenv()


class Config:
    """
    Класс для управления конфигурацией проекта.
    
    Загружает параметры из YAML файла и предоставляет доступ к ним через атрибуты.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Инициализация конфигурации.
        
        Args:
            config_path: Путь к YAML файлу с конфигурацией
        """
        self.config_path = Path(config_path)
        self._config = self._load_config()
        
        # Создание атрибутов для удобного доступа
        self.model = self._config.get("model", {})
        self.training = self._config.get("training", {})
        self.generation = self._config.get("generation", {})
        self.device = self._config.get("device", {})
        self.data = self._config.get("data", {})
        self.paths = self._config.get("paths", {})
        self.logging = self._config.get("logging", {})
        self.evaluation = self._config.get("evaluation", {})
        self.huggingface = self._config.get("huggingface", {})
        self.web = self._config.get("web", {})
        
        # Создание директорий
        self._create_directories()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Загрузка конфигурации из YAML файла.
        
        Returns:
            Словарь с параметрами конфигурации
        """
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Файл конфигурации не найден: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Ошибка парсинга YAML: {e}")
    
    def _create_directories(self):
        """Создание необходимых директорий."""
        directories = [
            self.paths.get("data_dir", "data"),
            self.paths.get("dataset_dir", "data/dataset"),
            self.paths.get("results_dir", "results"),
            self.paths.get("models_dir", "results/models"),
            self.paths.get("logs_dir", "logs"),
            self.paths.get("images_dir", "results/images"),
            self.paths.get("docs_dir", "docs"),
            self.paths.get("notebooks_dir", "notebooks"),
            self.paths.get("scripts_dir", "scripts"),
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def get_model_name(self) -> str:
        """Получение названия модели."""
        return self.model.get("name", "ai-forever/rugpt3small_based_on_gpt2")
    
    def get_max_length(self) -> int:
        """Получение максимальной длины входного текста."""
        return self.model.get("max_length", 320)
    
    def get_max_new_tokens(self) -> int:
        """Получение максимального количества новых токенов."""
        return self.model.get("max_new_tokens", 128)
    
    def get_training_params(self) -> Dict[str, Any]:
        """Получение параметров обучения."""
        return self.training
    
    def get_generation_params(self) -> Dict[str, Any]:
        """Получение параметров генерации."""
        return self.generation
    
    def get_device(self) -> str:
        """Получение устройства для вычислений."""
        import torch
        
        if self.device.get("auto_device", True):
            if torch.cuda.is_available():
                return f"cuda:{self.device.get('gpu_id', 0)}"
            else:
                return "cpu"
        
        if self.device.get("use_gpu", True) and torch.cuda.is_available():
            return f"cuda:{self.device.get('gpu_id', 0)}"
        
        return "cpu"
    
    def get_dataset_url(self) -> str:
        """Получение URL датасета."""
        return self.data.get("dataset_url", "")
    
    def get_dataset_path(self) -> str:
        """Получение пути к датасету."""
        return self.data.get("dataset_path", "data/dataset/news.csv")
    
    def get_num_examples(self) -> int:
        """Получение количества примеров для использования."""
        return self.data.get("num_examples", 10000)
    
    def get_results_dir(self) -> Path:
        """Получение пути к директории с результатами."""
        return Path(self.paths.get("results_dir", "results"))
    
    def get_logs_dir(self) -> Path:
        """Получение пути к директории с логами."""
        return Path(self.paths.get("logs_dir", "logs"))
    
    def get_models_dir(self) -> Path:
        """Получение пути к директории с моделями."""
        return Path(self.paths.get("models_dir", "results/models"))
    
    def get_images_dir(self) -> Path:
        """Получение пути к директории с графиками."""
        return Path(self.paths.get("images_dir", "results/images"))
    
    def get_dataset_stats_path(self) -> Path:
        """Получение пути к файлу со статистикой датасета."""
        return self.get_results_dir() / "dataset_stats.json"
    
    def get_test_results_path(self) -> Path:
        """Получение пути к файлу с результатами тестирования."""
        return self.get_results_dir() / "test_results.json"
    
    def get_test_report_path(self) -> Path:
        """Получение пути к файлу с отчетом тестирования."""
        return self.get_results_dir() / "test_report.md"
    
    def get_final_model_path(self) -> Path:
        """Получение пути к финальной модели."""
        return self.get_results_dir() / "final_model"
    
    def get_training_log_path(self) -> Path:
        """Получение пути к файлу лога обучения."""
        return self.get_logs_dir() / "training.log"
    
    def get_inference_log_path(self) -> Path:
        """Получение пути к файлу лога инференса."""
        return self.get_logs_dir() / "inference.log"
    
    def get_loss_plot_path(self) -> Path:
        """Получение пути к файлу графика потерь."""
        return self.get_images_dir() / "loss_plot.png"
    
    def get_comparison_plot_path(self) -> Path:
        """Получение пути к файлу графика сравнения."""
        return self.get_images_dir() / "comparison_plot.png"
    
    def get_dataset_stats_plot_path(self) -> Path:
        """Получение пути к файлу графика статистики датасета."""
        return self.get_images_dir() / "dataset_stats.png"
    
    def get_generation_stats_plot_path(self) -> Path:
        """Получение пути к файлу графика статистики генерации."""
        return self.get_images_dir() / "generation_stats.png"
    
    def get_hf_token(self) -> Optional[str]:
        """Получение токена HuggingFace."""
        import os
        # Сначала проверяем переменную окружения HF_TOKEN
        token = os.environ.get("HF_TOKEN")
        if token:
            return token
        
        # Если переменной окружения нет, пытаемся получить из конфига
        token = self.huggingface.get("token", "")
        if not token:
            return None
        return token
    
    def get_web_host(self) -> str:
        """Получение хоста для веб-интерфейса."""
        return self.web.get("host", "0.0.0.0")
    
    def get_web_port(self) -> int:
        """Получение порта для веб-интерфейса."""
        return self.web.get("port", 8000)
    
    def get_web_title(self) -> str:
        """Получение названия веб-приложения."""
        return self.web.get("title", "Нейрогенератор заголовков")
    
    def get_web_description(self) -> str:
        """Получение описания веб-приложения."""
        return self.web.get("description", "Сервис для генерации заголовков к новостным текстам")
    
    def get_web_version(self) -> str:
        """Получение версии веб-приложения."""
        return self.web.get("version", "1.0.0")
    
    def __repr__(self) -> str:
        """Строковое представление конфигурации."""
        return f"Config(path={self.config_path})"
    
    def __str__(self) -> str:
        """Строковое представление конфигурации."""
        return f"Конфигурация из {self.config_path}"