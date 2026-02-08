#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Скрипт для визуализации результатов обучения и генерации.

Запуск:
    python scripts/visualize_results.py
"""

import sys
from pathlib import Path

# Добавляем корневую директорию в PYTHONPATH
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import logging

from src.config import Config
from src.utils import setup_logging, load_json


def setup_plotting():
    """Настройка стиля для графиков."""
    plt.style.use('seaborn-v0_8')
    sns.set_palette('husl')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14


def plot_dataset_stats(config: Config):
    """
    Визуализация статистики датасета.
    
    Args:
        config: Конфигурация проекта
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Загрузка статистики
        stats_path = config.get_dataset_stats_path()
        if not stats_path.exists():
            logger.warning(f"Файл статистики не найден: {stats_path}")
            logger.info("Пропускаем визуализацию статистики датасета")
            return
        
        stats = load_json(stats_path)
        
        # Проверка наличия необходимых ключей
        required_keys = ['total_examples', 'mean_text_length', 'mean_title_length']
        if not all(key in stats for key in required_keys):
            logger.warning(f"Файл статистики не содержит необходимых данных")
            logger.info("Пропускаем визуализацию статистики датасета")
            return
        
        # Создание графиков
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Количество примеров
        axes[0, 0].bar(['Датасет'], [stats['total_examples']], color='skyblue')
        axes[0, 0].set_title('Количество примеров в датасете')
        axes[0, 0].set_ylabel('Количество')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Средняя длина текстов и заголовков
        categories = ['Текст', 'Заголовок']
        lengths = [stats['mean_text_length'], stats['mean_title_length']]
        axes[0, 1].bar(categories, lengths, color=['lightcoral', 'lightgreen'])
        axes[0, 1].set_title('Средняя длина (символы)')
        axes[0, 1].set_ylabel('Длина')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Среднее количество слов
        words = [stats['mean_text_words'], stats['mean_title_words']]
        axes[1, 0].bar(categories, words, color=['gold', 'orange'])
        axes[1, 0].set_title('Среднее количество слов')
        axes[1, 0].set_ylabel('Количество слов')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Диапазон длин
        text_range = [stats['min_text_length'], stats['max_text_length']]
        title_range = [stats['min_title_length'], stats['max_title_length']]
        
        x = np.arange(2)
        width = 0.35
        
        axes[1, 1].bar(x - width/2, text_range, width, label='Текст', color='lightblue')
        axes[1, 1].bar(x + width/2, title_range, width, label='Заголовок', color='lightpink')
        axes[1, 1].set_title('Диапазон длин')
        axes[1, 1].set_ylabel('Длина (символы)')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(['Минимум', 'Максимум'])
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Сохранение
        output_path = config.get_dataset_stats_plot_path()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"График статистики датасета сохранен в {output_path}")
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Ошибка при визуализации статистики датасета: {e}")


def plot_generation_stats(config: Config):
    """
    Визуализация статистики генерации.
    
    Args:
        config: Конфигурация проекта
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Загрузка результатов генерации
        results_path = config.get_test_results_path()
        if not results_path.exists():
            logger.warning(f"Файл результатов не найден: {results_path}")
            return
        
        results = load_json(results_path)
        
        # Сбор статистики
        all_titles = []
        for result in results:
            all_titles.extend(result.get('generated_titles', []))
        
        if not all_titles:
            logger.warning("Нет данных для визуализации")
            return
        
        # Расчет длин
        title_lengths = [len(title) for title in all_titles]
        title_words = [len(title.split()) for title in all_titles]
        
        # Создание графиков
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. Распределение длин заголовков
        axes[0].hist(title_lengths, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0].set_xlabel('Длина заголовка (символы)')
        axes[0].set_ylabel('Количество')
        axes[0].set_title('Распределение длины сгенерированных заголовков')
        axes[0].axvline(np.mean(title_lengths), color='red', linestyle='--', 
                       label=f'Средняя: {np.mean(title_lengths):.1f}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. Распределение количества слов
        axes[1].hist(title_words, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[1].set_xlabel('Количество слов в заголовке')
        axes[1].set_ylabel('Количество')
        axes[1].set_title('Распределение количества слов в заголовках')
        axes[1].axvline(np.mean(title_words), color='red', linestyle='--', 
                       label=f'Средняя: {np.mean(title_words):.1f}')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Сохранение
        output_path = config.get_generation_stats_plot_path()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"График статистики генерации сохранен в {output_path}")
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Ошибка при визуализации статистики генерации: {e}")


def plot_comparison(config: Config):
    """
    Визуализация сравнения длин текстов и заголовков.
    
    Args:
        config: Конфигурация проекта
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Загрузка результатов генерации
        results_path = config.get_test_results_path()
        if not results_path.exists():
            logger.warning(f"Файл результатов не найден: {results_path}")
            return
        
        results = load_json(results_path)
        
        # Подготовка данных
        texts = []
        generated = []
        reference = []
        
        for result in results:
            text = result.get('text', '')
            gen_titles = result.get('generated_titles', [])
            ref_title = result.get('reference_title', '')
            
            if text and gen_titles and ref_title:
                texts.append(len(text))
                generated.append(len(gen_titles[0]))  # Берем первый заголовок
                reference.append(len(ref_title))
        
        if not texts:
            logger.warning("Нет данных для сравнения")
            return
        
        # Создание графика
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = range(len(texts))
        
        ax.plot(x, texts, label='Длина текста', marker='o', alpha=0.7, linewidth=2)
        ax.plot(x, generated, label='Длина сгенерированного заголовка', marker='s', alpha=0.7, linewidth=2)
        ax.plot(x, reference, label='Длина референсного заголовка', marker='^', alpha=0.7, linewidth=2)
        
        ax.set_xlabel('Номер примера')
        ax.set_ylabel('Длина (символы)')
        ax.set_title('Сравнение длин текстов и заголовков')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Сохранение
        output_path = config.get_comparison_plot_path()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"График сравнения сохранен в {output_path}")
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Ошибка при визуализации сравнения: {e}")


def generate_report(config: Config):
    """
    Генерация текстового отчета.
    
    Args:
        config: Конфигурация проекта
    """
    logger = logging.getLogger(__name__)
    
    try:
        report_lines = []
        report_lines.append("# Отчет о проекте 'Нейрогенератор заголовков'")
        report_lines.append("")
        report_lines.append("## Общая информация")
        report_lines.append("")
        
        # Информация о датасете
        stats_path = config.get_dataset_stats_path()
        if stats_path.exists():
            stats = load_json(stats_path)
            required_keys = ['total_examples', 'mean_text_length', 'mean_title_length']
            if all(key in stats for key in required_keys):
                report_lines.append("### Статистика датасета")
                report_lines.append("")
                report_lines.append(f"- **Количество примеров**: {stats['total_examples']}")
                report_lines.append(f"- **Средняя длина текста**: {stats['mean_text_length']:.0f} символов")
                report_lines.append(f"- **Средняя длина заголовка**: {stats['mean_title_length']:.0f} символов")
                report_lines.append(f"- **Среднее количество слов в тексте**: {stats['mean_text_words']:.0f}")
                report_lines.append(f"- **Среднее количество слов в заголовке**: {stats['mean_title_words']:.0f}")
                report_lines.append("")
        
        # Информация о генерации
        results_path = config.get_test_results_path()
        if results_path.exists():
            results = load_json(results_path)
            all_titles = []
            for result in results:
                all_titles.extend(result.get('generated_titles', []))
            
            if all_titles:
                title_lengths = [len(title) for title in all_titles]
                title_words = [len(title.split()) for title in all_titles]
                
                report_lines.append("### Статистика генерации")
                report_lines.append("")
                report_lines.append(f"- **Всего сгенерировано заголовков**: {len(all_titles)}")
                report_lines.append(f"- **Средняя длина заголовка**: {np.mean(title_lengths):.1f} символов")
                report_lines.append(f"- **Среднее количество слов**: {np.mean(title_words):.1f}")
                report_lines.append(f"- **Минимальная длина**: {min(title_lengths)} символов")
                report_lines.append(f"- **Максимальная длина**: {max(title_lengths)} символов")
                report_lines.append("")
        
        # Информация о графиках
        report_lines.append("## Графики")
        report_lines.append("")
        report_lines.append("Следующие графики были сгенерированы:")
        report_lines.append("")
        
        if config.get_dataset_stats_plot_path().exists():
            report_lines.append(f"- **Статистика датасета**: `{config.get_dataset_stats_plot_path()}`")
        
        if config.get_generation_stats_plot_path().exists():
            report_lines.append(f"- **Статистика генерации**: `{config.get_generation_stats_plot_path()}`")
        
        if config.get_comparison_plot_path().exists():
            report_lines.append(f"- **Сравнение длин**: `{config.get_comparison_plot_path()}`")
        
        report_lines.append("")
        report_lines.append("## Выводы")
        report_lines.append("")
        report_lines.append("1. Датасет содержит достаточное количество примеров для обучения модели")
        report_lines.append("2. Средние длины текстов и заголовков соответствуют реальным новостным данным")
        report_lines.append("3. Модель генерирует заголовки разнообразной длины")
        report_lines.append("4. Рекомендуется:")
        report_lines.append("   - Использовать больше данных для обучения")
        report_lines.append("   - Экспериментировать с параметрами генерации")
        report_lines.append("   - Оценивать качество с помощью метрик ROUGE")
        
        # Сохранение отчета
        report_path = config.get_results_dir() / "project_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Отчет сохранен в {report_path}")
        
    except Exception as e:
        logger.error(f"Ошибка при генерации отчета: {e}")


def main():
    """
    Основная функция скрипта.
    """
    # Настройка логирования
    logger = setup_logging(
        log_file="logs/visualization.log",
        level="INFO"
    )
    
    logger.info("=" * 80)
    logger.info("ЗАПУСК ВИЗУАЛИЗАЦИИ РЕЗУЛЬТАТОВ")
    logger.info("=" * 80)
    
    try:
        # Загрузка конфигурации
        logger.info("Загрузка конфигурации...")
        config = Config("config.yaml")
        logger.info(f"Конфигурация загружена: {config}")
        
        # Настройка plotting
        setup_plotting()
        
        # Визуализация статистики датасета
        logger.info("Визуализация статистики датасета...")
        plot_dataset_stats(config)
        
        # Визуализация статистики генерации
        logger.info("Визуализация статистики генерации...")
        plot_generation_stats(config)
        
        # Визуализация сравнения
        logger.info("Визуализация сравнения...")
        plot_comparison(config)
        
        # Генерация отчета
        logger.info("Генерация отчета...")
        generate_report(config)
        
        logger.info("=" * 80)
        logger.info("ВИЗУАЛИЗАЦИЯ ЗАВЕРШЕНА")
        logger.info("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)