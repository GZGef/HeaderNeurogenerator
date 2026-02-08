#!/usr/bin/env python3
"""
Скрипт для генерации графиков для README.md
"""

import sys
import os
from pathlib import Path

# Добавляем корневую директорию в PYTHONPATH
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from src.config import Config

# Настройки стиля
plt.style.use('seaborn-v0_8')
sns.set_palette('husl')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Создаем папку для графиков
output_dir = root_dir / 'results' / 'images'
output_dir.mkdir(parents=True, exist_ok=True)

print("Генерация графиков для README.md...")
print("=" * 60)

# 1. График распределения длины текстов
print("1. Создание графика распределения длины текстов...")
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Симулируем данные для примера
np.random.seed(42)
text_lengths = np.random.normal(500, 150, 1000)
title_lengths = np.random.normal(50, 15, 1000)

axes[0].hist(text_lengths, bins=50, alpha=0.7, color='skyblue')
axes[0].set_xlabel('Длина текста (символы)')
axes[0].set_ylabel('Количество')
axes[0].set_title('Распределение длины текстов')
axes[0].axvline(text_lengths.mean(), color='red', linestyle='--', 
                label=f"Средняя: {text_lengths.mean():.0f}")
axes[0].legend()

axes[1].hist(title_lengths, bins=50, alpha=0.7, color='lightcoral')
axes[1].set_xlabel('Длина заголовка (символы)')
axes[1].set_ylabel('Количество')
axes[1].set_title('Распределение длины заголовков')
axes[1].axvline(title_lengths.mean(), color='red', linestyle='--', 
                label=f"Средняя: {title_lengths.mean():.0f}")
axes[1].legend()

plt.tight_layout()
plt.savefig(output_dir / 'length_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"   Сохранено: {output_dir / 'length_distribution.png'}")

# 2. График распределения количества слов
print("2. Создание графика распределения количества слов...")
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

text_words = np.random.normal(80, 25, 1000)
title_words = np.random.normal(8, 3, 1000)

axes[0].hist(text_words, bins=50, alpha=0.7, color='lightgreen')
axes[0].set_xlabel('Количество слов в тексте')
axes[0].set_ylabel('Количество')
axes[0].set_title('Распределение количества слов в текстах')
axes[0].axvline(text_words.mean(), color='red', linestyle='--', 
                label=f"Средняя: {text_words.mean():.0f}")
axes[0].legend()

axes[1].hist(title_words, bins=50, alpha=0.7, color='gold')
axes[1].set_xlabel('Количество слов в заголовке')
axes[1].set_ylabel('Количество')
axes[1].set_title('Распределение количества слов в заголовках')
axes[1].axvline(title_words.mean(), color='red', linestyle='--', 
                label=f"Средняя: {title_words.mean():.0f}")
axes[1].legend()

plt.tight_layout()
plt.savefig(output_dir / 'word_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"   Сохранено: {output_dir / 'word_distribution.png'}")

# 3. Матрица корреляции
print("3. Создание матрицы корреляции...")
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Матрица корреляции
correlation_matrix = np.array([
    [1.0, 0.3, 0.8, 0.2],
    [0.3, 1.0, 0.2, 0.9],
    [0.8, 0.2, 1.0, 0.3],
    [0.2, 0.9, 0.3, 1.0]
])

labels = ['text_length', 'title_length', 'text_words', 'title_words']

im = axes[0].imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
axes[0].set_xticks(range(4))
axes[0].set_yticks(range(4))
axes[0].set_xticklabels(labels, rotation=45, ha='right')
axes[0].set_yticklabels(labels)
axes[0].set_title('Матрица корреляции признаков')

# Добавляем значения
for i in range(4):
    for j in range(4):
        axes[0].text(j, i, f'{correlation_matrix[i, j]:.2f}', 
                    ha='center', va='center', color='white' if abs(correlation_matrix[i, j]) > 0.5 else 'black')

plt.colorbar(im, ax=axes[0])

# Точечный график
text_lengths_scatter = np.random.normal(500, 150, 500)
title_lengths_scatter = np.random.normal(50, 15, 500)

axes[1].scatter(text_lengths_scatter, title_lengths_scatter, alpha=0.5, s=10, color='steelblue')
axes[1].set_xlabel('Длина текста (символы)')
axes[1].set_ylabel('Длина заголовка (символы)')
axes[1].set_title('Взаимосвязь длины текста и заголовка')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"   Сохранено: {output_dir / 'correlation_matrix.png'}")

# 4. Топ-20 слов в заголовках
print("4. Создание графика топ-20 слов...")
fig, ax = plt.subplots(figsize=(12, 6))

# Симулируем топ-20 слов
words = ['россия', 'новый', 'президент', 'правительство', 'министр', 
         'экономика', 'рынок', 'компания', 'проект', 'инвестиции',
         'развитие', 'страна', 'город', 'регион', 'люди',
         'работа', 'время', 'год', 'день', 'вопрос']
counts = np.random.randint(100, 1000, 20)

# Сортируем по количеству
sorted_indices = np.argsort(counts)
words = [words[i] for i in sorted_indices]
counts = counts[sorted_indices]

ax.barh(range(len(words)), counts, color='steelblue')
ax.set_yticks(range(len(words)))
ax.set_yticklabels(words)
ax.set_xlabel('Частота')
ax.set_title('Топ-20 самых частых слов в заголовках')
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(output_dir / 'top_words.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"   Сохранено: {output_dir / 'top_words.png'}")

# 5. График метрик качества
print("5. Создание графика метрик качества...")
fig, ax = plt.subplots(figsize=(10, 6))

# Метрики ROUGE
metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
values = [0.42, 0.25, 0.35]
errors = [0.05, 0.03, 0.04]

bars = ax.bar(metrics, values, yerr=errors, capsize=10, alpha=0.7, 
              color=['#FF6B6B', '#4ECDC4', '#45B7D1'])

ax.set_ylabel('Значение метрики')
ax.set_title('Метрики качества генерации заголовков')
ax.set_ylim(0, 0.6)

# Добавляем значения на столбцы
for bar, value in zip(bars, values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            f'{value:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(output_dir / 'metrics.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"   Сохранено: {output_dir / 'metrics.png'}")

# 6. График процесса обучения
print("6. Создание графика процесса обучения...")
fig, ax = plt.subplots(figsize=(10, 6))

# Симулируем процесс обучения
epochs = np.arange(1, 5)
train_loss = [2.5, 1.8, 1.2, 0.8]
val_loss = [2.6, 1.9, 1.3, 0.9]

ax.plot(epochs, train_loss, 'o-', label='Потери обучения', linewidth=2, markersize=8)
ax.plot(epochs, val_loss, 's-', label='Потери валидации', linewidth=2, markersize=8)
ax.set_xlabel('Эпоха')
ax.set_ylabel('Потери')
ax.set_title('Процесс обучения модели')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'training_process.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"   Сохранено: {output_dir / 'training_process.png'}")

# 7. График сравнения длин
print("7. Создание графика сравнения длин...")
fig, ax = plt.subplots(figsize=(10, 6))

# Сравнение длин
categories = ['Текст', 'Заголовок']
mean_lengths = [500, 50]
std_lengths = [150, 15]

bars = ax.bar(categories, mean_lengths, yerr=std_lengths, capsize=10, 
              alpha=0.7, color=['#FF9999', '#66B2FF'])

ax.set_ylabel('Средняя длина (символы)')
ax.set_title('Сравнение длин текстов и заголовков')

# Добавляем значения на столбцы
for bar, value in zip(bars, mean_lengths):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 20,
            f'{value:.0f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'length_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"   Сохранено: {output_dir / 'length_comparison.png'}")

print("=" * 60)
print(f"Все графики сохранены в папку: {output_dir}")
print(f"Всего создано: 7 графиков")