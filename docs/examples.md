# Примеры использования проекта

## 1. Обучение модели

### Запуск обучения

```bash
python scripts/train.py
```

### Что происходит во время обучения:

1. **Загрузка конфигурации** из `config.yaml`
2. **Загрузка датасета** с удалённого источника
3. **Предобработка данных** (удаление пустых значений, дубликатов)
4. **Загрузка модели** RuGPT-3 Small
5. **Обучение модели** на 10 000 примеров
6. **Сохранение модели** в папку `results/final_model`
7. **Логирование** в `logs/training.log`

### Пример вывода:

```
2026-02-06 14:00:00 - __main__ - INFO - Запуск обучения модели GPT для генерации заголовков
2026-02-06 14:00:01 - __main__ - INFO - 1. Загрузка конфигурации...
2026-02-06 14:00:02 - __main__ - INFO - Конфигурация загружена: Config(path=config.yaml)
2026-02-06 14:00:03 - __main__ - INFO - 2. Загрузка и подготовка данных...
2026-02-06 14:00:10 - __main__ - INFO - Информация о датасете: {'length': 10000, 'columns': ['title', 'text'], ...}
2026-02-06 14:00:15 - __main__ - INFO - 3. Загрузка модели...
2026-02-06 14:00:30 - __main__ - INFO - Информация о модели: {'name': 'ai-forever/rugpt3small_based_on_gpt2', 'parameters': 124000000, ...}
2026-02-06 14:00:35 - __main__ - INFO - 4. Обучение модели...
2026-02-06 14:00:40 - src.trainer - INFO - Настройка аргументов обучения...
2026-02-06 14:00:41 - src.trainer - INFO - Настройка оптимизатора...
2026-02-06 14:00:42 - src.trainer - INFO - Настройка тренера...
2026-02-06 14:00:43 - src.trainer - INFO - Запуск обучения...
2026-02-06 14:00:45 - transformers.trainer - INFO -   Num examples = 10000
2026-02-06 14:00:45 - transformers.trainer - INFO -   Num Epochs = 4
2026-02-06 14:00:45 - transformers.trainer - INFO -   Batch size = 16
2026-02-06 14:00:45 - transformers.trainer - INFO -   Total optimization steps = 2500
...
2026-02-06 15:30:00 - src.trainer - INFO - Обучение завершено
2026-02-06 15:30:01 - __main__ - INFO - 5. Сохранение модели...
2026-02-06 15:30:05 - src.trainer - INFO - Сохранение модели в results/final_model...
2026-02-06 15:30:10 - __main__ - INFO - Модель сохранена
2026-02-06 15:30:11 - __main__ - INFO - 6. Очистка памяти...
2026-02-06 15:30:12 - src.trainer - INFO - Память тренера очищена
2026-02-06 15:30:13 - src.model - INFO - Память очищена
2026-02-06 15:30:14 - __main__ - INFO - Память очищена
2026-02-06 15:30:15 - __main__ - INFO - Обучение успешно завершено!
```

## 2. Генерация заголовков

### Интерактивный режим

```bash
python scripts/inference.py --interactive
```

### Пример использования:

```
Введите текст новости: На Красной площади прошёл праздничный парад в честь Дня города. Сегодня, около 11:00, на Красной площади начался торжественный парад...

Сгенерированные заголовки:
1. Парад на Красной площади в День города
2. Праздничный парад на Красной площади
3. День города: парад на Красной площади
```

### Генерация из файла

```bash
python scripts/inference.py --input scripts/example_data.json --output results/generated_titles.json
```

### Формат входного файла (JSON):

```json
[
  {
    "text": "На Красной площади прошёл праздничный парад...",
    "title": "Парад на Красной площади в День города"
  },
  {
    "text": "В центре Москвы открылся новый парк...",
    "title": "В Москве открылся новый экологический парк"
  }
]
```

### Формат выходного файла (JSON):

```json
[
  {
    "text": "На Красной площади прошёл праздничный парад...",
    "generated_title": "Парад на Красной площади в День города",
    "reference_title": "Парад на Красной площади в День города"
  },
  {
    "text": "В центре Москвы открылся новый парк...",
    "generated_title": "В Москве открылся новый экологический парк",
    "reference_title": "В Москве открылся новый экологический парк"
  }
]
```

## 3. Использование в коде

### Загрузка и использование модели

```python
from src.config import Config
from src.model import ModelLoader
from src.inference import Inference

# Загрузка конфигурации
config = Config("config.yaml")

# Загрузка модели
model_loader = ModelLoader(config)
model, tokenizer, _ = model_loader.prepare_model_for_training()

# Создание инференса
inference = Inference(config, model, tokenizer)

# Генерация заголовка
text = "На Красной площади прошёл праздничный парад в честь Дня города..."
titles = inference.generate_titles(text)

print("Сгенерированные заголовки:")
for i, title in enumerate(titles, 1):
    print(f"{i}. {title}")

# Очистка памяти
inference.cleanup()
```

### Обучение модели в коде

```python
from src.config import Config
from src.dataset import DatasetLoader
from src.model import ModelLoader
from src.trainer import Trainer

# Загрузка конфигурации
config = Config("config.yaml")

# Загрузка данных
dataset_loader = DatasetLoader(config)
train_dataset = dataset_loader.prepare_dataset()

# Загрузка модели
model_loader = ModelLoader(config)
model, tokenizer, data_collator = model_loader.prepare_model_for_training()

# Обучение
trainer = Trainer(config, model, tokenizer, data_collator, train_dataset)
trainer.train()

# Сохранение модели
trainer.save_model()

# Очистка памяти
trainer.cleanup()
model_loader.cleanup()
```

## 4. Настройка параметров

### Изменение параметров обучения

В файле `config.yaml` можно изменить:

```yaml
training:
  epochs: 4                    # Количество эпох
  batch_size: 16               # Размер батча
  learning_rate: 2e-5          # Скорость обучения
  gradient_accumulation_steps: 2  # Накопление градиента
  warmup_ratio: 0.1            # Коэффициент warmup
  weight_decay: 0.01           # Регуляризация
  seed: 42                     # Семя для воспроизводимости
  fp16: true                   # Смешанная точность
  num_train_examples: 10000    # Количество примеров для обучения
```

### Изменение параметров генерации

```yaml
generation:
  num_beams: 1                 # Количество лучей в beam search
  no_repeat_ngram_size: 2      # Размер n-грамма для избежания повторов
  num_return_sequences: 3      # Количество вариантов заголовков
  temperature: 0.7             # Температура генерации
  top_k: 20                    # Top-k sampling
  top_p: 0.9                   # Top-p sampling (nucleus sampling)
```

### Изменение модели

```yaml
model:
  name: "ai-forever/rugpt3small_based_on_gpt2"  # Модель для обучения
  max_length: 320              # Максимальная длина входного текста
  max_new_tokens: 128          # Максимальное количество новых токенов
```

## 5. Визуализация результатов

### Создание графиков

```python
import matplotlib.pyplot as plt
import json

# Загрузка результатов
with open('results/generated_titles.json', 'r', encoding='utf-8') as f:
    results = json.load(f)

# Создание графика
texts = [r['text'][:100] + '...' for r in results]
generated = [r['generated_title'] for r in results]
reference = [r['reference_title'] for r in results]

fig, ax = plt.subplots(figsize=(12, 8))
x = range(len(results))

ax.plot(x, [len(t) for t in texts], label='Длина текста', marker='o')
ax.plot(x, [len(g) for g in generated], label='Длина сгенерированного заголовка', marker='s')
ax.plot(x, [len(r) for r in reference], label='Длина референсного заголовка', marker='^')

ax.set_xlabel('Номер примера')
ax.set_ylabel('Длина (символы)')
ax.set_title('Сравнение длин текстов и заголовков')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('images/comparison_plot.png', dpi=300, bbox_inches='tight')
plt.show()
```

### Сохранение в Markdown

```python
# Создание отчета
report = "# Отчет о генерации заголовков\n\n"
report += f"Всего примеров: {len(results)}\n\n"
report += "## Результаты\n\n"

for i, result in enumerate(results, 1):
    report += f"### Пример {i}\n\n"
    report += f"**Текст:** {result['text'][:200]}...\n\n"
    report += f"**Сгенерированный заголовок:** {result['generated_title']}\n\n"
    report += f"**Референсный заголовок:** {result['reference_title']}\n\n"
    report += "---\n\n"

with open('results/report.md', 'w', encoding='utf-8') as f:
    f.write(report)
```

## 6. Дополнительные примеры

### Аугментация данных

```python
from src.dataset import DatasetLoader
from src.config import Config

config = Config("config.yaml")
loader = DatasetLoader(config)

# Загрузка данных
dataset = loader.prepare_dataset()

# Аугментация (пример)
def augment_text(text):
    """Аугментация текста для увеличения датасета."""
    import random
    
    # Случайное удаление слов
    words = text.split()
    if len(words) > 10:
        num_to_remove = random.randint(1, 3)
        indices = random.sample(range(len(words)), num_to_remove)
        words = [w for i, w in enumerate(words) if i not in indices]
        return ' '.join(words)
    
    return text

# Применение аугментации
augmented_texts = [augment_text(example['text']) for example in dataset]
```

### Оценка качества

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

scores = []
for result in results:
    score = scorer.score(result['reference_title'], result['generated_title'])
    scores.append(score)

# Средние значения
avg_rouge1 = sum(s['rouge1'].fmeasure for s in scores) / len(scores)
avg_rouge2 = sum(s['rouge2'].fmeasure for s in scores) / len(scores)
avg_rougeL = sum(s['rougeL'].fmeasure for s in scores) / len(scores)

print(f"ROUGE-1: {avg_rouge1:.3f}")
print(f"ROUGE-2: {avg_rouge2:.3f}")
print(f"ROUGE-L: {avg_rougeL:.3f}")
```

## 7. Интеграция с веб-приложением

### FastAPI пример

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.config import Config
from src.model import ModelLoader
from src.inference import Inference

app = FastAPI(title="News Title Generator")

# Загрузка модели при старте
config = Config("config.yaml")
model_loader = ModelLoader(config)
model, tokenizer, _ = model_loader.prepare_model_for_training()
inference = Inference(config, model, tokenizer)

class NewsRequest(BaseModel):
    text: str
    num_titles: int = 3

class NewsResponse(BaseModel):
    titles: list[str]

@app.post("/generate", response_model=NewsResponse)
async def generate_titles(request: NewsRequest):
    try:
        titles = inference.generate_titles(
            request.text,
            num_return_sequences=request.num_titles
        )
        return NewsResponse(titles=titles)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": inference.model is not None}
```

### Запуск сервера

```bash
uvicorn web_app:app --host 0.0.0.0 --port 8000
```

### Использование API

```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"text": "На Красной площади прошёл праздничный парад...", "num_titles": 3}'
```

## 8. Мониторинг обучения

### TensorBoard

```bash
tensorboard --logdir logs/
```

### Просмотр логов

```bash
tail -f logs/training.log
```

### Просмотр метрик

```python
import pandas as pd
import matplotlib.pyplot as plt

# Чтение логов
logs = []
with open('logs/training.log', 'r') as f:
    for line in f:
        if 'loss' in line:
            logs.append(line)

# Парсинг логов
loss_values = []
for log in logs:
    try:
        loss = float(log.split('loss=')[1].split()[0])
        loss_values.append(loss)
    except:
        pass

# График потерь
plt.figure(figsize=(10, 6))
plt.plot(loss_values)
plt.xlabel('Шаг')
plt.ylabel('Loss')
plt.title('График потерь во время обучения')
plt.grid(True, alpha=0.3)
plt.savefig('images/loss_plot.png', dpi=300, bbox_inches='tight')
plt.show()
```

## 9. Эксперименты

### Сравнение моделей

```python
models = [
    "ai-forever/rugpt3small_based_on_gpt2",
    "ai-forever/rugpt3medium_based_on_gpt2",
    "sberbank-ai/rugpt3small"
]

results = {}

for model_name in models:
    config.model['name'] = model_name
    # Обучение и оценка
    # ...
```

### Подбор гиперпараметров

```python
from sklearn.model_selection import ParameterGrid

param_grid = {
    'learning_rate': [1e-5, 2e-5, 5e-5],
    'batch_size': [8, 16, 32],
    'epochs': [3, 4, 5]
}

for params in ParameterGrid(param_grid):
    config.training.update(params)
    # Обучение и оценка
    # ...
```

## 10. Деплой

### Docker

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "scripts/train.py"]
```

### Запуск в Docker

```bash
docker build -t news-title-generator .
docker run -v $(pwd)/results:/app/results news-title-generator
```

### Облачный деплой (AWS/GCP)

```bash
# Загрузка модели в S3
aws s3 cp results/final_model s3://my-bucket/models/news-title-generator/ --recursive

# Запуск на EC2
aws ec2 run-instances \
  --image-id ami-12345678 \
  --instance-type g4dn.xlarge \
  --key-name my-key \
  --security-group-ids sg-12345678 \
  --user-data file://user-data.sh