# Установка и настройка проекта

## Системные требования

### Минимальные требования

- **Операционная система**: Windows 10/11, Linux (Ubuntu 20.04+), macOS 10.15+
- **Python**: 3.8+
- **RAM**: 16 GB (рекомендуется 32 GB)
- **GPU**: NVIDIA GPU с поддержкой CUDA (рекомендуется 8+ GB VRAM)
- **Дисковое пространство**: 20 GB (для данных, моделей и результатов)

### Рекомендуемые требования

- **Операционная система**: Linux (Ubuntu 22.04+)
- **Python**: 3.9+
- **RAM**: 32 GB
- **GPU**: NVIDIA RTX 3090/4090 (24 GB VRAM)
- **Дисковое пространство**: 50 GB (SSD рекомендуется)

## Установка Python

### Windows

1. Скачайте Python 3.9+ с [официального сайта](https://www.python.org/downloads/)
2. Установите Python, отметив галочку "Add Python to PATH"
3. Проверьте установку:
   ```bash
   python --version
   pip --version
   ```

### Linux (Ubuntu)

```bash
# Обновление пакетов
sudo apt update && sudo apt upgrade -y

# Установка Python
sudo apt install python3.9 python3.9-venv python3.9-dev -y

# Установка pip
sudo apt install python3-pip -y

# Проверка установки
python3.9 --version
pip3 --version
```

### macOS

```bash
# Установка Homebrew (если не установлен)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Установка Python
brew install python@3.9

# Проверка установки
python3 --version
pip3 --version
```

## Установка CUDA (для GPU)

### Windows

1. Скачайте CUDA Toolkit 11.8 с [официального сайта NVIDIA](https://developer.nvidia.com/cuda-11-8-0-download-archive)
2. Установите CUDA Toolkit
3. Скачайте cuDNN 8.6+ с [официального сайта NVIDIA](https://developer.nvidia.com/cudnn)
4. Распакуйте cuDNN и скопируйте файлы в папку CUDA

### Linux (Ubuntu)

```bash
# Установка CUDA Toolkit 11.8
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run

# Добавление в PATH
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Установка cuDNN
# Скачайте cuDNN с сайта NVIDIA и распакуйте:
sudo tar -xvf cudnn-linux-x86_64-8.6.0.163_cuda11-archive.tar.xz
sudo cp cudnn-*-archive/include/cudnn*.h /usr/local/cuda/include
sudo cp -P cudnn-*-archive/lib/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

### Проверка установки CUDA

```bash
nvidia-smi
nvcc --version
```

## Клонирование проекта

```bash
git clone https://github.com/your-username/news-title-generator.git
cd news-title-generator
```

## Создание виртуального окружения

### Windows

```bash
# Создание виртуального окружения
python -m venv venv

# Активация
venv\Scripts\activate

# Обновление pip
python -m pip install --upgrade pip
```

### Linux/macOS

```bash
# Создание виртуального окружения
python3.9 -m venv venv

# Активация
source venv/bin/activate

# Обновление pip
python -m pip install --upgrade pip
```

## Установка зависимостей

### Базовая установка

```bash
# Активация виртуального окружения (если не активировано)
# Windows: venv\Scripts\activate
# Linux/macOS: source venv/bin/activate

# Установка зависимостей
pip install -r requirements.txt
```

### Установка с GPU поддержкой

```bash
# Установка PyTorch с GPU поддержкой
# Для CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Установка остальных зависимостей
pip install -r requirements.txt
```

### Установка дополнительных библиотек (опционально)

```bash
# Для визуализации
pip install matplotlib seaborn plotly

# Для оценки качества
pip install rouge-score

# Для веб-интерфейса
pip install fastapi uvicorn

# Для Jupyter ноутбуков
pip install jupyter ipykernel
```

## Проверка установки

### Проверка PyTorch

```python
import torch

print(f"PyTorch версия: {torch.__version__}")
print(f"Доступен GPU: {torch.cuda.is_available()}")
print(f"Название GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Нет'}")
print(f"Всего GPU: {torch.cuda.device_count()}")
```

### Проверка библиотек

```python
import transformers
import datasets
import pandas as pd
import numpy as np

print(f"Transformers версия: {transformers.__version__}")
print(f"Datasets версия: {datasets.__version__}")
print(f"Pandas версия: {pd.__version__}")
print(f"NumPy версия: {np.__version__}")
```

## Настройка HuggingFace

### Регистрация

1. Перейдите на [huggingface.co](https://huggingface.co)
2. Зарегистрируйтесь (можно через GitHub или Google)
3. Перейдите в [Settings → Access Tokens](https://huggingface.co/settings/tokens)
4. Создайте новый токен с правами `read`

### Установка токена

#### Способ 1: Через командную строку

```bash
huggingface-cli login
```

Вставьте ваш токен при запросе.

#### Способ 2: Через переменную окружения

```bash
# Windows (PowerShell)
$env:HF_TOKEN="your_token_here"

# Linux/macOS
export HF_TOKEN="your_token_here"
```

#### Способ 3: Через код

```python
from huggingface_hub import login
login(token="your_token_here")
```

## Настройка проекта

### Копирование конфигурации

```bash
# Файл config.yaml уже создан, но можно скопировать из примера
cp config.example.yaml config.yaml
```

### Настройка параметров

Откройте `config.yaml` и настройте параметры:

```yaml
# Настройки модели
model:
  name: "ai-forever/rugpt3small_based_on_gpt2"  # Модель для обучения
  max_length: 320              # Максимальная длина входного текста
  max_new_tokens: 128          # Максимальное количество новых токенов

# Настройки обучения
training:
  epochs: 4                    # Количество эпох
  batch_size: 16               # Размер батча (уменьшите до 8 или 4 если не хватает VRAM)
  learning_rate: 2e-5          # Скорость обучения
  gradient_accumulation_steps: 2  # Накопление градиента
  warmup_ratio: 0.1            # Коэффициент warmup
  weight_decay: 0.01           # Регуляризация
  seed: 42                     # Семя для воспроизводимости
  fp16: true                   # Смешанная точность (только для GPU)
  num_train_examples: 10000    # Количество примеров для обучения

# Настройки устройства
device:
  use_gpu: true                # Использовать GPU
  gpu_id: 0                    # ID GPU (если несколько)
  auto_device: true            # Автоматический выбор устройства
```

## Запуск проекта

### 1. Обучение модели

```bash
# Активация виртуального окружения
# Windows: venv\Scripts\activate
# Linux/macOS: source venv/bin/activate

# Запуск обучения
python scripts/train.py
```

### 2. Генерация заголовков (интерактивный режим)

```bash
python scripts/inference.py --interactive
```

### 3. Генерация заголовков из файла

```bash
python scripts/inference.py --input scripts/example_data.json --output results/generated_titles.json
```

## Устранение проблем

### Проблема 1: "Out of Memory" (не хватает VRAM)

**Решение:**
1. Уменьшите `batch_size` в `config.yaml` до 8 или 4
2. Увеличьте `gradient_accumulation_steps` до 4 или 8
3. Отключите `fp16` (установите `false`)
4. Используйте меньшую модель:
   ```yaml
   model:
     name: "ai-forever/rugpt3small_based_on_gpt2"  # Маленькая модель
   ```

### Проблема 2: Ошибка загрузки модели

**Решение:**
1. Проверьте подключение к интернету
2. Убедитесь, что токен HuggingFace корректный
3. Попробуйте загрузить модель вручную:
   ```python
   from transformers import AutoModelForCausalLM
   model = AutoModelForCausalLM.from_pretrained("ai-forever/rugpt3small_based_on_gpt2")
   ```

### Проблема 3: Ошибка CUDA

**Решение:**
1. Проверьте совместимость версий:
   ```bash
   python -c "import torch; print(torch.__version__)"
   nvcc --version
   nvidia-smi
   ```
2. Переустановите PyTorch с правильной версией CUDA:
   ```bash
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

### Проблема 4: Ошибка памяти RAM

**Решение:**
1. Уменьшите `num_train_examples` в `config.yaml`
2. Закройте другие приложения
3. Используйте `gradient_checkpointing`:
   ```python
   model.gradient_checkpointing_enable()
   ```

### Проблема 5: Ошибка загрузки датасета

**Решение:**
1. Проверьте URL датасета в `config.yaml`
2. Скачайте датасет вручную:
   ```bash
   wget https://storage.yandexcloud.net/google-colab-bucket/Datasets/russiannews_dataset.zip -O data/russiannews_dataset.zip
   unzip data/russiannews_dataset.zip -d data/dataset/
   ```

## Проверка окружения

### Полная проверка

```bash
# Активация виртуального окружения
# Windows: venv\Scripts\activate
# Linux/macOS: source venv/bin/activate

# Проверка Python
python --version

# Проверка PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'GPU: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# Проверка библиотек
python -c "import transformers, datasets, pandas, numpy; print('Все библиотеки установлены')"

# Проверка HuggingFace
python -c "from huggingface_hub import whoami; print('HuggingFace доступен')"

# Проверка проекта
python -c "from src.config import Config; print('Проект доступен')"
```

## Дополнительные настройки

### Для Windows

#### Установка Visual Studio Build Tools

1. Скачайте [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. Установите с компонентом "Desktop development with C++"

#### Установка Git

1. Скачайте [Git для Windows](https://git-scm.com/download/win)
2. Установите и настройте

### Для Linux

#### Установка системных зависимостей

```bash
sudo apt update
sudo apt install -y build-essential python3-dev python3-pip python3-venv
sudo apt install -y wget unzip curl git
```

### Для macOS

#### Установка Xcode Command Line Tools

```bash
xcode-select --install
```

## Быстрый старт (Docker)

### Установка Docker

#### Windows
1. Скачайте [Docker Desktop](https://www.docker.com/products/docker-desktop/)
2. Установите и запустите

#### Linux
```bash
# Установка Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
# Перелогиньтесь
```

#### macOS
```bash
# Установка Docker Desktop
brew install --cask docker
```

### Запуск в Docker

```bash
# Сборка образа
docker build -t news-title-generator .

# Запуск контейнера
docker run -it --gpus all -v $(pwd)/results:/app/results news-title-generator

# Запуск без GPU
docker run -it -v $(pwd)/results:/app/results news-title-generator
```

## Проверка работы

### Тестовый запуск

```bash
# Активация виртуального окружения
# Windows: venv\Scripts\activate
# Linux/macOS: source venv/bin/activate

# Запуск теста
python -c "
from src.config import Config
from src.dataset import DatasetLoader
from src.model import ModelLoader
from src.inference import Inference

print('Загрузка конфигурации...')
config = Config('config.yaml')

print('Загрузка данных...')
loader = DatasetLoader(config)
dataset = loader.prepare_dataset()
print(f'Датасет загружен: {len(dataset)} примеров')

print('Загрузка модели...')
model_loader = ModelLoader(config)
model, tokenizer, _ = model_loader.prepare_model_for_training()
print('Модель загружена')

print('Тест генерации...')
inference = Inference(config, model, tokenizer)
text = 'На Красной площади прошёл праздничный парад в честь Дня города.'
titles = inference.generate_titles(text)
print(f'Сгенерировано заголовков: {len(titles)}')
for i, title in enumerate(titles, 1):
    print(f'{i}. {title}')

inference.cleanup()
print('Тест завершен успешно!')
"
```

## Готово!

После выполнения всех шагов вы готовы к:

1. **Обучению модели** на собственных данных
2. **Генерации заголовков** к новостным текстам
3. **Экспериментам** с параметрами и моделями
4. **Деплою** в production

## Следующие шаги

- [Прочитать README.md](../README.md)
- [Изучить примеры использования](examples.md)
- [Посмотреть информацию о датасете](dataset_info.md)
- [Настроить параметры обучения](../config.yaml)