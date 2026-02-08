# Используем официальный образ Python
FROM python:3.9-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Установка рабочей директории
WORKDIR /app

# Копирование requirements
COPY requirements.txt .

# Установка Python зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Копирование всего проекта
COPY . .

# Создание необходимых директорий
RUN mkdir -p data results logs

# Установка прав доступа
RUN chmod +x scripts/*.py

# Экспорт порта для веб-приложения
EXPOSE 8000

# Запуск по умолчанию (можно переопределить при запуске контейнера)
CMD ["python", "scripts/train.py"]