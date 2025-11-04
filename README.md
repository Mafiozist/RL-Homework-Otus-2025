# RL-Homework

Домашние задания по Reinforcement Learning на PyTorch в рамках обучения на курсе RL в OTUS. Рабочая среда — JupyterLab. Пакеты и индекс для PyTorch (CUDA 12.1) заданы в `requirements.txt`.

## Запуск через скрипты

```bash
# сделать исполняемыми (один раз)
chmod +x scripts/*.sh

# 1) Подготовить окружение (.venv, зависимости, ядро Jupyter)
./scripts/setup.sh

# 2) Запустить JupyterLab (в консоли будет URL)
./scripts/run_lab.sh
```

Примечания:
- По умолчанию `scripts/setup.sh` устанавливает зависимости из `requirements-gpu.txt`. Если такого файла нет, либо создайте его на основе `requirements.txt`, либо замените в скрипте строку на `pip install -r requirements.txt`.
- `scripts/run_lab.sh` использует `--no-browser`. Откройте вручную `http://127.0.0.1:8888/lab`, либо уберите флаг в скрипте, чтобы браузер открывался автоматически.

## Быстрый старт

```bash
# 1) Создать venv
python3 -m venv .venv

# 2) Активировать
source .venv/bin/activate

# 3) Установить зависимости
pip install --upgrade pip
pip install -r requirements.txt

# 4) Запустить JupyterLab (откроется сайт в браузере)
jupyter lab
```

В браузере откроется интерфейс по адресу `http://127.0.0.1:8888/lab`. Создайте новый Notebook и выберите ядро “Python (rl-homework)” при необходимости.

## Ядро для Jupyter

Чтобы ядро из venv было видно в Jupyter:
```bash
python -m ipykernel install --user --name rl-homework --display-name "Python (rl-homework)"
```

## Проверка PyTorch и CUDA

```bash
python - << 'PY'
import torch
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "is_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device 0:", torch.cuda.get_device_name(0))
PY
```

Если GPU недоступен или драйвер не подходит, можно перейти на CPU‑сборки PyTorch, заменив индекс в `requirements.txt` на `https://download.pytorch.org/whl/cpu` и переустановив пакеты.

## Полезные команды

```bash
pip check
pip install pipdeptree && pipdeptree --warn fail
pip freeze > requirements-lock.txt
```
