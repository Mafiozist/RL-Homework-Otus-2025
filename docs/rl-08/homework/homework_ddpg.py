"""
DDPG (Deep Deterministic Policy Gradient) для CarRacing-v3 с непрерывными действиями.

DDPG - это алгоритм обучения с подкреплением, специально разработанный для задач с непрерывным пространством действий.
В отличие от A2C, который использует стохастическую политику и работает on-policy, DDPG использует детерминистическую
политику и может обучаться off-policy, что делает его более эффективным в использовании данных.

Основные компоненты DDPG:
- Actor (политика): детерминистическая сеть, которая для каждого состояния возвращает конкретное действие
- Critic (Q-сеть): оценивает качество действия в состоянии через Q(s,a)
- Replay Buffer: хранит прошлый опыт для off-policy обучения
- Target Networks: стабилизируют обучение через медленное обновление
- Exploration Noise: добавляет шум к действиям для исследования пространства действий

Обучение происходит по шагам, а не по эпизодам. Агент собирает опыт в replay buffer и обучается на случайных
батчах из этого буфера, что разрывает временную корреляцию и делает обучение более стабильным.
"""

import os
import random
from dataclasses import dataclass
from collections import deque

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm.auto import trange


@dataclass
class Config:
    """
    Конфигурация для обучения DDPG на CarRacing.
    
    Содержит все гиперпараметры, необходимые для настройки алгоритма,
    включая параметры среды, обучения, архитектуры сетей и обработки изображений.
    """
    # Фиксируем seed для воспроизводимости результатов
    seed: int = 42

    # Используем непрерывную версию CarRacing
    env_id: str = "CarRacing-v3"
    continuous: bool = True
    lap_complete_percent: float = 0.95

    # Коэффициент дисконтирования будущих наград
    # Определяет, насколько важны будущие награды по сравнению с текущими
    gamma: float = 0.99

    # Скорости обучения для Actor и Critic
    # Обычно Critic обучается немного быстрее, чем Actor
    lr_actor: float = 1e-4
    lr_critic: float = 3e-4

    # Параметры для replay buffer
    buffer_size: int = 100000  # Максимальный размер буфера опыта
    batch_size: int = 512  # Размер батча для обучения

    # Параметры для target networks
    tau: float = 0.005  # Коэффициент мягкого обновления (очень мал для стабильности)

    # Параметры для exploration noise (Ornstein-Uhlenbeck)
    ou_theta: float = 0.15  # Скорость возврата к среднему
    ou_sigma: float = 0.3  # Волатильность шума (больше = больше исследование)
    ou_dt: float = 0.01  # Временной шаг для OU-процесса

    # Параметры обучения
    total_episodes: int = 3500  # Максимальное количество эпизодов
    train_freq: int = 1  # Частота обучения (каждые N шагов)
    train_steps: int = 1  # Количество шагов обучения за раз
    
    # Параметры для устойчивости обучения
    grad_clip_norm: float = 1.75  # Максимальная норма градиента для clipping

    # Окно для усреднения награды
    log_window: int = 100

    # Критерий остановки обучения
    stop_reward: float = 900.0

    # Параметры обработки изображения
    use_grayscale: bool = True  # Использовать оттенки серого для уменьшения размерности
    out_h: int = 84  # Высота обработанного изображения
    out_w: int = 84  # Ширина обработанного изображения

    # Сохранение результатов
    save_dir: str = "docs/rl-08/homework"
    model_name: str = "ddpg_carracing"
    plot_name: str = "ddpg_reward_plot.png"
    
    # Автоматическое сохранение и загрузка checkpoint
    auto_save_best: bool = True  # Автоматически сохранять при улучшении
    auto_load_checkpoint: bool = True  # Автоматически загружать веса при старте, если они есть
    improvement_threshold: float = 50.0  # Минимальное улучшение награды для сохранения (например, +50)
    save_best_only: bool = True  # Сохранять только лучшие веса (True) или также промежуточные (False)
    
    # Параметры для адаптивной смешанной выборки из буфера
    mixed_sampling_buffer_fill: float = 0.7  # Минимальная заполненность буфера для смешанной выборки (0.7 = 70%)
    mixed_sampling_good_ratio_start: float = 0.8  # Доля хороших примеров в начале (0.8 = 80% хороших, 20% случайных)
    mixed_sampling_good_ratio_end: float = 0.4  # Доля хороших примеров при достижении цели (0.4 = 40% хороших, 60% случайных)
    mixed_sampling_reward_offset: float = 50.0  # Дополнительный порог к среднему для хороших примеров (среднее + offset)

    # Выбираем GPU, если он доступен
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class CNNEncoder(nn.Module):
    """
    Сверточная нейронная сеть для извлечения признаков из изображений.
    
    Эта сеть получает на вход изображение трассы и преобразует его в компактный
    вектор признаков, который содержит информацию о:
    - форме и краях дороги
    - положении автомобиля относительно трассы
    - окружающей обстановке
    
    Архитектура использует последовательность сверточных слоев с уменьшением
    пространственного разрешения и увеличением количества фильтров, что позволяет
    сети выделять иерархические признаки от простых краев до сложных паттернов.
    """
    def __init__(self, in_channels: int, out_h: int, out_w: int):
        super().__init__()

        # Последовательность сверточных слоев для извлечения признаков
        # Уменьшенная архитектура для более быстрого обучения и меньшего количества параметров
        self.conv = nn.Sequential(
            # Первый слой: большая свертка с большим шагом для быстрого уменьшения размера
            nn.Conv2d(in_channels, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            # Второй слой: средняя свертка для дальнейшего сжатия
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            # Третий слой: маленькая свертка для тонкой обработки
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # Автоматически вычисляем размерность признаков после сверток
        # Это нужно, чтобы правильно настроить полносвязный слой
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, out_h, out_w)
            y = self.conv(dummy)
            feat_dim = int(np.prod(y.shape[1:]))  # Произведение всех размерностей кроме батча

        # Полносвязный слой для финального представления признаков
        self.fc = nn.Sequential(
            nn.Flatten(),  # Преобразуем многомерный тензор в вектор
            nn.Linear(feat_dim, 256),  # Сжимаем до 256 признаков
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через сеть.
        
        Args:
            x: Тензор изображений [batch_size, channels, height, width]
        
        Returns:
            Вектор признаков [batch_size, 256]
        """
        return self.fc(self.conv(x))


class ActorNet(nn.Module):
    """
    Сеть Actor (политика) для DDPG.
    
    В отличие от стохастических политик в A2C, эта сеть возвращает детерминистическое
    действие для каждого состояния. То есть для одного и того же состояния она всегда
    вернет одно и то же действие.
    
    Для CarRacing действия представляют собой трехмерный вектор:
    - руль: [-1, 1] (влево/вправо)
    - газ: [0, 1] (насколько сильно нажать на газ)
    - тормоз: [0, 1] (насколько сильно нажать на тормоз)
    
    Сеть использует tanh для руля (чтобы ограничить в [-1, 1]) и sigmoid для газа и тормоза
    (чтобы ограничить в [0, 1]).
    """
    def __init__(self, in_channels: int, action_dim: int, out_h: int, out_w: int):
        super().__init__()
        # Энкодер для извлечения признаков из изображения
        self.encoder = CNNEncoder(in_channels, out_h, out_w)
        # Промежуточный слой для более глубокого представления
        self.hidden = nn.Linear(256, 128)
        # Голова сети, которая преобразует признаки в действия
        self.head = nn.Linear(128, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через сеть политики.
        
        Args:
            x: Тензор изображений [batch_size, channels, height, width]
        
        Returns:
            Действия [batch_size, action_dim]
            Для CarRacing: [batch_size, 3] где [руль, газ, тормоз]
        """
        # Извлекаем признаки из изображения
        features = self.encoder(x)
        # Проходим через промежуточный слой
        hidden = F.relu(self.hidden(features))
        # Преобразуем признаки в действия
        actions = self.head(hidden)
        
        # Применяем активации для ограничения диапазона действий
        # Руль: tanh ограничивает в [-1, 1]
        # Газ и тормоз: sigmoid ограничивает в [0, 1]
        steering = torch.tanh(actions[:, 0:1])  # руль
        gas = torch.sigmoid(actions[:, 1:2])  # газ
        brake = torch.sigmoid(actions[:, 2:3])  # тормоз
        
        # Объединяем действия обратно в один тензор
        actions = torch.cat([steering, gas, brake], dim=1)
        
        return actions


class CriticNet(nn.Module):
    """
    Сеть Critic (Q-сеть) для DDPG.
    
    Эта сеть оценивает качество действия в состоянии через Q-функцию Q(s, a).
    В отличие от V-функции в A2C, которая оценивает только состояние, Q-функция
    оценивает пару (состояние, действие), что позволяет сравнивать разные действия
    в одном и том же состоянии.
    
    Архитектура: сеть принимает на вход признаки состояния (из CNN) и действие,
    объединяет их и выдает оценку Q(s, a) - ожидаемое будущее вознаграждение.
    """
    def __init__(self, in_channels: int, action_dim: int, out_h: int, out_w: int):
        super().__init__()
        # Энкодер для извлечения признаков из изображения
        self.encoder = CNNEncoder(in_channels, out_h, out_w)
        # Слой для объединения признаков состояния и действия
        self.fusion = nn.Linear(256 + action_dim, 128)
        # Промежуточный слой
        self.hidden = nn.Linear(128, 64)
        # Голова сети, которая выдает Q-значение
        self.head = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через Q-сеть.
        
        Args:
            x: Тензор изображений [batch_size, channels, height, width]
            a: Тензор действий [batch_size, action_dim]
        
        Returns:
            Q-значения [batch_size, 1]
        """
        # Извлекаем признаки из изображения
        state_features = self.encoder(x)
        # Объединяем признаки состояния и действие
        combined = torch.cat([state_features, a], dim=-1)
        # Проходим через слой объединения
        x = F.relu(self.fusion(combined))
        # Промежуточный слой
        x = F.relu(self.hidden(x))
        # Получаем Q-значение
        return self.head(x)


class ReplayBuffer:
    """
    Буфер воспроизведения (Replay Buffer) для off-policy обучения.
    
    Основная идея replay buffer - хранить прошлый опыт агента и обучаться на
    случайных батчах из этого опыта, а не только на последних данных. Это дает
    несколько важных преимуществ:
    
    1. Off-policy обучение: можем обучаться на данных, собранных старой политикой
    2. Разрыв временной корреляции: случайная выборка разрывает корреляцию между
       последовательными состояниями, что делает обучение более стабильным
    3. Эффективное использование данных: каждый опыт может быть использован многократно
    4. Стабильность: обучение на разнообразных данных из разных эпизодов
    
    Буфер работает по принципу FIFO (First In, First Out) - когда он заполнен,
    старые данные перезаписываются новыми.
    """
    def __init__(self, buffer_size: int):
        """
        Инициализация буфера.
        
        Args:
            buffer_size: Максимальный размер буфера
        """
        self.buffer_size = buffer_size
        self.buffer = []
        self._next_idx = 0  # Индекс для циклической записи

    def add(self, state, action, reward, next_state, done):
        """
        Добавляет новый опыт в буфер.
        
        Args:
            state: Текущее состояние
            action: Выполненное действие
            reward: Полученная награда
            next_state: Следующее состояние
            done: Флаг завершения эпизода
        """
        # Если буфер еще не заполнен, добавляем новый элемент
        if len(self.buffer) < self.buffer_size:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            # Иначе перезаписываем старый опыт (FIFO)
            self.buffer[self._next_idx] = (state, action, reward, next_state, done)
        
        # Циклически обновляем индекс
        self._next_idx = (self._next_idx + 1) % self.buffer_size

    def sample(self, batch_size: int, avg_reward: float = 0.0, stop_reward: float = 900.0,
               buffer_fill_threshold: float = 0.7, good_ratio_start: float = 0.8,
               good_ratio_end: float = 0.4, reward_offset: float = 50.0):
        """
        Выбирает батч опыта из буфера с адаптивным соотношением хороших/случайных примеров.
        
        Если буфер заполнен на buffer_fill_threshold (70%), использует смешанную выборку:
        - В начале (низкий avg_reward): больше хороших примеров (good_ratio_start, например 80%)
        - По мере приближения к stop_reward: больше случайных для исследования (good_ratio_end, например 40%)
        - Хорошие примеры: награды > (среднее + offset)
        - Остальное: случайная выборка для исследования
        
        Args:
            batch_size: Размер батча
            avg_reward: Средняя награда для определения порога и прогресса
            stop_reward: Целевая награда для вычисления прогресса
            buffer_fill_threshold: Минимальная заполненность буфера (0.7 = 70%)
            good_ratio_start: Доля хороших примеров в начале (0.8 = 80%)
            good_ratio_end: Доля хороших примеров при приближении к цели (0.4 = 40%)
            reward_offset: Дополнительный порог к среднему (среднее + offset)
        
        Returns:
            Кортеж (states, actions, rewards, next_states, dones)
        """
        if len(self.buffer) == 0:
            raise ValueError("Buffer is empty, cannot sample")
        
        # Проверяем условия для смешанной выборки:
        # 1. Буфер заполнен на buffer_fill_threshold (70%)
        # 2. Средняя награда строго положительная (> 0)
        buffer_fill_ratio = len(self.buffer) / self.buffer_size
        should_use_mixed = (buffer_fill_ratio >= buffer_fill_threshold and avg_reward > 0.0)
        
        if should_use_mixed:
            # Вычисляем прогресс: как близко avg_reward к stop_reward (0.0 = начало, 1.0 = достигли цели)
            # Ограничиваем сверху 1.0, чтобы не превышать
            progress = min(avg_reward / stop_reward, 1.0) if stop_reward > 0 else 0.0
            
            # Линейная интерполяция между good_ratio_start и good_ratio_end
            # В начале (progress=0): good_ratio = good_ratio_start (больше хороших)
            # При приближении к цели (progress→1): good_ratio = good_ratio_end (больше случайных)
            good_ratio = good_ratio_start * (1.0 - progress) + good_ratio_end * progress
            
            # Вычисляем количество хороших и случайных примеров
            num_good = int(batch_size * good_ratio)
            num_random = batch_size - num_good
            
            # Извлекаем все награды
            rewards_array = np.array([self.buffer[i][2] for i in range(len(self.buffer))])
            
            # Порог для хороших примеров: средняя награда + offset
            threshold = avg_reward + reward_offset
            
            # Находим индексы опыта с наградами выше порога И строго положительными (> 0)
            positive_mask = rewards_array > 0.0
            good_mask = positive_mask & (rewards_array > threshold)
            if good_mask.sum() > 0:
                good_indices = np.where(good_mask)[0].tolist()
                
                # Выбираем хорошие примеры
                if len(good_indices) >= num_good:
                    good_selected = random.sample(good_indices, num_good)
                else:
                    # Если хороших примеров меньше, берем все
                    good_selected = good_indices.copy()
                    # Дополняем до нужного количества случайными из хороших (с повторениями)
                    while len(good_selected) < num_good:
                        good_selected.append(random.choice(good_indices))
                
                # Выбираем случайные примеры из всего буфера (исключая уже выбранные хорошие)
                remaining_indices = [i for i in range(len(self.buffer)) if i not in good_selected]
                if len(remaining_indices) >= num_random:
                    random_selected = random.sample(remaining_indices, num_random)
                else:
                    # Если осталось мало, дополняем из всего буфера
                    random_selected = remaining_indices.copy()
                    while len(random_selected) < num_random:
                        random_selected.append(random.choice(range(len(self.buffer))))
                
                selected_indices = good_selected + random_selected
            else:
                # Если нет хороших примеров, используем полностью случайную выборку
                selected_indices = random.sample(range(len(self.buffer)), batch_size)
        else:
            # Полностью случайная выборка (если буфер не заполнен достаточно)
            selected_indices = random.sample(range(len(self.buffer)), batch_size)
        
        # Извлекаем данные по выбранным индексам
        states = [self.buffer[i][0] for i in selected_indices]
        actions = [self.buffer[i][1] for i in selected_indices]
        rewards = [self.buffer[i][2] for i in selected_indices]
        next_states = [self.buffer[i][3] for i in selected_indices]
        dones = [self.buffer[i][4] for i in selected_indices]
        
        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Возвращает текущий размер буфера."""
        return len(self.buffer)


class OrnsteinUhlenbeckActionNoise:
    """
    Генератор шума Орнштейна-Уленбека для исследования действий.
    
    В отличие от белого шума, который независим на каждом шаге, OU-шум имеет
    "память" - следующее значение зависит от предыдущего. Это делает шум более
    плавным и естественным для физических систем, таких как управление автомобилем.
    
    OU-процесс описывается стохастическим дифференциальным уравнением:
    dx_t = theta * (mu - x_{t-1}) * dt + sigma * sqrt(dt) * dW_t
    
    где:
    - theta: скорость возврата к среднему (чем больше, тем быстрее возврат)
    - mu: среднее значение (обычно 0)
    - sigma: волатильность шума (чем больше, тем больше амплитуда)
    - dt: временной шаг
    - dW_t: винеровский процесс (белый шум)
    
    Для исследования действий мы добавляем этот шум к детерминистическим действиям
    политики, что позволяет агенту исследовать пространство действий, не теряя
    при этом основного направления, задаваемого политикой.
    """
    def __init__(self, mu, sigma, theta=0.15, dt=1e-2, x0=None):
        """
        Инициализация OU-процесса.
        
        Args:
            mu: Среднее значение (обычно массив нулей размерности действия)
            sigma: Стандартное отклонение шума
            theta: Скорость возврата к среднему
            dt: Временной шаг
            x0: Начальное значение (если None, то 0)
        """
        self.theta = theta
        self.mu = np.asarray(mu)
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        """
        Генерирует следующее значение OU-шума.
        
        Returns:
            Новое значение шума
        """
        # Формула OU-процесса в дискретном виде
        x = (self.x_prev + 
             self.theta * (self.mu - self.x_prev) * self.dt + 
             self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape))
        self.x_prev = x
        return x

    def set_sigma(self, sigma):
        self.sigma=sigma

    def reset(self):
        """Сбрасывает процесс в начальное состояние."""
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)


class CarRacingDDPG:
    """
    Основной класс для обучения DDPG на CarRacing.
    
    Этот класс объединяет все компоненты DDPG:
    - Actor и Critic сети (основные и целевые)
    - Replay buffer для хранения опыта
    - OU-шум для исследования
    - Функции обучения и обновления целевых сетей
    
    Обучение происходит по шагам: агент собирает опыт, сохраняет его в буфер,
    и периодически обучается на случайных батчах из буфера.
    """
    def __init__(self, cfg: Config):
        self.cfg = cfg

        # Создаем среду CarRacing с непрерывными действиями
        self.env = gym.make(
            cfg.env_id,
            lap_complete_percent=cfg.lap_complete_percent,
            continuous=cfg.continuous,
        )

        # Проверяем, что действия действительно непрерывные
        assert isinstance(self.env.action_space, gym.spaces.Box)
        assert len(self.env.action_space.shape) == 1

        # Фиксируем случайность для воспроизводимости
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)

        # Получаем размерности действия и наблюдения
        self.action_dim = self.env.action_space.shape[0]  # Для CarRacing: 3
        self.in_channels = 1 if cfg.use_grayscale else 3

        print(f"Device: {cfg.device}, action_dim: {self.action_dim}")

        # Создаем основные сети (Actor и Critic)
        self.actor = ActorNet(self.in_channels, self.action_dim, cfg.out_h, cfg.out_w).to(cfg.device)
        self.critic = CriticNet(self.in_channels, self.action_dim, cfg.out_h, cfg.out_w).to(cfg.device)

        # Инициализация весов для лучшей сходимости
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                m.bias.data.fill_(0.01)
            elif isinstance(m, nn.Conv2d):
                torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    m.bias.data.fill_(0.01)

        self.actor.apply(init_weights)
        self.critic.apply(init_weights)

        # Создаем целевые сети (копии основных)
        # Целевые сети обновляются медленно и используются для стабильного обучения
        self.actor_target = ActorNet(self.in_channels, self.action_dim, cfg.out_h, cfg.out_w).to(cfg.device)
        self.critic_target = CriticNet(self.in_channels, self.action_dim, cfg.out_h, cfg.out_w).to(cfg.device)

        # Инициализируем целевые сети копированием весов основных сетей
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Отключаем градиенты для целевых сетей (они обновляются через soft update)
        for param in self.actor_target.parameters():
            param.requires_grad = False
        for param in self.critic_target.parameters():
            param.requires_grad = False

        # Создаем оптимизаторы для основных сетей
        self.actor_opt = optim.AdamW(self.actor.parameters(), lr=cfg.lr_actor)
        self.critic_opt = optim.AdamW(self.critic.parameters(), lr=cfg.lr_critic)

        # Создаем replay buffer
        self.buffer = ReplayBuffer(cfg.buffer_size)

        # Создаем генератор OU-шума для исследования
        # mu - нулевой вектор размерности действия
        # sigma - волатильность шума (из конфига)
        self.ou_noise = OrnsteinUhlenbeckActionNoise(
            mu=np.zeros(self.action_dim),
            sigma=cfg.ou_sigma,
            theta=cfg.ou_theta,
            dt=cfg.ou_dt
        )

        # Для логирования результатов
        self.episode_rewards = []
        self.recent_rewards = deque(maxlen=cfg.log_window)

        # Для отслеживания лучшей награды и автоматического сохранения
        self.best_reward = float('-inf')  # Лучшая средняя награда за последние N эпизодов
        self.best_episode = 0  # Номер эпизода с лучшей наградой
        self.last_saved_reward = float('-inf')  # Последняя награда, при которой было сохранение

        # Создаем директорию для сохранения результатов
        os.makedirs(cfg.save_dir, exist_ok=True)
        
        # Автоматическая загрузка checkpoint при старте, если он существует
        if cfg.auto_load_checkpoint:
            loaded = self.load_checkpoint(silent=False)
            if not loaded:
                print("Checkpoint не найден. Начинаем обучение с нуля.")

    def preprocess_obs(self, obs: np.ndarray) -> torch.Tensor:
        """
        Предобработка изображения для подачи в нейронную сеть.
        
        Выполняет следующие преобразования:
        - Нормализация пикселей в диапазон [0, 1]
        - Перестановка каналов из HWC в CHW формат
        - Опциональное преобразование в оттенки серого
        - Изменение размера до нужного разрешения
        
        Args:
            obs: Изображение из среды [height, width, channels]
        
        Returns:
            Обработанный тензор [channels, height, width]
        """
        # Преобразуем в тензор и нормализуем в [0, 1]
        x = torch.from_numpy(obs).float() / 255.0
        # Переставляем каналы: HWC -> CHW
        x = x.permute(2, 0, 1)

        # Опционально преобразуем в оттенки серого
        if self.cfg.use_grayscale:
            x = x.mean(dim=0, keepdim=True)

        # Добавляем размерность батча для интерполяции
        x = x.unsqueeze(0)
        # Изменяем размер до нужного разрешения
        x = F.interpolate(
            x,
            size=(self.cfg.out_h, self.cfg.out_w),
            mode="bilinear",
            align_corners=False
        )
        # Убираем размерность батча
        x = x.squeeze(0).contiguous()

        return x.cpu()

    def pick_action(self, obs: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """
        Выбирает действие для текущего состояния.
        
        Для обучения добавляет OU-шум к детерминистическому действию политики
        для исследования пространства действий. Для оценки (add_noise=False)
        возвращает чистое действие политики.
        
        Args:
            obs: Текущее наблюдение из среды
            add_noise: Добавлять ли шум для исследования
        
        Returns:
            Действие [action_dim]
        """
        # Предобрабатываем изображение
        x = self.preprocess_obs(obs).unsqueeze(0).to(self.cfg.device)

        # Получаем детерминистическое действие от политики
        with torch.no_grad():
            action = self.actor(x)
            action = action.squeeze(0).cpu().numpy()

        # Добавляем шум для исследования (только при обучении)
        if add_noise:
            noise = self.ou_noise()
            action = action + noise
            # Ограничиваем действие в допустимый диапазон
            action = np.clip(
                action,
                self.env.action_space.low,
                self.env.action_space.high
            )

        return action

    def optimize(self, states, actions, rewards, next_states, dones):
        """
        Выполняет один шаг обучения для Critic и Actor сетей.
        
        Это ключевая функция алгоритма DDPG. Она реализует обучение на основе
        уравнения Беллмана:
        
        Q*(s, a) = r + gamma * max_a' Q*(s', a')
        
        Для непрерывных действий максимизация заменяется на использование
        детерминистической политики:
        
        Q*(s, a) ≈ r + gamma * Q*(s', μ(s'))
        
        где μ(s') - действие от целевой политики для следующего состояния.
        
        Args:
            states: Список текущих состояний
            actions: Список выполненных действий
            rewards: Список полученных наград
            next_states: Список следующих состояний
            dones: Список флагов завершения эпизода
        
        Returns:
            dict: Словарь с метриками обучения (losses, Q-values, gradients)
        """
        # Преобразуем данные в тензоры
        states_t = torch.stack(states).to(self.cfg.device)
        actions_t = torch.from_numpy(np.array(actions)).float().to(self.cfg.device)
        rewards_t = torch.from_numpy(np.array(rewards)).float().to(self.cfg.device).unsqueeze(1)
        next_states_t = torch.stack(next_states).to(self.cfg.device)
        dones_t = torch.from_numpy(np.array(dones)).float().to(self.cfg.device).unsqueeze(1)

        # ========== ОБУЧЕНИЕ CRITIC (Q-сеть) ==========
        # Минимизируем ошибку предсказания Q-значений
        
        self.critic_opt.zero_grad()
        
        # Текущие Q-значения: Q(s, a)
        q_current = self.critic(states_t, actions_t)
        
        # Вычисляем целевые Q-значения используя целевые сети
        # Действие для следующего состояния: μ_target(s')
        with torch.no_grad():
            next_actions = self.actor_target(next_states_t)
            # Q-значение следующего состояния: Q_target(s', μ_target(s'))
            q_next = self.critic_target(next_states_t, next_actions)
            # Целевое Q-значение: r + gamma * (1 - done) * Q_target(s', μ_target(s'))
            # (1 - done) обнуляет будущие награды, если эпизод завершен
            q_target = rewards_t + self.cfg.gamma * (1.0 - dones_t) * q_next
        
        # Функция потерь: MSE между текущими и целевыми Q-значениями
        critic_loss = F.mse_loss(q_current, q_target)
        critic_loss.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.grad_clip_norm)
        self.critic_opt.step()

        # ========== ОБУЧЕНИЕ ACTOR (политика) ==========
        # Максимизируем Q(s, μ(s)) - качество действий от текущей политики
        
        self.actor_opt.zero_grad()
        
        # Получаем действия от текущей политики: μ(s)
        current_actions = self.actor(states_t)
        
        # Вычисляем Q-значения для действий от текущей политики
        # Градиенты должны идти через Critic к Actor
        q_values = self.critic(states_t, current_actions)
        
        # Максимизируем Q-значения (минимизируем отрицательные)
        actor_loss = -q_values.mean()
        actor_loss.backward()
        
        # Обрезаем градиенты только для Actor (не для Critic)
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.grad_clip_norm)
        self.actor_opt.step()
        
        # Очищаем градиенты Critic, которые могли накопиться при backward()
        # (мы не хотим обновлять Critic при обучении Actor)
        self.critic_opt.zero_grad()
        
        # Возвращаем метрики для логирования
        with torch.no_grad():
            return {
                "critic_loss": critic_loss.item(),
                "actor_loss": actor_loss.item(),
                "q_current_mean": q_current.mean().item(),
                "q_target_mean": q_target.mean().item(),
                "q_values_mean": q_values.mean().item(),
                "critic_grad_norm": critic_grad_norm.item(),
                "actor_grad_norm": actor_grad_norm.item(),
            }

    def update_target_networks(self):
        """
        Выполняет мягкое обновление целевых сетей (soft update).
        
        Вместо резкого копирования весов основных сетей в целевые, мы делаем
        взвешенную комбинацию:
        
        target = tau * origin + (1 - tau) * target
        
        где tau << 1 (обычно 0.001-0.01). Это обеспечивает медленное обновление
        целевых сетей, что стабилизирует обучение, предотвращая резкие изменения
        целевых значений Q-функции.
        
        Без этого механизма обучение может быть нестабильным, так как целевые
        значения будут постоянно меняться вместе с основными сетями.
        """
        tau = self.cfg.tau

        # Обновляем целевую сеть Actor
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data = tau * param.data + (1.0 - tau) * target_param.data

        # Обновляем целевую сеть Critic
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data = tau * param.data + (1.0 - tau) * target_param.data

    def start_learning(self, save_weights: bool = False):
        """
        Основной цикл обучения DDPG.
        
        Алгоритм обучения:
        1. Агент взаимодействует со средой, собирая опыт
        2. Опыт сохраняется в replay buffer
        3. Периодически агент обучается на случайных батчах из буфера
        4. Целевые сети обновляются мягко после каждого шага обучения
        
        Обучение происходит по шагам, а не по эпизодам, что позволяет
        более эффективно использовать данные и делать более частые обновления.
        """
        pbar = trange(self.cfg.total_episodes, desc="Training")

        step_count = 0  # Счетчик шагов для определения частоты обучения
        
        # Для накопления метрик за эпизод
        episode_metrics = {
            "critic_loss": [],
            "actor_loss": [],
            "q_current_mean": [],
            "q_target_mean": [],
            "q_values_mean": [],
            "critic_grad_norm": [],
            "actor_grad_norm": [],
        }

        for ep in pbar:
            # Запускаем новый эпизод
            obs, _ = self.env.reset(seed=self.cfg.seed + ep)
            done = False
            ep_reward = 0.0
            ep_length = 0

            # Сбрасываем OU-шум в начале каждого эпизода
            self.ou_noise.reset()
            
            # Сбрасываем метрики эпизода
            for key in episode_metrics:
                episode_metrics[key].clear()

            # Выполняем шаги до завершения эпизода
            while not done:
                # Выбираем действие с добавлением шума для исследования
                action = self.pick_action(obs, add_noise=True)

                # Выполняем действие в среде
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                # Предобрабатываем состояния для сохранения в буфер
                state = self.preprocess_obs(obs)
                next_state = self.preprocess_obs(next_obs)

                # Сохраняем опыт в replay buffer
                self.buffer.add(state, action, reward, next_state, done)

                # Обучаемся, если в буфере достаточно данных
                if len(self.buffer) >= self.cfg.batch_size:
                    # Обучаемся с заданной частотой
                    if step_count % self.cfg.train_freq == 0:
                        # Выполняем несколько шагов обучения
                        for _ in range(self.cfg.train_steps):
                            # Выбираем батч из буфера (адаптивная смешанная выборка при заполнении буфера на 70%+)
                            avg_reward = float(np.mean(self.recent_rewards)) if len(self.recent_rewards) > 0 else 0.0
                            states, actions, rewards, next_states, dones = self.buffer.sample(
                                batch_size=self.cfg.batch_size,
                                avg_reward=avg_reward,
                                stop_reward=self.cfg.stop_reward,
                                buffer_fill_threshold=self.cfg.mixed_sampling_buffer_fill,
                                good_ratio_start=self.cfg.mixed_sampling_good_ratio_start,
                                good_ratio_end=self.cfg.mixed_sampling_good_ratio_end,
                                reward_offset=self.cfg.mixed_sampling_reward_offset
                            )
                            # Выполняем один шаг обучения и получаем метрики
                            metrics = self.optimize(states, actions, rewards, next_states, dones)
                            # Сохраняем метрики
                            for key in metrics:
                                episode_metrics[key].append(metrics[key])
                            # Мягко обновляем целевые сети
                            self.update_target_networks()

                # Переходим к следующему состоянию
                obs = next_obs
                ep_reward += reward
                ep_length += 1
                step_count += 1

            # Логируем результаты эпизода
            self.episode_rewards.append(ep_reward)
            self.recent_rewards.append(ep_reward)

            avg_r = float(np.mean(self.recent_rewards))
            
            # Проверяем на улучшение и сохраняем при необходимости
            if self.cfg.auto_save_best and len(self.recent_rewards) >= min(self.cfg.log_window, 10):
                # Обновляем лучшую награду, если текущая средняя лучше
                if avg_r > self.best_reward:
                    improvement = avg_r - self.best_reward
                    
                    # Для первого сохранения (best_reward == -inf) сохраняем сразу
                    # Для последующих сохраняем только при достаточном улучшении
                    should_save = (self.best_reward == float('-inf')) or (improvement >= self.cfg.improvement_threshold)
                    
                    if should_save:
                        old_best = self.best_reward
                        self.best_reward = avg_r
                        self.best_episode = ep
                        self.last_saved_reward = avg_r
                        
                        # Сохраняем лучший checkpoint
                        self.save_models(suffix="_best")
                        
                        # Если нужно сохранять промежуточные, также сохраняем обычный
                        if not self.cfg.save_best_only:
                            self.save_models(suffix="")
                        
                        if old_best != float('-inf'):
                            print(f"\n[Ep {ep}] Улучшение: {old_best:.1f} → {avg_r:.1f} (+{improvement:.1f})")
                    else:
                        # Обновляем best_reward даже без сохранения, если улучшение меньше порога
                        self.best_reward = avg_r
                        self.best_episode = ep
            
            # Вычисляем средние метрики за эпизод
            if episode_metrics["critic_loss"]:
                avg_critic_loss = np.mean(episode_metrics["critic_loss"])
                avg_actor_loss = np.mean(episode_metrics["actor_loss"])
                avg_q_current = np.mean(episode_metrics["q_current_mean"])
                avg_q_target = np.mean(episode_metrics["q_target_mean"])
                avg_q_values = np.mean(episode_metrics["q_values_mean"])
                avg_critic_grad = np.mean(episode_metrics["critic_grad_norm"])
                avg_actor_grad = np.mean(episode_metrics["actor_grad_norm"])
            else:
                avg_critic_loss = 0.0
                avg_actor_loss = 0.0
                avg_q_current = 0.0
                avg_q_target = 0.0
                avg_q_values = 0.0
                avg_critic_grad = 0.0
                avg_actor_grad = 0.0

            # Определяем статус сохранения для отображения
            save_status = ""
            if self.cfg.auto_save_best:
                if self.best_reward != float('-inf'):
                    if avg_r >= self.best_reward:
                        # Проверяем, была ли это награда сохранена
                        if abs(avg_r - self.last_saved_reward) < 0.1:  # Учитываем погрешность округления
                            save_status = "★"  # Сохранено
                        elif avg_r >= self.last_saved_reward + self.cfg.improvement_threshold:
                            save_status = "★+"  # Готово к сохранению
                        else:
                            save_status = "*"  # Лучшее, но еще не сохранено
                    else:
                        save_status = f"best:{self.best_reward:.0f}"
            
            pbar.set_postfix({
                "avgR": f"{avg_r:.1f}",
                "lastR": f"{ep_reward:.1f}",
                "len": f"{ep_length}",
                "buffer": f"{len(self.buffer)}",
                "cL": f"{avg_critic_loss:.3f}",
                "aL": f"{avg_actor_loss:.3f}",
                "qC": f"{avg_q_current:.1f}",
                "qT": f"{avg_q_target:.1f}",
                "qV": f"{avg_q_values:.1f}",
                "gC": f"{avg_critic_grad:.2f}",
                "gA": f"{avg_actor_grad:.2f}",
                "save": save_status,
            })

            # Останавливаем обучение, если достигли целевой награды
            if len(self.recent_rewards) == self.cfg.log_window and avg_r >= self.cfg.stop_reward:
                print(f"\nОбучение остановлено: средняя награда {avg_r:.2f}")
                break

        # Сохраняем финальные модели и график
        if save_weights:
            # Сохраняем финальные веса (если не были сохранены как best)
            if not self.cfg.save_best_only or avg_r < self.best_reward:
                self.save_models(suffix="")
            print(f"\nОбучение завершено. Лучшая награда: {self.best_reward:.1f} (эпизод {self.best_episode})")

        self.save_plot()

    def save_models(self, suffix: str = ""):
        """
        Сохраняет веса обученных сетей и полный checkpoint.
        
        Args:
            suffix: Суффикс для имени файла (например, "_best" для лучших весов)
        """
        actor_path = os.path.join(self.cfg.save_dir, self.cfg.model_name + suffix + "_actor.pth")
        critic_path = os.path.join(self.cfg.save_dir, self.cfg.model_name + suffix + "_critic.pth")
        
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        
        # Также сохраняем целевые сети, оптимизаторы и метаданные в полный checkpoint
        checkpoint_path = os.path.join(self.cfg.save_dir, self.cfg.model_name + suffix + "_checkpoint.pth")
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_opt.state_dict(),
            'critic_optimizer_state_dict': self.critic_opt.state_dict(),
            'best_reward': self.best_reward,
            'best_episode': self.best_episode,
            'episode_rewards': self.episode_rewards.copy(),
            'last_saved_reward': self.last_saved_reward,
        }
        torch.save(checkpoint, checkpoint_path)
        
        if suffix == "_best":
            print(f"  ✓ Checkpoint сохранен: {checkpoint_path}")
            print(f"    Best reward: {self.best_reward:.1f} (episode {self.best_episode})")

    def load_checkpoint(self, suffix: str = "_best", silent: bool = False):
        """
        Загружает полный checkpoint для продолжения обучения.
        
        Загружает веса основных и целевых сетей, состояния оптимизаторов и метаданные
        (лучшая награда, история эпизодов). Используется для восстановления обучения
        с последней сохраненной точки.
        
        Args:
            suffix: Суффикс для имени файла (по умолчанию "_best" для лучшего checkpoint)
            silent: Если True, не выводит сообщения о загрузке
        
        Returns:
            bool: True если checkpoint загружен успешно, False если файл не найден
        """
        checkpoint_path = os.path.join(self.cfg.save_dir, self.cfg.model_name + suffix + "_checkpoint.pth")
        
        # Пробуем сначала загрузить полный checkpoint
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.cfg.device)
            
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.actor_opt.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic_opt.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            
            # Восстанавливаем метаданные
            if 'best_reward' in checkpoint:
                self.best_reward = checkpoint['best_reward']
            if 'best_episode' in checkpoint:
                self.best_episode = checkpoint['best_episode']
            if 'episode_rewards' in checkpoint:
                self.episode_rewards = checkpoint['episode_rewards'].copy()
            if 'last_saved_reward' in checkpoint:
                self.last_saved_reward = checkpoint['last_saved_reward']
            
            if not silent:
                print(f"✓ Checkpoint загружен из {checkpoint_path}")
                print(f"  Best reward: {self.best_reward:.1f} (episode {self.best_episode})")
                print(f"  Эпизодов в истории: {len(self.episode_rewards)}")
            
            # Переводим в режим обучения (не eval)
            self.actor.train()
            self.critic.train()
            
            return True
        
        # Если полный checkpoint не найден, пробуем загрузить отдельные файлы весов
        actor_path = os.path.join(self.cfg.save_dir, self.cfg.model_name + suffix + "_actor.pth")
        critic_path = os.path.join(self.cfg.save_dir, self.cfg.model_name + suffix + "_critic.pth")
        
        if os.path.exists(actor_path) and os.path.exists(critic_path):
            self.actor.load_state_dict(torch.load(actor_path, map_location=self.cfg.device))
            self.critic.load_state_dict(torch.load(critic_path, map_location=self.cfg.device))
            
            # Обновляем целевые сети
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.critic_target.load_state_dict(self.critic.state_dict())
            
            if not silent:
                print(f"✓ Веса загружены из {actor_path} и {critic_path}")
                print(f"  (Оптимизаторы не восстановлены - начнется новый цикл обучения)")
            
            self.actor.train()
            self.critic.train()
            
            return True
        
        return False

    def load_models(self):
        """
        Загружает сохраненные веса Actor и Critic сетей для тестирования.
        
        Переводит сети в режим eval для тестирования и выводит подтверждение
        успешной загрузки. Использует метод load_checkpoint для загрузки.
        
        Raises:
            FileNotFoundError: Если файлы с весами не найдены
        """
        # Пробуем загрузить лучший checkpoint
        if not self.load_checkpoint(suffix="_best", silent=False):
            # Если не найден, пробуем обычный
            if not self.load_checkpoint(suffix="", silent=False):
                actor_path = os.path.join(self.cfg.save_dir, self.cfg.model_name + "_actor.pth")
                raise FileNotFoundError(f"Checkpoint not found. Expected: {actor_path}")
        
        # Для тестирования переводим в режим eval
        self.actor.eval()
        self.critic.eval()
        
        print(f"Models loaded from {self.cfg.save_dir} (ready for testing)")

    def test(self, num_episodes: int = 5):
        """
        Тестирует обученную модель с визуализацией.
        
        Загружает сохраненные веса, создает среду с визуализацией (render_mode="human")
        и запускает указанное количество эпизодов. Действия выбираются детерминистически
        (без шума) для оценки реальной производительности обученной политики.
        
        Args:
            num_episodes: Количество эпизодов для тестирования
        
        Returns:
            list: Список наград за каждый эпизод
        """
        self.load_models()
        
        test_env = gym.make(
            self.cfg.env_id,
            lap_complete_percent=self.cfg.lap_complete_percent,
            continuous=self.cfg.continuous,
            render_mode="human"
        )
        
        assert isinstance(test_env.action_space, gym.spaces.Box)
        
        episode_rewards = []
        
        for ep in range(num_episodes):
            obs, _ = test_env.reset()
            done = False
            ep_reward = 0.0
            ep_length = 0
            
            while not done:
                # Выбираем действие детерминистически (без шума)
                action = self.pick_action(obs, add_noise=False)
                obs, r, term, trunc, _ = test_env.step(action)
                done = term or trunc
                
                ep_reward += float(r)
                ep_length += 1
            
            episode_rewards.append(ep_reward)
            print(f"Episode {ep + 1}/{num_episodes}: reward = {ep_reward:.1f}, length = {ep_length}")
        
        test_env.close()
        
        avg_reward = np.mean(episode_rewards)
        print(f"\nTest completed: average reward = {avg_reward:.1f} over {num_episodes} episodes")
        
        return episode_rewards

    def save_plot(self):
        """Сохраняет график наград по эпизодам."""
        rewards = np.array(self.episode_rewards)
        window = self.cfg.log_window

        # Вычисляем скользящее среднее
        if len(rewards) >= window:
            moving_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
        else:
            moving_avg = None

        # Строим график
        plt.figure(figsize=(12, 6))
        plt.plot(rewards, alpha=0.5, label="Награда за эпизод")
        if moving_avg is not None:
            plt.plot(
                range(window - 1, len(rewards)),
                moving_avg,
                label=f"Средняя награда ({window} эпизодов)",
                linewidth=2
            )

        plt.xlabel("Номер эпизода")
        plt.ylabel("Награда")
        plt.title("Обучение DDPG на CarRacing")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.cfg.save_dir, self.cfg.plot_name))
        plt.close()


if __name__ == "__main__":
    cfg = Config()
    trainer = CarRacingDDPG(cfg)
    #trainer.start_learning(save_weights=True)
    
    # Раскомментируйте для тестирования:
    trainer.test(num_episodes=5)


