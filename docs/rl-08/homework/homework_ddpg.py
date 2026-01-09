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
    lr_critic: float = 1e-3

    # Параметры для replay buffer
    buffer_size: int = 100000  # Максимальный размер буфера опыта
    batch_size: int = 64  # Размер батча для обучения

    # Параметры для target networks
    tau: float = 0.005  # Коэффициент мягкого обновления (очень мал для стабильности)

    # Параметры для exploration noise (Ornstein-Uhlenbeck)
    ou_theta: float = 0.15  # Скорость возврата к среднему
    ou_sigma: float = 0.2  # Волатильность шума (больше = больше исследование)
    ou_dt: float = 0.01  # Временной шаг для OU-процесса

    # Параметры обучения
    total_episodes: int = 200  # Максимальное количество эпизодов
    train_freq: int = 1  # Частота обучения (каждые N шагов)
    train_steps: int = 1  # Количество шагов обучения за раз

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
        # Каждый слой уменьшает пространственное разрешение и увеличивает глубину
        self.conv = nn.Sequential(
            # Первый слой: большая свертка с большим шагом для быстрого уменьшения размера
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            # Второй слой: средняя свертка для дальнейшего сжатия
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            # Третий слой: маленькая свертка для тонкой обработки
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
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
            nn.Linear(feat_dim, 512),  # Сжимаем до 512 признаков
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через сеть.
        
        Args:
            x: Тензор изображений [batch_size, channels, height, width]
        
        Returns:
            Вектор признаков [batch_size, 512]
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
        # Голова сети, которая преобразует признаки в действия
        self.head = nn.Linear(512, action_dim)

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
        # Преобразуем признаки в действия
        actions = self.head(features)
        
        # Применяем активации для ограничения диапазона действий
        # Руль: tanh ограничивает в [-1, 1]
        # Газ и тормоз: sigmoid ограничивает в [0, 1]
        actions[:, 0] = torch.tanh(actions[:, 0])  # руль
        actions[:, 1] = torch.sigmoid(actions[:, 1])  # газ
        actions[:, 2] = torch.sigmoid(actions[:, 2])  # тормоз
        
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
        self.fusion = nn.Linear(512 + action_dim, 256)
        # Голова сети, которая выдает Q-значение
        self.head = nn.Linear(256, 1)

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

    def sample(self, batch_size: int):
        """
        Случайно выбирает батч опыта из буфера.
        
        Args:
            batch_size: Размер батча
        
        Returns:
            Кортеж (states, actions, rewards, next_states, dones)
        """
        # Случайно выбираем индексы из текущего размера буфера
        indices = [random.randint(0, len(self.buffer) - 1) for _ in range(batch_size)]
        
        # Извлекаем данные по выбранным индексам
        states = [self.buffer[i][0] for i in indices]
        actions = [self.buffer[i][1] for i in indices]
        rewards = [self.buffer[i][2] for i in indices]
        next_states = [self.buffer[i][3] for i in indices]
        dones = [self.buffer[i][4] for i in indices]
        
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

        # Создаем директорию для сохранения результатов
        os.makedirs(cfg.save_dir, exist_ok=True)

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
        """
        # Преобразуем данные в тензоры
        states_t = torch.stack(states).to(self.cfg.device)
        actions_t = torch.tensor(actions, dtype=torch.float32).to(self.cfg.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).to(self.cfg.device).unsqueeze(1)
        next_states_t = torch.stack(next_states).to(self.cfg.device)
        dones_t = torch.tensor(dones, dtype=torch.float32).to(self.cfg.device).unsqueeze(1)

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
        self.critic_opt.step()

        # ========== ОБУЧЕНИЕ ACTOR (политика) ==========
        # Максимизируем Q(s, μ(s)) - качество действий от текущей политики
        
        self.actor_opt.zero_grad()
        
        # Получаем действия от текущей политики: μ(s)
        current_actions = self.actor(states_t)
        
        # Временно отключаем градиенты Critic при обучении Actor
        # Это нужно, чтобы градиенты не распространялись обратно в Critic
        for param in self.critic.parameters():
            param.requires_grad = False
        
        # Вычисляем Q-значения для действий от текущей политики
        q_values = self.critic(states_t, current_actions)
        # Максимизируем Q-значения (минимизируем отрицательные)
        actor_loss = -q_values.mean()
        actor_loss.backward()
        self.actor_opt.step()
        
        # Включаем градиенты Critic обратно
        for param in self.critic.parameters():
            param.requires_grad = True

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

        for ep in pbar:
            # Запускаем новый эпизод
            obs, _ = self.env.reset(seed=self.cfg.seed + ep)
            done = False
            ep_reward = 0.0
            ep_length = 0

            # Сбрасываем OU-шум в начале каждого эпизода
            self.ou_noise.reset()

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
                            # Случайно выбираем батч из буфера
                            states, actions, rewards, next_states, dones = self.buffer.sample(
                                self.cfg.batch_size
                            )
                            # Выполняем один шаг обучения
                            self.optimize(states, actions, rewards, next_states, dones)
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

            pbar.set_postfix({
                "avgR": f"{avg_r:.1f}",
                "lastR": f"{ep_reward:.1f}",
                "len": f"{ep_length}",
                "buffer": f"{len(self.buffer)}",
            })

            # Останавливаем обучение, если достигли целевой награды
            if len(self.recent_rewards) == self.cfg.log_window and avg_r >= self.cfg.stop_reward:
                print(f"\nОбучение остановлено: средняя награда {avg_r:.2f}")
                break

        # Сохраняем модели и график
        if save_weights:
            self.save_models()

        self.save_plot()

    def save_models(self):
        """Сохраняет веса обученных сетей."""
        torch.save(
            self.actor.state_dict(),
            os.path.join(self.cfg.save_dir, self.cfg.model_name + "_actor.pth")
        )
        torch.save(
            self.critic.state_dict(),
            os.path.join(self.cfg.save_dir, self.cfg.model_name + "_critic.pth")
        )

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
    trainer.start_learning(save_weights=True)


