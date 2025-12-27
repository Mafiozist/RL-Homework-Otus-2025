import os
import random
from dataclasses import dataclass
from collections import deque

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim


# Конфиг обучения
@dataclass
class Config:
    # Seed нужен для частичной воспроизводимости
    seed: int = 42

    # γ (gamma) - коэффициент дисконтирования
    # формально участвует в целевом значении Беллмана
    # y = r + γ * max_a' Q(s', a')
    # Чем ближе γ к 1, тем сильнее агент "думает о будущем"
    gamma: float = 0.99

    # learning rate - насколько быстро сеть меняет веса
    # В DQN слишком большой lr часто приводит к расходимости (цель нестабильна)
    # слишком маленький - обучение не происходит
    lr: float = 1e-3

    # ε-greedy
    # В начале сеть ничего не знает
    # Если сразу действовать жадно (argmax Q), агент застрянет в случайной плохой политике
    # Поэтому мы принуждаем исследование: с вероятностью ε делаем случайное действие
    epsilon_start: float = 1.0   # в начале почти всегда исследуем
    epsilon_end: float = 0.05    # в конце редко, но всё ещё иногда исследуем
    epsilon_decay_steps: int = 200_000
    # decay_steps - за сколько шагов ε линейно упадёт

    # Replay Buffer
    # DQN обучается на "опыте": (s, a, r, s', done)
    # Если учиться только на свежих переходах подряд - данные коррелированы
    # (состояния соседние -> похожие), градиент становится "шумным"
    # ReplayBuffer решает это путем перемешивания опыта
    buffer_size: int = 200_000     # сколько переходов храним
    batch_size: int = 64           # сколько переходов используем в одном градиентном шаге
    learning_starts: int = 5_000   # пока буфер пустой/маленький - не учимся, копим опыт

    # Шаги обучения
    total_steps: int = 400_000

    # как часто делать градиентный шаг.
    train_freq: int = 1

    # как часто копировать веса online-сети в target-сеть.
    # Target-сеть нужна, чтобы "цель Беллмана" не менялась каждый шаг.
    target_update_freq: int = 1_000

    # ------------------------------------------------------------
    # Стабилизация
    # ------------------------------------------------------------
    # Градиенты в DQN иногда "взрываются" из-за больших ошибок на старте.
    # Gradient clipping ограничивает норму градиента и делает оптимизацию устойчивее.
    grad_clip_norm: float = 10.0

    # Huber loss (SmoothL1) менее чувствителен к выбросам, чем MSE.
    huber_delta: float = 1.0 

    eval_every_episodes: int = 20
    eval_episodes: int = 10

    # ------------------------------------------------------------
    # Артефакты и устройство
    # ------------------------------------------------------------
    save_dir: str = "docs/rl-06/homework"
    model_name: str = "dqn_lunarlander.pt"
    plot_name: str = "reward_plot.png"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# Replay Buffer: память опыта агента (DQN - off-policy метод)
# Это значит, что мы можем учиться по переходам, собранным старой политикой
# даже если текущая политика уже другая
# ReplayBuffer - ключевой механизм, который позволяет это делать эффективно
class ReplayBuffer:
    def __init__(self, obs_dim: int, size: int):
        # size - максимальная ёмкость памяти. Когда память заполнится
        # мы начинаем перезаписывать старые переходы "по кругу"
        self.size = size
        self.ptr = 0       # куда вставлять следующий переход
        self.count = 0     # сколько переходов реально сохранено (до size)

        # Мы храним всё в numpy-массивах фиксированного размера
        # это быстрее и проще, чем list of tuples
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)       # s
        self.next_obs = np.zeros((size, obs_dim), dtype=np.float32)  # s'
        self.acts = np.zeros((size,), dtype=np.int64)               # a (индекс действия)
        self.rews = np.zeros((size,), dtype=np.float32)             # r
        self.dones = np.zeros((size,), dtype=np.float32)            # done (0/1)

    def add(self, o, a, r, no, done):
        # Записываем переход (s, a, r, s', done) в текущую позицию ptr
        self.obs[self.ptr] = o
        self.acts[self.ptr] = a
        self.rews[self.ptr] = r
        self.next_obs[self.ptr] = no
        self.dones[self.ptr] = float(done)

        # Кольцевой буфер - ptr идёт по кругу и начинает перезаписывать старое
        self.ptr = (self.ptr + 1) % self.size
        self.count = min(self.count + 1, self.size)

    def sample(self, batch_size: int):
        # Случайная выборка разрушает корреляцию последовательных переходов.
        # Именно это делает обучение более "IID-подобным", как в обычном supervised learning.
        idx = np.random.randint(0, self.count, size=batch_size)
        return (
            self.obs[idx],
            self.acts[idx],
            self.rews[idx],
            self.next_obs[idx],
            self.dones[idx],
        )

    def __len__(self):
        return self.count


# Q-Network: аппроксимация Q(s, a)
# Сеть принимает состояние s (вектор), и выдаёт Q(s,a) для всех действий a
# То есть выход - это вектор длины n_actions
# Потом:
# - greedy действие = argmax_a Q(s,a)
# - для обучения мы берём Q(s, a_taken) через gather
class QNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()

        # выход без активации, потому что Q-значения могут быть любыми (и +, и -)
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class LunarDQNWrapper(gym.Wrapper):
    def __init__(self, cfg: Config = Config()):
        self.cfg = cfg

        env = gym.make("LunarLander-v3")
        super().__init__(env)
        self.env = env

        # помогает сделать эксперименты сравнимыми
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        # Размерности
        obs_dim = env.observation_space.shape[0]
        n_actions = env.action_space.n

        # Online и Target сети
        # Online сеть Qθ - обучается градиентом
        self.q_net = QNet(obs_dim, n_actions).to(cfg.device)

        # Target сеть Qθ- используется для построения цели y и обновляется редко
        self.target_net = QNet(obs_dim, n_actions).to(cfg.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        # Оптимизатор для обучения online сети
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=cfg.lr)

        # Replay buffer - память опыта
        self.buffer = ReplayBuffer(obs_dim=obs_dim, size=cfg.buffer_size)

        # Логи по эпизодам
        self.episode_rewards = []
        self.episode_lengths = []

        os.makedirs(cfg.save_dir, exist_ok=True)

    # как падает exploration со временем
    def _epsilon(self, step: int) -> float:
        # В начале много random, потом больше жадной политики.
        # Линейный спад - простой и рабочий baseline.
        if step >= self.cfg.epsilon_decay_steps:
            return self.cfg.epsilon_end

        frac = step / float(self.cfg.epsilon_decay_steps)
        return self.cfg.epsilon_start + frac * (self.cfg.epsilon_end - self.cfg.epsilon_start)

    # Выбор действия по ε-greedy
    def _select_action(self, obs: np.ndarray, step: int) -> int:
        eps = self._epsilon(step)

        # Exploration шаг - случайное действие.
        # Это нужно не для разнообразия, а чтобы собрать опыт в разных регионах пространства состояний.
        if np.random.rand() < eps:
            return self.env.action_space.sample()

        # Exploitation шаг: жадное действие по текущей оценке Qθ
        # Важно использовать no_grad, потому что выбор действия не должен строить граф градиентов.
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32, device=self.cfg.device).unsqueeze(0)
            q = self.q_net(x)  # [1, n_actions]
            return int(torch.argmax(q, dim=1).item())

    # Один шаг обучения DQN по минибатчу
    def _train_step(self) -> float:
        cfg = self.cfg

        # Берём случайный минибатч переходов из буфера:
        # (s, a, r, s', done)
        o, a, r, no, d = self.buffer.sample(cfg.batch_size)

        # Переводим в torch тензоры на нужное устройство
        o_t  = torch.as_tensor(o,  dtype=torch.float32, device=cfg.device)               # [B, obs_dim]
        a_t  = torch.as_tensor(a,  dtype=torch.int64,   device=cfg.device).unsqueeze(1)  # [B, 1]
        r_t  = torch.as_tensor(r,  dtype=torch.float32, device=cfg.device).unsqueeze(1)  # [B, 1]
        no_t = torch.as_tensor(no, dtype=torch.float32, device=cfg.device)               # [B, obs_dim]
        d_t  = torch.as_tensor(d,  dtype=torch.float32, device=cfg.device).unsqueeze(1)  # [B, 1]

        # Текущая оценка Qθ(s, a_taken) 
        # q_net(o_t) даёт Qθ(s, a) для всех a: [B, n_actions]
        # gather(1, a_t) выбирает для каждой строки то действие, которое реально было выполнено
        q_sa = self.q_net(o_t).gather(1, a_t)  # [B, 1]

        #  Цель Беллмана y 
        # y = r + γ(1-done) max_a' Qθ-(s', a')
        # Важно:
        # - используем target_net, чтобы цель была более фиксированной
        # - no_grad, чтобы не гонять градиент в target ветку
        with torch.no_grad():
            q_next = self.target_net(no_t).max(dim=1, keepdim=True).values  # [B,1]
            target = r_t + cfg.gamma * (1.0 - d_t) * q_next                 # [B,1]

        #  Функция потерь 
        # MSE часто нестабилен из-за выбросов TD-error на старте.
        # Huber (SmoothL1) делает большие ошибки линейными
        loss = nn.SmoothL1Loss()(q_sa, target)

        #  Градиентный шаг 
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # если TD-error внезапно огромный, градиенты могут разнести веса
        nn.utils.clip_grad_norm_(self.q_net.parameters(), cfg.grad_clip_norm)

        self.optimizer.step()

        return float(loss.item())

    # Тест политики (без exploration (greedy))
    def test_policy(self, episodes: int = 20, render: bool = False):
        # Для честной оценки надо выключить ε-greedy и действовать greedy
        env = self.env
        if render:
            env = gym.make(self.cfg.env_id, render_mode="human")

        rewards = []
        for _ in range(episodes):
            obs, info = env.reset(seed=self.cfg.seed)
            ep_r = 0.0

            while True:
                with torch.no_grad():
                    x = torch.as_tensor(obs, dtype=torch.float32, device=self.cfg.device).unsqueeze(0)
                    a = int(torch.argmax(self.q_net(x), dim=1).item())

                obs, r, terminated, truncated, info = env.step(a)
                ep_r += r

                if terminated or truncated:
                    rewards.append(ep_r)
                    break

        return float(np.mean(rewards)), float(np.std(rewards))

    # Главный цикл обучения
    def start_learning(self, total_steps: int | None = None, saveWeights: bool = False):
        cfg = self.cfg
        if total_steps is None:
            total_steps = cfg.total_steps

        # Начинаем эпизод
        obs, info = self.env.reset(seed=cfg.seed)
        ep_reward = 0.0
        ep_len = 0
        ep_idx = 0

        # Для удобства усредняем loss по последним шагам
        losses = deque(maxlen=1000)

        for step in range(1, total_steps + 1):
            # Выбираем действие по ε-greedy
            action = self._select_action(obs, step)

            # Делаем шаг в среде
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # Сохраняем опыт в replay buffer
            self.buffer.add(obs, action, reward, next_obs, done)

            # Обновляем текущий state
            obs = next_obs
            ep_reward += reward
            ep_len += 1

            # Учимся, только когда есть адекватный объём данных
            # learning_starts защищает от обучения по нулевому опыту
            if len(self.buffer) >= cfg.learning_starts and (step % cfg.train_freq == 0):
                loss = self._train_step()
                losses.append(loss)

            # Периодически обновляем target-сеть
            if step % cfg.target_update_freq == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())

            # Если эпизод завершился - логируем и стартуем новый
            if done:
                self.episode_rewards.append(ep_reward)
                self.episode_lengths.append(ep_len)
                ep_idx += 1

                # Периодический eval
                # - обучаемся всегда с exploration
                # - оцениваем всегда greedy
                if ep_idx % cfg.eval_every_episodes == 0:
                    mean_r, std_r = self.test_policy(cfg.eval_episodes, render=False)
                    avg_loss = float(np.mean(losses)) if len(losses) else float("nan")
                    print(
                        f"ep={ep_idx:4d} step={step:7d} trainR={ep_reward:8.2f} "
                        f"evalR={mean_r:8.2f}±{std_r:6.2f} loss={avg_loss:8.4f} eps={self._epsilon(step):.3f}"
                    )
                else:
                    print(
                        f"ep={ep_idx:4d} step={step:7d} trainR={ep_reward:8.2f} eps={self._epsilon(step):.3f}"
                    )

                # reset
                obs, info = self.env.reset()
                ep_reward = 0.0
                ep_len = 0

        if saveWeights:
            self.save_results()

        self.print_results()

    # reward per episode + moving average
    def print_results(self):
        rewards = np.array(self.episode_rewards, dtype=np.float32)
        if len(rewards) == 0:
            print("No rewards to plot.")
            return

        episodes = np.arange(len(rewards))

        window = 50
        moving_avg = None
        if len(rewards) >= window:
            moving_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")

        plt.figure(figsize=(12, 6))
        plt.plot(episodes, rewards, alpha=0.5, label="Reward per episode")

        if moving_avg is not None:
            plt.plot(np.arange(window - 1, len(rewards)), moving_avg, linewidth=2, label=f"Moving avg ({window})")

        plt.xlabel("Episode")
        plt.ylabel("Sum reward")
        plt.title(
            f"DQN LunarLander\n"
            f"Mean last {min(100, len(rewards))} episodes: {np.mean(rewards[-min(100, len(rewards)):]):.2f}"
        )
        plt.grid(True)
        plt.legend()

        out_path = os.path.join(self.cfg.save_dir, self.cfg.plot_name)
        plt.savefig(out_path)
        plt.close()
        print(f"Saved plot: {out_path}")

    # Сохраняем только веса online сети
    def save_results(self):
        path = os.path.join(self.cfg.save_dir, self.cfg.model_name)
        torch.save(self.q_net.state_dict(), path)
        print(f"Saved model: {path}")

    def load_results(self):
        path = os.path.join(self.cfg.save_dir, self.cfg.model_name)
        if not os.path.exists(path):
            print(f"File not found: {path}")
            return False

        state = torch.load(path, map_location=self.cfg.device)
        self.q_net.load_state_dict(state)

        # После загрузки обязательно синхронизируем target
        # иначе target будет случайным и первые train_step могут стать нестабильными
        self.target_net.load_state_dict(self.q_net.state_dict())
        print(f"Loaded model: {path}")
        return True


if __name__ == "__main__":
    agent = LunarDQNWrapper(Config())

    agent.load_results()
    print(agent.test_policy(20, render=True))

    #agent.start_learning(saveWeights=True)
