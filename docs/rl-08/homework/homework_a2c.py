"""
A2C для CarRacing-v3 с дискретными действиями.

Идея обучения:
- агент едет один эпизод и запоминает всё, что с ним произошло
- после эпизода он анализирует:
  * какие состояния были хорошими
  * какие действия вели к хорошему результату
- Critic учится оценивать состояние
- Actor учится чаще выбирать удачные действия

Обучение идёт по эпизодам, а не по шагам.
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
    # фиксируем seed, чтобы поведение было воспроизводимым
    seed: int = 42

    # используем дискретную версию CarRacing
    env_id: str = "CarRacing-v3"
    continuous: bool = False
    lap_complete_percent: float = 0.95

    # коэффициент дисконтирования будущих наград
    gamma: float = 0.99

    # скорости обучения для актёра и критика
    lr_actor: float = 1e-4
    lr_critic: float = 3e-4

    # сколько эпизодов максимум обучаемся
    total_episodes: int = 6000

    # окно для усреднения награды
    log_window: int = 100

    # если средняя награда стала очень высокой — можно остановиться
    stop_reward: float = 900.0

    # параметры для устойчивости обучения
    grad_clip_norm: float = 5.0
    entropy_coef: float = 0.1

    # параметры обработки изображения
    use_grayscale: bool = True
    out_h: int = 84
    out_w: int = 84

    # сохранение результатов
    save_dir: str = "docs/rl-08/homework"
    model_name: str = "a2c_carracing"
    plot_name: str = "reward_plot.png"

    # выбираем GPU, если он доступен
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class CNNEncoder(nn.Module):
    """
    Эта сеть получает изображение и превращает его в компактный вектор признаков.

    Она учится выделять:
    - края трассы
    - форму дороги
    - положение машины относительно трассы
    """
    def __init__(self, in_channels: int, out_h: int, out_w: int):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # размер выхода считаем автоматически
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, out_h, out_w)
            y = self.conv(dummy)
            feat_dim = int(np.prod(y.shape[1:]))

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x))


class ActorNet(nn.Module):
    """
    Actor отвечает за выбор действий.

    Он получает признаки состояния и выдаёт logits —
    числа, из которых затем получается распределение действий.
    """
    def __init__(self, in_channels: int, n_actions: int, out_h: int, out_w: int):
        super().__init__()
        self.encoder = CNNEncoder(in_channels, out_h, out_w)
        self.hidden = nn.Linear(512, 256)
        self.head = nn.Linear(256, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        # Проходим через промежуточный слой с активацией ReLU
        hidden = F.relu(self.hidden(features))
        return self.head(hidden)


class CriticNet(nn.Module):
    """
    Critic оценивает состояние.

    Его задача — сказать, насколько хороша текущая ситуация в среднем,
    независимо от конкретного действия.
    """
    def __init__(self, in_channels: int, out_h: int, out_w: int):
        super().__init__()
        self.encoder = CNNEncoder(in_channels, out_h, out_w)
        self.hidden = nn.Linear(512, 256)
        self.head = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        hidden = F.relu(self.hidden(features))
        return self.head(hidden).squeeze(-1)


class CarRacingA2C:
    def __init__(self, cfg: Config):
        self.cfg = cfg

        self.env = gym.make(
            cfg.env_id,
            lap_complete_percent=cfg.lap_complete_percent,
            continuous=cfg.continuous,
        )

        # убеждаемся, что действия действительно дискретные
        assert isinstance(self.env.action_space, gym.spaces.Discrete)

        # фиксируем случайность
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        self.n_actions = self.env.action_space.n
        self.in_channels = 1 if cfg.use_grayscale else 3

        print(f"Device: {cfg.device}, actions: {self.n_actions}")

        # создаём две независимые сети
        self.actor = ActorNet(self.in_channels, self.n_actions, cfg.out_h, cfg.out_w).to(cfg.device)
        self.critic = CriticNet(self.in_channels, cfg.out_h, cfg.out_w).to(cfg.device)

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

        # и два независимых оптимизатора
        self.actor_opt = optim.AdamW(self.actor.parameters(), lr=cfg.lr_actor)
        self.critic_opt = optim.AdamW(self.critic.parameters(), lr=cfg.lr_critic)

        self.episode_rewards = []
        self.episode_lengths = []

        self.recent_rewards = deque(maxlen=cfg.log_window)
        self.recent_lengths = deque(maxlen=cfg.log_window)

        os.makedirs(cfg.save_dir, exist_ok=True)

    def preprocess_obs(self, obs: np.ndarray) -> torch.Tensor:
        """
        Приводим изображение к формату, удобному для сети.

        Делается:
        - перевод в float
        - нормализация
        - перестановка каналов
        - уменьшение размера
        - (опционально) перевод в оттенки серого
        """
        x = torch.from_numpy(obs).float() / 255.0
        x = x.permute(2, 0, 1)

        if self.cfg.use_grayscale:
            x = x.mean(dim=0, keepdim=True)

        x = x.unsqueeze(0)
        x = F.interpolate(
            x,
            size=(self.cfg.out_h, self.cfg.out_w),
            mode="bilinear",
            align_corners=False
        )
        x = x.squeeze(0).contiguous()

        return x.cpu()

    def pick_action(self, obs: np.ndarray, deterministic: bool = False) -> int:
        """
        Для обучения действие выбирается случайно согласно вероятностям.
        Для оценки — берётся самое вероятное.
        """
        x = self.preprocess_obs(obs).unsqueeze(0).to(self.cfg.device)

        with torch.no_grad():
            logits = self.actor(x)

            if deterministic:
                return int(torch.argmax(logits, dim=-1).item())

            probs = torch.softmax(logits, dim=-1)
            return int(torch.multinomial(probs, 1).item())

    def compute_returns(self, rewards: list[float]) -> np.ndarray:
        """
        Считаем суммарную будущую награду для каждого шага эпизода.
        """
        returns = np.zeros(len(rewards), dtype=np.float32)
        G = 0.0
        for i in reversed(range(len(rewards))):
            G = rewards[i] + self.cfg.gamma * G
            returns[i] = G
        return returns

    def start_learning(self, save_weights: bool = False):
        pbar = trange(self.cfg.total_episodes, desc="Training")

        for ep in pbar:
            obs, _ = self.env.reset(seed=self.cfg.seed + ep)

            states, actions, rewards = [], [], []
            done = False
            ep_reward = 0.0

            # агент проезжает эпизод
            while not done:
                states.append(self.preprocess_obs(obs))
                action = self.pick_action(obs)

                obs, r, term, trunc, _ = self.env.step(action)
                done = term or trunc

                actions.append(action)
                rewards.append(float(r))
                ep_reward += float(r)

            # готовим данные для обучения
            states_t = torch.stack(states).to(self.cfg.device)
            actions_t = torch.tensor(actions, dtype=torch.long, device=self.cfg.device)

            if len(rewards) < 10:
                continue

            returns = self.compute_returns(rewards)
            returns_t = torch.tensor(returns, dtype=torch.float32, device=self.cfg.device)

            self.critic_opt.zero_grad()
            values = self.critic(states_t)
            critic_loss = F.mse_loss(values, returns_t)
            critic_loss.backward()
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.grad_clip_norm)
            self.critic_opt.step()

            with torch.no_grad():
                values_detached = self.critic(states_t)

            advantages = returns_t - values_detached
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            self.actor_opt.zero_grad()
            logits = self.actor(states_t)
            log_probs = -F.cross_entropy(logits, actions_t, reduction="none")

            policy_loss = -(log_probs * advantages).mean()

            probs = torch.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()

            actor_loss = policy_loss - self.cfg.entropy_coef * entropy
            actor_loss.backward()
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.grad_clip_norm)
            self.actor_opt.step()

            self.episode_rewards.append(ep_reward)
            self.recent_rewards.append(ep_reward)

            avg_r = float(np.mean(self.recent_rewards))

            adv_mean = advantages.mean().item()
            adv_std = advantages.std().item()
            ret_mean = returns_t.mean().item()
            val_mean = values_detached.mean().item()
            entropy_val = entropy.item()
            ep_len = len(rewards)

            pbar.set_postfix({
                "avgR": f"{avg_r:.1f}",
                "lastR": f"{ep_reward:.1f}",
                "len": f"{ep_len}",
                "pi": f"{policy_loss.item():.3f}",
                "v": f"{critic_loss.item():.3f}",
                "ent": f"{entropy_val:.3f}",
                "advM": f"{adv_mean:.2f}", # среднее отклонение advantages
                "advS": f"{adv_std:.2f}", # стандартное отклонение advantages
                "retM": f"{ret_mean:.1f}",
                "valM": f"{val_mean:.1f}",
                "gA": f"{actor_grad_norm:.2f}",
                "gC": f"{critic_grad_norm:.2f}",
            })

            if len(self.recent_rewards) == self.cfg.log_window and avg_r >= self.cfg.stop_reward:
                print(f"\nОбучение остановлено: средняя награда {avg_r:.2f}")
                break

        if save_weights:
            self.save_models()

        self.save_plot()

    def save_models(self):
        torch.save(self.actor.state_dict(), os.path.join(self.cfg.save_dir, self.cfg.model_name + "_actor.pth"))
        torch.save(self.critic.state_dict(), os.path.join(self.cfg.save_dir, self.cfg.model_name + "_critic.pth"))

    def load_models(self):
        actor_path = os.path.join(self.cfg.save_dir, self.cfg.model_name + "_actor.pth")
        critic_path = os.path.join(self.cfg.save_dir, self.cfg.model_name + "_critic.pth")
        
        if not os.path.exists(actor_path):
            raise FileNotFoundError(f"Actor weights not found: {actor_path}")
        if not os.path.exists(critic_path):
            raise FileNotFoundError(f"Critic weights not found: {critic_path}")
        
        self.actor.load_state_dict(torch.load(actor_path, map_location=self.cfg.device))
        self.critic.load_state_dict(torch.load(critic_path, map_location=self.cfg.device))
        
        self.actor.eval()
        self.critic.eval()
        
        print(f"Models loaded from {self.cfg.save_dir}")

    def save_plot(self):
        rewards = np.array(self.episode_rewards)
        window = self.cfg.log_window

        if len(rewards) >= window:
            moving_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
        else:
            moving_avg = None

        plt.figure(figsize=(12, 6))
        plt.plot(rewards, alpha=0.5, label="reward")
        if moving_avg is not None:
            plt.plot(range(window - 1, len(rewards)), moving_avg, label="moving average")

        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.cfg.save_dir, self.cfg.plot_name))
        plt.close()

    def test(self, num_episodes: int = 5):
        self.load_models()
        
        test_env = gym.make(
            self.cfg.env_id,
            lap_complete_percent=self.cfg.lap_complete_percent,
            continuous=self.cfg.continuous,
            render_mode="human"
        )
        
        assert isinstance(test_env.action_space, gym.spaces.Discrete)
        
        episode_rewards = []
        
        for ep in range(num_episodes):
            obs, _ = test_env.reset()
            done = False
            ep_reward = 0.0
            ep_length = 0
            
            while not done:
                action = self.pick_action(obs, deterministic=True)
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


if __name__ == "__main__":
    cfg = Config()
    trainer = CarRacingA2C(cfg)
    #trainer.start_learning(save_weights=True)
    trainer.test(num_episodes=5)
