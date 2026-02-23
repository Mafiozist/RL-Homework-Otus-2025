"""
Dyna-Q для FrozenLake 8x8: модель стохастической среды и сравнение с Q-learning.

Зачем нужна модель среды
------------------------
В обычном Q-learning агент обновляет оценку Q(s, a) только когда реально попал
в переход (s, a) -> (s', r). За один эпизод каждый переход встречается редко,
поэтому обучение идёт медленно. Идея Dyna: выучить по реальным переходам модель
«что бывает, если из (s, a) пойти» и потом многократно «проигрывать» в голове
случайные переходы из модели, делая по ним такие же TD-обновления Q. Один
реальный опыт даёт много обновлений — обучение ускоряется.

Чем модель для стохастической среды отличается от детерминированной
--------------------------------------------------------------------
В детерминированной среде из (s, a) всегда один и тот же исход (s', r). Достаточно
хранить одну запись на пару (s, a). В FrozenLake с is_slippery=True лёд скользкий:
из одной клетки при одном и том же действии можно с вероятностью 1/3 уйти в нужную
сторону или в одну из двух перпендикулярных. Поэтому одна пара (s, a) может
приводить к разным (s', r) в разных шагах. Модель должна хранить не один исход,
а распределение исходов (например, список всех наблюдавшихся исходов) и при
планировании сэмплировать из него — иначе планирование будет искажать реальность.

Что делает код
-------------
Реализованы: класс модели детерминированной среды (как база), класс модели
стохастической среды для FrozenLake 8x8, табличный Q-learning без модели,
Dyna-Q с планированием по стохастической модели, сравнение обоих агентов по
доле выигранных эпизодов и построение графика.
"""
from __future__ import annotations

import os
import pickle
import random
from collections import defaultdict
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from tqdm.auto import trange


# FrozenLake: состояние — номер клетки 0..n*n-1; действия 0=влево, 1=вниз, 2=вправо, 3=вверх.
# Награда 1 при достижении цели (G), 0 при падении в прорубь (H) и на каждом шаге по льду (F).
# При is_slippery=True с вероятностью 1/3 выполняется выбранное действие, с 1/3 — каждое из двух перпендикулярных.


def _obs_to_state(obs) -> int:
    """Приведение наблюдения Gymnasium к целому состоянию (на случай 0-d array или обёртки)."""
    return int(np.asarray(obs).flat[0])


@dataclass
class Config:
    """
    Параметры эксперимента и агентов.

    Все гиперпараметры собраны здесь, чтобы было легко менять размер карты,
    силу планирования и длину обучения без правок кода агентов.
    """

    # Воспроизводимость и среда
    seed: int = 42
    map_name: str = "8x8"           # размер карты FrozenLake (4x4 или 8x8)
    is_slippery: bool = True         # True — скользкий лёд (стохастика), False — детерминированные переходы

    # Параметры Q-learning (общие для Q и Dyna-Q)
    gamma: float = 0.99              # коэффициент дисконтирования: насколько важна будущая награда
    alpha: float = 0.2                # шаг обучения при TD-обновлении: Q += alpha * (target - Q)
    epsilon_start: float = 1.0        # начальная вероятность случайного действия (исследование)
    epsilon_end: float = 0.05        # к какому epsilon сходим к концу затухания
    epsilon_decay_episodes: int = 6000  # за сколько эпизодов epsilon линейно снижается (для 8x8 нужно больше фазы эксплуатации)

    # Специфика Dyna-Q: сколько «воображаемых» TD-обновлений делать после каждого реального шага
    n_planning_steps: int = 50        # больше шагов планирования — быстрее распространение Q по модели
    total_episodes: int = 12000      # для 8x8 скользкого льда нужно много эпизодов, иначе агент редко добирается до цели
    max_steps_per_episode: int = 200  # лимит шагов в эпизоде (для FrozenLake 8x8 по умолчанию 200)

    # Логирование и сохранение
    log_window: int = 100             # окно для скользящего среднего и вывода доли выигрышей
    save_dir: str = "docs/rl-12/homework"
    plot_name: str = "dyna_vs_q_reward.png"
    rewards_file: str = "rewards.npz"  # файл с массивами наград (rewards_q, rewards_dyna)
    q_table_file: str = "q_table.npy"  # Q-таблица для Q-learning агента
    dyna_q_file: str = "dyna_q_table.npy"  # Q-таблица для Dyna-Q
    dyna_model_file: str = "dyna_model.pkl"  # модель переходов для Dyna-Q (для восстановления/дообучения)


class DeterministicEnvModel:
    """
    Модель среды с одним исходом на пару (состояние, действие).

    Подходит для детерминированных переходов: при каждом вызове add(s, a, ...)
    запись для (s, a) перезаписывается. Используется как основа для интерфейса
    модели (add / sample / state_actions); для скользкого льда нужна
    StochasticEnvModel, которая хранит несколько исходов на (s, a).
    """

    def __init__(self, n_states: int, n_actions: int):
        self.n_states = n_states
        self.n_actions = n_actions
        # Одна запись на (s, a): следующий state, награда, признак конца эпизода
        self._transitions: dict[tuple[int, int], tuple[int, float, bool]] = {}

    def add(self, s: int, a: int, s_next: int, r: float, terminated: bool) -> None:
        """Записать переход (s, a) -> (s_next, r, terminated). Детерминированно: новый исход перезаписывает старый."""
        self._transitions[(int(s), int(a))] = (int(s_next), float(r), bool(terminated))

    def sample(self, s: int, a: int) -> tuple[int, float, bool] | None:
        """Вернуть единственный запомненный исход для (s, a) или None, если пару ещё не видели."""
        key = (int(s), int(a))
        return self._transitions.get(key)

    def state_actions(self) -> list[tuple[int, int]]:
        """Список пар (s, a), по которым есть переход — из него Dyna-Q выбирает случайную пару для планирования."""
        return list(self._transitions.keys())


class StochasticEnvModel:
    """
    Модель стохастической среды FrozenLake 8x8.

    Для каждой пары (s, a) хранит список всех наблюдавшихся исходов (s_next, r, terminated).
    При вызове sample(s, a) возвращает один исход, выбранный равномерно из этого списка —
    то есть с вероятностью, пропорциональной частоте наблюдений. Так при планировании
    отражается скользкость: из одной клетки при одном действии в реальности могли
    получаться разные переходы, и модель это воспроизводит.
    """

    def __init__(self, n_states: int, n_actions: int):
        self.n_states = n_states
        self.n_actions = n_actions
        # Ключ (s, a) -> список всех исходов (s_next, r, terminated); при add просто append
        self._outcomes: dict[tuple[int, int], list[tuple[int, float, bool]]] = defaultdict(list)

    def add(self, s: int, a: int, s_next: int, r: float, terminated: bool) -> None:
        """Добавить ещё одно наблюдение перехода: (s, a) однажды привело к (s_next, r, terminated)."""
        key = (int(s), int(a))
        self._outcomes[key].append((int(s_next), float(r), bool(terminated)))

    def sample(self, s: int, a: int) -> tuple[int, float, bool] | None:
        """
        Сэмплировать один исход для (s, a) из накопленного списка (равновероятно по наблюдениям).
        Возвращает None, если эту пару ещё ни разу не наблюдали.
        """
        key = (int(s), int(a))
        outcomes = self._outcomes.get(key)
        if not outcomes:
            return None
        return random.choice(outcomes)

    def state_actions(self) -> list[tuple[int, int]]:
        """Список пар (s, a), по которым есть хотя бы одно наблюдение — откуда Dyna-Q выбирает пару для планирования."""
        return list(self._outcomes.keys())


def _td_update(
    Q: np.ndarray,
    s: int,
    a: int,
    r: float,
    s_next: int,
    terminated: bool,
    gamma: float,
    alpha: float,
) -> None:
    """
    Одно TD-обновление Q-learning.

    Целевое значение (Bellman): если шаг терминальный, то target = r (после цели или ямы
    нет следующего состояния). Иначе target = r + gamma * max_{a'} Q(s_next, a') —
    награда за шаг плюс дисконтированная оценка лучшего продолжения из s_next.
    Обновление: Q(s, a) сдвигается к target с шагом alpha.
    """
    s, a, s_next = int(s), int(a), int(s_next)
    target = r
    if not terminated:
        target += gamma * np.max(Q[s_next])
    Q[s, a] += alpha * (target - Q[s, a])


class QLearningAgent:
    """
    Табличный Q-learning без модели среды.

    На каждом шаге: выбираем действие ε-greedy, получаем (s', r, terminated) из среды,
    делаем одно TD-обновление Q(s, a). Никакого планирования — только реальный опыт.
    """

    def __init__(self, n_states: int, n_actions: int, cfg: Config):
        self.n_states = n_states
        self.n_actions = n_actions
        self.cfg = cfg
        # Q[s, a] — оценка ожидаемой суммарной (дисконтированной) награды, если из s выбрать a и дальше жадно
        self.Q = np.zeros((n_states, n_actions), dtype=np.float64)

    def _epsilon(self, episode: int) -> float:
        """Вероятность случайного действия: линейно падает от epsilon_start до epsilon_end за epsilon_decay_episodes эпизодов."""
        if episode >= self.cfg.epsilon_decay_episodes:
            return self.cfg.epsilon_end
        t = episode / max(1, self.cfg.epsilon_decay_episodes)
        return self.cfg.epsilon_start + t * (self.cfg.epsilon_end - self.cfg.epsilon_start)

    def select_action(self, s: int, episode: int, deterministic: bool = False) -> int:
        """ε-greedy: с вероятностью epsilon — случайное действие (исследование), иначе argmax_a Q(s, a). При равенстве Q — случайный выбор среди лучших."""
        s = int(s)
        if deterministic or random.random() >= self._epsilon(episode):
            q = self.Q[s]
            max_q = np.max(q)
            best_actions = np.where(q == max_q)[0]
            return int(random.choice(best_actions))
        return random.randint(0, self.n_actions - 1)

    def update(self, s: int, a: int, r: float, s_next: int, terminated: bool) -> None:
        """Одно TD-обновление по реальному переходу: подтягиваем Q(s, a) к r + gamma * max Q(s', ·) (или к r, если terminated)."""
        _td_update(
            self.Q, s, a, r, s_next, terminated,
            self.cfg.gamma, self.cfg.alpha,
        )


class DynaQAgent(QLearningAgent):
    """
    Dyna-Q: тот же Q-learning, но после каждого реального шага добавляется планирование по модели.

    Порядок после перехода (s, a, r, s_next, terminated):
    1. Добавить этот переход в модель (чтобы потом «проигрывать» его в воображении).
    2. Сделать одно TD-обновление по реальному переходу (как в обычном Q-learning).
    3. n_planning_steps раз: взять случайную пару (s_plan, a_plan) из тех, что уже есть в модели,
       сэмплировать из модели исход (s'_plan, r_plan, term_plan) и сделать по нему такое же TD-обновление.
    Так один реальный шаг даёт 1 + n_planning_steps обновлений Q — обучение ускоряется.
    """

    def __init__(self, n_states: int, n_actions: int, cfg: Config):
        super().__init__(n_states, n_actions, cfg)
        self.model = StochasticEnvModel(n_states, n_actions)

    def update_with_planning(
        self,
        s: int,
        a: int,
        r: float,
        s_next: int,
        terminated: bool,
        episode: int,
    ) -> None:
        """
        Обработать реальный переход: записать в модель, обновить Q по нему, затем
        выполнить n_planning_steps воображаемых обновлений по сэмплам из модели.
        """
        # Реальный опыт попадает в модель и сразу используется для одного TD-обновления
        self.model.add(s, a, s_next, r, terminated)
        _td_update(
            self.Q, s, a, r, s_next, terminated,
            self.cfg.gamma, self.cfg.alpha,
        )

        # Планирование: случайные (s, a) из модели, исход сэмплируем (стохастика учтена в sample)
        state_actions = self.model.state_actions()
        if not state_actions:
            return
        n_planning = self.cfg.n_planning_steps
        for _ in range(n_planning):
            s_plan, a_plan = random.choice(state_actions)
            outcome = self.model.sample(s_plan, a_plan)
            if outcome is None:
                continue
            s_next_plan, r_plan, term_plan = outcome
            _td_update(
                self.Q, s_plan, a_plan, r_plan, s_next_plan, term_plan,
                self.cfg.gamma, self.cfg.alpha,
            )


def run_episodes(
    env: gym.Env,
    agent: QLearningAgent | DynaQAgent,
    n_episodes: int,
    cfg: Config,
    is_dyna: bool,
    pbar: trange | None = None,
) -> list[float]:
    """
    Запуск n_episodes эпизодов с агентом. Возвращает список наград по эпизодам (0 или 1 для FrozenLake).

    Для DynaQAgent после каждого шага вызывается update_with_planning (модель + планирование),
    для QLearningAgent — только update (одно TD-обновление по реальному переходу).
    """
    rewards_list: list[float] = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=cfg.seed + ep)
        s = _obs_to_state(obs)
        ep_reward = 0.0
        for _ in range(cfg.max_steps_per_episode):
            a = agent.select_action(s, ep)
            next_obs, r, term, trunc, _ = env.step(a)
            s_next = _obs_to_state(next_obs)
            done = term or trunc
            ep_reward += float(r)

            # Обновление Q: у Dyna — с записью в модель и планированием, у Q — только TD по реальному переходу
            if is_dyna and isinstance(agent, DynaQAgent):
                agent.update_with_planning(s, a, r, s_next, term, ep)
            else:
                agent.update(s, a, r, s_next, term)

            if done:
                break
            s = s_next

        rewards_list.append(ep_reward)
        if pbar is not None:
            pbar.update(1)
            win_rate = np.mean(rewards_list[-cfg.log_window:]) if len(rewards_list) >= cfg.log_window else np.mean(rewards_list)
            pbar.set_postfix({"lastR": f"{ep_reward:.2f}", "win%": f"{win_rate * 100:.1f}%"})
    return rewards_list


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Скользящее среднее по окну: сглаживает сырые награды для наглядного графика (длина результата len(x) - window + 1)."""
    if len(x) < window:
        return np.full_like(x, np.nan)
    return np.convolve(x, np.ones(window) / window, mode="valid")


def save_rewards(
    rewards_q: list[float] | np.ndarray,
    rewards_dyna: list[float] | np.ndarray,
    cfg: Config,
) -> str:
    """
    Сохраняет массивы наград по эпизодам в save_dir (файл rewards_file).
    Возвращает путь к сохранённому файлу. Удобно для последующей загрузки и построения графика.
    """
    os.makedirs(cfg.save_dir, exist_ok=True)
    path = os.path.join(cfg.save_dir, cfg.rewards_file)
    np.savez_compressed(path, rewards_q=np.asarray(rewards_q), rewards_dyna=np.asarray(rewards_dyna))
    return path


def load_rewards(cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Загружает сохранённые награды из save_dir. Возвращает (rewards_q, rewards_dyna)."""
    path = os.path.join(cfg.save_dir, cfg.rewards_file)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Rewards file not found: {path}")
    data = np.load(path)
    return data["rewards_q"], data["rewards_dyna"]


def save_plot(
    rewards_q: list[float] | np.ndarray,
    rewards_dyna: list[float] | np.ndarray,
    cfg: Config,
    save_path: str | None = None,
) -> str:
    """
    Строит график сравнения Q-learning и Dyna-Q по массивам наград и сохраняет в save_dir.
    Если save_path не указан, используется cfg.plot_name. Возвращает путь к сохранённому изображению.
    """
    os.makedirs(cfg.save_dir, exist_ok=True)
    path = save_path or os.path.join(cfg.save_dir, cfg.plot_name)
    if not os.path.isabs(path) and not path.startswith(cfg.save_dir):
        path = os.path.join(cfg.save_dir, path)
    window = cfg.log_window
    rewards_q = np.asarray(rewards_q)
    rewards_dyna = np.asarray(rewards_dyna)
    fig, ax = plt.subplots(figsize=(10, 5))
    episodes = np.arange(len(rewards_q))
    ax.plot(episodes, rewards_q, alpha=0.4, label="Q-learning (raw)")
    ax.plot(episodes, rewards_dyna, alpha=0.4, label="Dyna-Q (raw)")
    ma_q = moving_average(rewards_q, window)
    ma_dyna = moving_average(rewards_dyna, window)
    ax.plot(range(window - 1, len(rewards_q)), ma_q, label=f"Q-learning (MA {window})")
    ax.plot(range(window - 1, len(rewards_dyna)), ma_dyna, label=f"Dyna-Q (MA {window})")
    ax.set_xlabel("Эпизод")
    ax.set_ylabel("Награда эпизода")
    ax.set_title("FrozenLake 8x8 (скользкий): Q-learning vs Dyna-Q")
    ax.legend()
    ax.grid(True)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def save_agent_q(agent: QLearningAgent, cfg: Config) -> str:
    """Сохраняет Q-таблицу Q-learning агента. Возвращает путь к файлу."""
    os.makedirs(cfg.save_dir, exist_ok=True)
    path = os.path.join(cfg.save_dir, cfg.q_table_file)
    np.save(path, agent.Q)
    return path


def load_agent_q(cfg: Config, n_states: int | None = None, n_actions: int | None = None) -> QLearningAgent:
    """
    Загружает Q-таблицу и создаёт Q-learning агента. n_states и n_actions берутся из размера массива, если не заданы.
    """
    path = os.path.join(cfg.save_dir, cfg.q_table_file)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Q-table file not found: {path}")
    Q = np.load(path)
    if n_states is None:
        n_states = Q.shape[0]
    if n_actions is None:
        n_actions = Q.shape[1]
    agent = QLearningAgent(n_states, n_actions, cfg)
    agent.Q = Q.astype(np.float64)
    return agent


def save_agent_dyna(agent: DynaQAgent, cfg: Config) -> tuple[str, str]:
    """
    Сохраняет Dyna-Q агента: Q-таблицу и модель переходов (для восстановления или дообучения).
    Возвращает пути к сохранённым файлам (q_table, model).
    """
    os.makedirs(cfg.save_dir, exist_ok=True)
    path_q = os.path.join(cfg.save_dir, cfg.dyna_q_file)
    path_model = os.path.join(cfg.save_dir, cfg.dyna_model_file)
    np.save(path_q, agent.Q)
    with open(path_model, "wb") as f:
        pickle.dump(dict(agent.model._outcomes), f)
    return path_q, path_model


def load_agent_dyna(cfg: Config, n_states: int | None = None, n_actions: int | None = None) -> DynaQAgent:
    """
    Загружает Q-таблицу и модель переходов, создаёт Dyna-Q агента.
    n_states и n_actions определяются по Q, если не заданы.
    """
    path_q = os.path.join(cfg.save_dir, cfg.dyna_q_file)
    path_model = os.path.join(cfg.save_dir, cfg.dyna_model_file)
    if not os.path.exists(path_q):
        raise FileNotFoundError(f"Dyna Q-table file not found: {path_q}")
    Q = np.load(path_q)
    if n_states is None:
        n_states = Q.shape[0]
    if n_actions is None:
        n_actions = Q.shape[1]
    agent = DynaQAgent(n_states, n_actions, cfg)
    agent.Q = Q.astype(np.float64)
    if os.path.exists(path_model):
        with open(path_model, "rb") as f:
            agent.model._outcomes = defaultdict(list, pickle.load(f))
    return agent


def run_comparison(
    cfg: Config | None = None,
    save_agents: bool = True,
    save_rewards_to_file: bool = True,
) -> tuple[list[float], list[float]]:
    """
    Обучает простой Q-learning и Dyna-Q на FrozenLake 8x8 (скользкий лёд),
    строит график сравнения наград и сохраняет в save_dir. Возвращает (rewards_q, rewards_dyna).

    Если save_agents=True, после обучения сохраняются Q-таблица и (для Dyna) модель переходов.
    Если save_rewards_to_file=True, массивы наград сохраняются в rewards_file для последующей загрузки.
    """
    if cfg is None:
        cfg = Config()

    env = gym.make(
        "FrozenLake-v1",
        map_name=cfg.map_name,
        is_slippery=cfg.is_slippery,
    )
    n_states = int(env.observation_space.n)
    n_actions = int(env.action_space.n)

    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    # Обучаем Q-learning: только реальные переходы, одно TD-обновление на шаг
    agent_q = QLearningAgent(n_states, n_actions, cfg)
    pbar_q = trange(cfg.total_episodes, desc="Q-learning")
    rewards_q = run_episodes(env, agent_q, cfg.total_episodes, cfg, is_dyna=False, pbar=pbar_q)
    pbar_q.close()

    # Обучаем Dyna-Q: те же переходы + модель + планирование после каждого шага
    env2 = gym.make(
        "FrozenLake-v1",
        map_name=cfg.map_name,
        is_slippery=cfg.is_slippery,
    )
    # Сбрасываем RNG, чтобы Dyna-Q видел ту же последовательность случайных решений при сравнении
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    agent_dyna = DynaQAgent(n_states, n_actions, cfg)
    pbar_dyna = trange(cfg.total_episodes, desc="Dyna-Q")
    rewards_dyna = run_episodes(env2, agent_dyna, cfg.total_episodes, cfg, is_dyna=True, pbar=pbar_dyna)
    pbar_dyna.close()

    env.close()
    env2.close()

    window = cfg.log_window
    if save_rewards_to_file:
        save_rewards(rewards_q, rewards_dyna, cfg)
    save_plot(rewards_q, rewards_dyna, cfg)

    if save_agents:
        save_agent_q(agent_q, cfg)
        save_agent_dyna(agent_dyna, cfg)

    win_rate_q = np.mean(rewards_q[-window:]) * 100
    win_rate_dyna = np.mean(rewards_dyna[-window:]) * 100
    print(f"Q-learning: доля выигрышей (последние {window} эп.) = {win_rate_q:.1f}%")
    print(f"Dyna-Q:     доля выигрышей (последние {window} эп.) = {win_rate_dyna:.1f}%")

    return rewards_q, rewards_dyna


def evaluate_agent(
    agent: QLearningAgent | DynaQAgent,
    env: gym.Env,
    cfg: Config,
    n_episodes: int = 100,
    deterministic: bool = True,
) -> float:
    """
    Оценка агента без исследования: действия выбираются жадно (argmax Q), без случайности.

    Передаём episode=cfg.total_episodes, чтобы epsilon был минимальным; при deterministic=True
    всё равно используется только argmax. Возвращает среднюю награду за n_episodes эпизодов —
    для FrozenLake это доля выигранных эпизодов (награда 0 или 1).
    """
    rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=cfg.seed + 10000 + ep)
        s = _obs_to_state(obs)
        ep_reward = 0.0
        for _ in range(cfg.max_steps_per_episode):
            a = agent.select_action(s, episode=cfg.total_episodes, deterministic=deterministic)
            obs, r, term, trunc, _ = env.step(a)
            s = _obs_to_state(obs)
            ep_reward += float(r)
            if term or trunc:
                break
        rewards.append(ep_reward)
    return float(np.mean(rewards))


def test(
    cfg: Config,
    agent_type: str = "dyna",
    num_episodes: int = 5,
    render: bool = True,
) -> list[float]:
    """
    Загружает сохранённого агента (Q-learning или Dyna-Q) и запускает num_episodes эпизодов
    с визуализацией (render_mode="human"), если render=True. Печатает награду и длину каждого эпизода.

    agent_type: "q" — Q-learning, "dyna" — Dyna-Q. Возвращает список наград по эпизодам.
    """
    env = gym.make(
        "FrozenLake-v1",
        map_name=cfg.map_name,
        is_slippery=cfg.is_slippery,
        render_mode="human" if render else None,
    )
    n_states = int(env.observation_space.n)
    n_actions = int(env.action_space.n)

    if agent_type.lower() == "q":
        agent = load_agent_q(cfg, n_states=n_states, n_actions=n_actions)
    elif agent_type.lower() == "dyna":
        agent = load_agent_dyna(cfg, n_states=n_states, n_actions=n_actions)
    else:
        raise ValueError('agent_type must be "q" or "dyna"')

    episode_rewards: list[float] = []
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=cfg.seed + 20000 + ep)
        s = _obs_to_state(obs)
        ep_reward = 0.0
        ep_length = 0
        for _ in range(cfg.max_steps_per_episode):
            a = agent.select_action(s, episode=cfg.total_episodes, deterministic=True)
            obs, r, term, trunc, _ = env.step(a)
            s = _obs_to_state(obs)
            ep_reward += float(r)
            ep_length += 1
            if term or trunc:
                break
        episode_rewards.append(ep_reward)
        print(f"Эпизод {ep + 1}/{num_episodes}: награда = {ep_reward:.2f}, длина = {ep_length}")

    env.close()
    avg = float(np.mean(episode_rewards))
    print(f"\nСредняя награда за {num_episodes} эпизодов: {avg:.2f} (доля выигрышей: {avg * 100:.1f}%)")
    return episode_rewards


def plot_from_saved_rewards(cfg: Config | None = None) -> str:
    """
    Строит график сравнения по ранее сохранённым наградам (load_rewards + save_plot).
    Удобно, если нужно перерисовать график с другим окном или без перезапуска обучения.
    """
    if cfg is None:
        cfg = Config()
    rewards_q, rewards_dyna = load_rewards(cfg)
    return save_plot(rewards_q, rewards_dyna, cfg)


if __name__ == "__main__":
    cfg = Config()
    run_comparison(cfg)  # обучает обоих агентов, сохраняет награды, график и веса

    # После обучения можно перерисовать график из сохранённых наград:
    # plot_from_saved_rewards(cfg)

    # Запуск с визуализацией (загружает сохранённого агента):
    # test(cfg, agent_type="dyna", num_episodes=5, render=True)
