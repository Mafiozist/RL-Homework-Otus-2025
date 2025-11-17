import gymnasium as gym
import numpy as np
from dataclasses import dataclass
import matplotlib as plt
import os

@dataclass
class Config:
    lr: float = 0.01 #alpha - то,насколько новое значение обновляет старое 
    gamma: float = 0.1 # коэф дисконтирования - вероятность, что в конретный ход прервется выполнение 
    epsilon: float = 0.95 # степень исследования - вероятность того, что в какой-то момент времени агент сделает рандомное действие


class TaxiWrapper(gym.Wrapper):
    env = None
    Q = None
    episode_rewards = [] #награда выдаваемая по эпизодам, где индекс массива - номер эпизода
    hyperparams : Config = None
    actions = []

    def __init__(self, hyperparams : Config = Config()):
        env = gym.make('Taxi-v3', render_mode="human")
        self.env = env
        
        #Инициализируем таблицу весов* Q-функции
        self.Q = np.zeros((env.observation_space.n, env.action_space.n))
        
        #Задаем гиперпараметры обучения
        self.hyperparams = hyperparams
        self.actions = range(0, env.action_space.n)
        self.episode_rewards = []

        super().__init__(env)
    
    def startLearning(self, episodes = 1000, saveWeights = True):
        #Получаем стартовую позицию и маску допустимых действий для текущей позиции
        s0, init_info = self.env.reset()
        mask = init_info['action_mask']

        #На данном этапе осуществляет
        for i in range(episodes):
            reward_sum = 0

            #Гоняем в рамках эпизода до тех пор пока не получим результат явное заврешение эпизода
            while True:
                allowed_indices = np.where(mask == 1)[0]
                action = None

                # первый шаг выбираем рандомно на основании маски допустимых действий
                # с учетом коэффициента исследования
                if np.random.rand() < self.hyperparams.epsilon:
                    action : int = np.random.choice(allowed_indices)
                
                #Иначе мы берем максимальное значение по таблице с учетом маски
                else:
                    q_values = self.Q[s0, allowed_indices]
                    best_local_index = np.argmax(q_values)
                    action = allowed_indices[best_local_index]

                # осуществляем действие и получаем обратную связь от среды
                s, reward, terminated, truncated, info = super().step(action)

                # заносим текущие веса в Q-таблицу
                # производим мягкое обновление параметров
                # gamma * max(Q[S]) - оценка будущей выгоды
                # reward + gamma * max(Q[S]) - целевое значение Q
                # reward + gamma * max(Q[S]) - S0, разница между  Q будщим, и старым Q 
                self.Q[s0, action] += self.hyperparams.lr * (  
                    reward 
                    + self.hyperparams.gamma * np.max(self.Q[s]) # будущая выгода
                    - self.Q[s0, action] # текущая оценка
                )
                mask = info['action_mask']
                s0 = s
                reward_sum += reward

                #Если получаем знак остановки сбрасываем среду, но продолжаем работать по уже готовой таблице

                if terminated or truncated:
                    s0, init_info = self.env.reset()
                    mask = init_info['action_mask']
                    self.episode_rewards.append(reward_sum)
                    print('episode ' , i, ' reward: ', reward_sum)
                    break

        if saveWeights:
            self.save_results()

        self.print_results()

    
    def print_results(self):
        rewards = np.array(self.episode_rewards)
        episodes = np.arange(len(rewards))

        # Скользящее среднее
        window = 100
        if len(rewards) >= window:
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode="valid")
        else:
            moving_avg = None

        plt.figure(figsize=(12, 6))
        plt.plot(episodes, rewards, label="Награда за эпизод", alpha=0.5)

        if moving_avg is not None:
            plt.plot(np.arange(window - 1, len(rewards)), moving_avg,
                    label="Скользящее среднее (100 эпизодов)", linewidth=2)

        plt.xlabel("Эпизод")
        plt.ylabel("Суммарная награда")
        plt.title(
            f"Обучение агента\n"
            f"Суммарная награда: {rewards.sum():.2f}, "
            f"Средняя за последние 100 эпизодов: {np.mean(rewards[-100:]):.2f}"
        )
        plt.legend()
        plt.grid(True)

        plt.savefig("reward_plot.png")
        plt.close()
        

    def save_results(self):
        # Сохранение Q-таблицы
        np.save("q_table.npy", self.Q)
    

    def load_results(self, filename="q_table.npy"):
        if not os.path.exists(filename):
            print(f"Файл '{filename}' не найден. Загрузка невозможна.")
            return False
        
        try:
            q_loaded = np.load(filename)
        except Exception as e:
            print(f"Ошибка при загрузке файла: {e}")
            return False

        # Проверяем совпадение формы Q-таблицы
        if q_loaded.shape != self.Q.shape:
            print(f"Несовместимая форма Q-таблицы! "
                f"Ожидалось {self.Q.shape}, получено {q_loaded.shape}")
            return False

        self.Q = q_loaded.copy()
        print(f"Q-таблица успешно загружена из '{filename}'.")
        return True



if __name__ == '__main__':
    
    #Инициализация среды 
    taxiEnv = TaxiWrapper()
    taxiEnv.load_results()
    taxiEnv.startLearning(1000, False)

    