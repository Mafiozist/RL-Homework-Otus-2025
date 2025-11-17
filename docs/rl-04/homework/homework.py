import gymnasium as gym
import numpy as np

class Config:
    lr: float = 0.001 #alpha - то,насколько новое значение обновляет старое 
    gamma: float = 0.99 # коэф дисконтирования - вероятность, что в конретный ход прервется выполнение 
    epsilon: float = 0.1 # степень исследования - вероятность того, что в какой-то момент времени агент сделает рандомное действие


class TaxiWrapper(gym.Wrapper):
    env = None
    Q = None
    episode_rewards = [] #награда выдаваемая по эпизодам, где индекс массива - номер эпизода
    hyperparams : Config = None
    actions = []

    def __init__(self, hyperparams : Config = {}):
        env = gym.make('Taxi-v3', 300, render_mode="human")
        self.env = env
        
        #Инициализируем таблицу весов* Q-функции
        self.Q = np.zeros((env.observation_space.n, env.action_space.n))
        
        #Задаем гиперпараметры обучения
        self.hyperparams = hyperparams

        self.actions = range(0, env.action_space.n)

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
                # при условии, что у нас еще нет наработанного результата в Q-таблице
                if self.Q[s0][np.argmax(self.Q[s0])] == 0:
                    action : int = np.random.choice(allowed_indices)
                
                #Иначе мы берем максимальное значение по таблице
                else:
                    action : int = np.argmax(self.Q[s0]) 

                # осуществляем действие и получаем обратную связь от среды
                s, reward, terminated, truncated, info = super().step(action)

                # заносим текущие веса в Q-таблицу
                self.Q[s0, action] = reward
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

            

if __name__ == '__main__':
    
    #Инициализация среды
    taxiEnv = TaxiWrapper()
    
    taxiEnv.startLearning(1000, False)

    