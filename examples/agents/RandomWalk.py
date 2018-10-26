import numpy as np
from time import sleep

token = 'df133af2-dccb-4f06-b31f-bf99c02b1ba9'


class Agent:
    '''
    Шаблон агентной модели для среды Gym-GPN
    '''
    def __init__(self, observations=None):
        self.observations = observations or []

    def act(self):
        raise NotImplementedError


class RandomWalkAgent(Agent):
    '''
    Агентная модель для среды Gym-GPN со случайным выбором цены
    '''
    def __init__(self, low_price, high_price):
        self.low_price = low_price
        self.high_price = high_price


    def act(self):
        return np.round(np.random.randint(self.low_price, self.high_price + 1), 2)


if __name__ == '__main__':

    import gym
    import gym_gpn

    env = gym.make('gpn-v0')
    env.create_thread(token=token)
    agnt = RandomWalkAgent(env.low_price, env.high_price)

    done = env.done
    while not done:
        price = agnt.act()
        ob, reward, done, _  = env.step(price)
        print(str(ob['date_time']) + ',' + str(env.cum_reward))