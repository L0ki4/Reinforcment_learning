import gym
from gym import spaces
import numpy as np
import requests


class GpnEnv(gym.Env):
    '''
    Среда, моделирующая спрос на топливо в зависимости от установленной цены
    и основанная на OpenAI Gym.
    '''
    def __init__(self,
                 url='http://flask-env.hemmumzjwr.us-east-1.elasticbeanstalk.com',
                 low_price=30, high_price=55,
                 cum_reward=0, state=None, timeout=5):
        '''
        Инициализация среды.

        Аргументы:
            url (): адрес сервера, от которого в среду поступают наблюдения
            low_price (int or float): нижняя цена
            high_price (int or float): верхняя цена
            cum_reward (): накопленное вознаграждение
            state (dict): текущее состояние
            timeout (int): (не используется)
        '''
        self.token = None
        self.url = url
        self.cum_reward = cum_reward
        self.reward = 0
        self.state = state
        self.done = False
        self.timeout = timeout
        self.low_price = low_price
        self.high_price = high_price


    def create_thread(self, token):
        '''
        Создание инстанса вирутальной АЗС.
        '''
        self.token = token

        params = {'token':self.token, 'timeout':self.timeout}
        req = requests.post(self.url + '/create_thread', params=params)
        self.thread = req.json()['thread']


    def step(self, new_price):
        '''
        Совершение действия в среде – установка цены. Среда принимает действие
        и возвращает кортеж (observation, reward, done, info).

        По окончании эпизода, среда возвращает флаг `done = True` и последнее
        доступное состояние. Для приведения среды в начальное состояние необходимо
        использовать  метод `.reset()`.


        Аргументы:
            new_price (int or float): действие, осуществляемое средой – задание цены

        Returns:
            observation (dict): наблюдение агента по итогам совершенного действия
            reward (float): сумма вознаграждения, полученного после совершенного действия
            done (boolean): флаг окончания эпизода
            info (dict): может содержать вспомогательную диагностическую информацию
        '''
        if self.done:
            return self.state, self.reward, self.done, {}

        if new_price > self.high_price:
            new_price = self.high_price
        elif new_price < self.low_price:
            new_price = self.low_price

        new_price = np.round(new_price, 2)
        params = {'price':new_price, 'thread':self.thread, 'timeout':self.timeout}
        req = requests.get(self.url + '/set_price', params=params)
        if (req.status_code == 500):
            if (req.json()['message'] == 'Internal server error : index 0 is out of bounds for axis 0 with size 0'):
                self.done = True
                return self.state, self.reward, self.done, {}

        self.state = req.json()
        self.reward = self.state['price'] * self.state['volume']
        self.cum_reward += self.reward

        return self.state, self.reward, self.done, {}


    def reset(self):
        '''
        Сбрасывает текущее состояние среды и возвращает начальное наблюдение.

        Returns:
            observation (object): начальное наблюдение
        '''

        params = {'thread':self.thread, 'timeout':self.timeout}
        requests.put(self.url + '/reset_thread', params=params)
        self.done = False
        self.state = None
        self.cum_reward = 0
