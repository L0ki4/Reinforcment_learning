token = 'a7bf92fc-2bd6-4ab6-9180-9f403f8d490b'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import gym_gpn
import pandas as pd

from time import sleep

# Hyper Parameters
BATCH_SIZE = 24
LR = 0.05                # learning rate
EPSILON = 0.9            # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY = 500
env = gym.make('gpn-v0')
env.create_thread(token = 'a7bf92fc-2bd6-4ab6-9180-9f403f8d490b')

N_ACTIONS = 20
N_STATES = 4
prev_loss = 0

class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.fc2 = nn.Linear(50, 100)
        self.fc2.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(100, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value



class Agent:
    '''
    Шаблон агентной модели для среды Gym-GPN
    '''
    def __init__(self, observations=None):
        self.observations = observations or []


    def act(self, x = [8, 24, 0, 0]):
        model = Net()
        model.load_state_dict(torch.load('1000_2603676.5.model'))

        model = torch.load('14000_986261.375.model')

        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        actions_value = model.forward(x)
        return 35 + torch.max(actions_value, 1)[1].data.numpy()[0]

class RandomWalkAgent(Agent):
    '''
    Агентная модель для среды Gym-GPN со случайным выбором цены
    '''
    def __init__(self, low_price, high_price):
        self.low_price = low_price
        self.high_price = high_price


    def act(self):
        return np.round(np.random.randint(self.low_price, self.high_price + 1), 2)

def get_state(s):
    dates = np.datetime64(s['date_time'])
    dates = pd.DatetimeIndex([dates])
    if dates.dayofweek[0] < 6:
        holiday = 0
    else:
        holiday = 1

    return [dates.month[0], dates.day[0], dates.hour[0], holiday]

if __name__ == '__main__':

    import gym
    import gym_gpn

    env = gym.make('gpn-v0')
    env.create_thread(token=token)
    agnt = Agent()

    ob = None
    done = env.done
    while not done:
        if ob == None:
            price = agnt.act()
        else:
            price = agnt.act(get_state(ob))
        ob, reward, done, _  = env.step(price)
        print(str(ob['date_time']) + ',' + str(env.cum_reward) + ',' + str(price))
