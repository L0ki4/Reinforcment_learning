token = 'df133af2-dccb-4f06-b31f-bf99c02b1ba9'

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
LR = 0.1         # learning rate
EPSILON = 0.9            # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 150   # target update frequency
MEMORY_CAPACITY = 168


env = gym.make('gpn-v0')
env.create_thread(token=token)

torch.set_num_threads(12)

N_ACTIONS = 20
N_STATES = 33

class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.fc2 = nn.Linear(50, 100)
        self.fc2.weight.data.normal_(0, 0.1)  # initialization
        self.fc3 = nn.Linear(100, 200)
        self.fc3.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(200, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.selu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.selu(x)
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        actions_value = self.eval_net.forward(x)
        action = 35 + torch.max(actions_value, 1)[1].data.numpy()[0]
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions

        b_memory = self.memory[-BATCH_SIZE:, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a-35)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)
        print(loss[0].data.numpy())
        # update weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def get_state(s, prev_pr=-1, prev_volume=-1):
    dates = np.datetime64(s['date_time'])
    dates = pd.DatetimeIndex([dates])

    if dates.dayofweek[0] < 6:
        holiday = 0
    else:
        holiday = 1

    month_vec = np.zeros(4)
    month_vec[dates.month[0] % 6] = 1

    day_of_week_vec = np.zeros(7)
    day_of_week_vec[dates.dayofweek[0]] = 1

    hour_vec = np.zeros(24)
    hour_vec[dates.hour] = 1

    return np.concatenate((day_of_week_vec, hour_vec, np.array([holiday, prev_pr])))

if __name__ == '__main__':

    env = gym.make('gpn-v0')
    env.create_thread(token=token)

    dqn = DQN()
    dqn.eval_net.load_state_dict(torch.load('5204_3.model'))
    dqn.target_net.load_state_dict(torch.load('5204_3.model'))
    # model = torch.load('5204_3.model')

    prev_price = 55
    done = env.done
    s = get_state({'date_time': '2018-07-26T23:00:00'}, prev_price)
    a = dqn.choose_action(s)
    s_, r, done, _ = env.step(a)

    prev_price = a
    my_s = get_state(s_, prev_price)

    counter = 0
    while True:
        a = dqn.choose_action(my_s)

        # take action
        s_, r, done, _ = env.step(a)

        # update prevs
        prev_price = a
        prev_volume = r / a

        # postproccess state to features
        my_s_ = get_state(s_, prev_price)

        dqn.store_transition(s, a, r, my_s_)

        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()

        if done:
            break

        print(str(s_['date_time']) + ',' + str(env.cum_reward) + ',' + str(a))
