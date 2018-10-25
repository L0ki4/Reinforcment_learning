import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import gym_gpn
import pandas as pd
import matplotlib.pyplot as plt

from time import sleep

# Hyper Parameters
BATCH_SIZE = 48
LR = 0.01         # learning rate
EPSILON = 0.9            # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY = 1000

env = gym.make('gpn-v0')
env.create_thread(token = 'a7bf92fc-2bd6-4ab6-9180-9f403f8d490b')

torch.set_num_threads(12)

SAVE_MODELS = False
SAVE_GRAPHS = False
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

N_ACTIONS = 20
N_STATES = 38
prev_loss = 0
x = []
y = []

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


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.prev_loss = 0
        self.min = np.inf

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        if np.random.uniform() < EPSILON:   # greedy
            actions_value = self.eval_net.forward(x)
            action = 35 + torch.max(actions_value, 1)[1].data.numpy()[0]
        else:   # random
            action = np.random.randint(35, 35+N_ACTIONS)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        first_save = True
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a-35)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)
        if self.learn_step_counter != 0:
            if loss[0].data.numpy() < self.min:
                self.min = loss[0].data.numpy()
                if SAVE_MODELS:
                    if first_save:
                        torch.save(self.eval_net.state_dict(), f'models/{self.learn_step_counter}_{loss[0].data.numpy()}.model')
                    else:
                        torch.save(self.eval_net, f'models/{self.learn_step_counter}_{self.min}.model')

            print(f'{self.learn_step_counter} -- {loss[0].data.numpy()}, min = {self.min}')
            x.append(self.learn_step_counter)
            y.append(loss[0].data.numpy())

        if self.learn_step_counter % 10 == 0:
            plt.title(f'net = {Net()}, batch = {BATCH_SIZE}, LR = {LR}')
            plt.plot(x,y)
            if SAVE_GRAPHS:
                plt.savefig('loss_graph.pdf', dpi = 100)
            plt.show()

        self.prev_loss = loss[0].data.numpy()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def get_state(s, prev_pr=-1, prev_reward=-1):
    dates = np.datetime64(s['date_time'])
    dates = pd.DatetimeIndex([dates])

    if dates.dayofweek[0] < 6:
        holiday = 0
    else:
        holiday = 1

    month_vec = np.zeros(4)
    month_vec[dates.month[0]%6] = 1

    day_of_week_vec = np.zeros(7)
    day_of_week_vec[dates.dayofweek[0]] = 1

    hour_vec = np.zeros(24)
    hour_vec[dates.hour] = 1

    return np.concatenate((month_vec, day_of_week_vec, hour_vec, np.array([holiday, prev_pr, prev_reward])))

dqn = DQN()
print('\nCollecting experience...')
for i_episode in range(400):
    env.reset()
    ep_r = 0

    s = get_state({'date_time':'2018-06-01T00:00:00'})
    a = dqn.choose_action(s)
    s_, r, done, info = env.step(a)
    prev_price = a
    prev_reward = r
    my_s_ = get_state(s_, prev_price, prev_reward)
    s = my_s_
    ep_r += r

    counter = 0
    while True:
        # env.render()
        a = dqn.choose_action(s)

        # take action
        s_, r, done, info = env.step(a)

        my_s_ = get_state(s_, prev_price, prev_reward)
        prev_price = a
        prev_reward = r
        dqn.store_transition(s, a, r, my_s_)

        ep_r += r
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()
            if done:
                print('Ep: ', i_episode,
                      '| Ep_r: ', round(ep_r, 2))

        if done:
            break
        s = my_s_
        counter += 1