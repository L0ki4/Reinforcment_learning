import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import gym_gpn
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from time import sleep

# Hyper Parameters
BATCH_SIZE = 48
LR = 0.001         # learning rate
EPSILON = 0.9            # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 150   # target update frequency
MEMORY_CAPACITY = 100#336  # 336 hours in a 2 weeks try month = 744 hours

env = gym.make('gpn-v0')
env.create_thread(token='a7bf92fc-2bd6-4ab6-9180-9f403f8d490b')

torch.set_num_threads(12)

SAVE_MODELS = False
SAVE_GRAPHS = False
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

N_ACTIONS = 20
N_STATES = 36
x = []
y = []

x_r = []
train_r = []
test_r = []

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
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
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

        if self.learn_step_counter != 0: # update min loss
            if loss[0].data.numpy() < self.min:
                self.min = loss[0].data.numpy()
                # save and print current loss
                print(f'{self.learn_step_counter} -- min = {self.min}')
            x.append(self.learn_step_counter)
            y.append(loss[0].data.numpy())

        # loss plot
        if self.learn_step_counter % 10 == 0:
            plt.figure(1)
            plt.title(f'net = {Net()}, batch = {BATCH_SIZE}, LR = {LR}')
            plt.plot(x,y)
            if SAVE_GRAPHS:
                plt.savefig('loss_graph.png', dpi=100)
            # plt.show()

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

    return np.concatenate((month_vec, day_of_week_vec, hour_vec, np.array([holiday])))


dqn = DQN()
print('\nCollecting experience...')
for i_episode in range(400):
    env.reset()
    ep_r_test = 0
    ep_r_train = 0
    done_train = False
    first_save = True

    s = get_state({'date_time': '2018-06-01T00:00:00'})
    a = dqn.choose_action(s)
    s_, r, done, _ = env.step(a)
    prev_price = a
    prev_volume = r/a
    my_s_ = get_state(s_, prev_price, prev_volume)
    s = my_s_
    ep_r_train += r

    counter = 0
    while True:
        a = dqn.choose_action(s)

        # take action
        s_, r, done, _ = env.step(a)

        # postproccess state to features
        my_s_ = get_state(s_, prev_price, prev_volume)

        # update prevs
        prev_price = a
        prev_volume = r/a

        if done_train:
            ep_r_test += r
        else:
            dqn.store_transition(s, a, r, my_s_)
            ep_r_train += r

        if dqn.memory_counter > MEMORY_CAPACITY and not done_train:
            dqn.learn()

        if done:
            print('Ep: ', i_episode,
                  '| Ep_r_train: ', round(ep_r_train, 2),
                  '| Ep_r_test: ', round(ep_r_test, 2))
            if SAVE_MODELS:  # model saving
                if first_save:
                    model_name = f'models/{dqn.learn_step_counter}_{i_episode}.model'
                    torch.save(dqn.eval_net.state_dict(), model_name)
                    first_save = False
                else:
                    torch.save(dqn.eval_net, f'models/{dqn.learn_step_counter}_{i_episode}.model')

            x_r.append(i_episode)
            train_r.append(ep_r_train)
            test_r.append(ep_r_test)
            if i_episode != 0 and SAVE_GRAPHS:
                plt.figure(2)
                plt.title('Ep: ', i_episode,
                  '| Ep_r_train: ', round(ep_r_train, 2),
                  '| Ep_r_test: ', round(ep_r_test, 2))
                plt.plot(x_r, train_r, 'r')
                plt.plot(x_r, test_r, 'b')
                plt.legend(['train', 'test'])
                plt.savefig('cum_rews.png', dpi=100)
            break

        if s_['date_time'] == '2018-07-26T23:00:00':
            done_train = True

        s = my_s_
        counter += 1