import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import constants
import matplotlib
if constants.MACHINE == 'remote':
    matplotlib.use('agg')
import matplotlib.pyplot as plt

x = []
y = []


class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(constants.N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.fc2 = nn.Linear(50, 100)
        self.fc2.weight.data.normal_(0, 0.1)  # initialization
        self.fc3 = nn.Linear(100, 200)
        self.fc3.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(200, constants.N_ACTIONS)
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
        self.memory = np.zeros((constants.MEMORY_CAPACITY, constants.N_STATES * 2 + 2))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=constants.LR)
        self.loss_func = nn.MSELoss()
        self.min = np.inf

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        if np.random.uniform() < constants.EPSILON:   # greedy
            actions_value = self.eval_net.forward(x)
            action = 35 + torch.max(actions_value, 1)[1].data.numpy()[0]
        else:   # random
            action = np.random.randint(35, 35+constants.N_ACTIONS)
        return action

    def get_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        actions_value = self.eval_net.forward(x)
        return 35 + torch.max(actions_value, 1)[1].data.numpy()[0]

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % constants.MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % constants.TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions

        b_memory = self.memory[-constants.BATCH_SIZE:, :]
        b_s = torch.FloatTensor(b_memory[:, :constants.N_STATES])
        b_a = torch.LongTensor(b_memory[:, constants.N_STATES:constants.N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, constants.N_STATES+1:constants.N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -constants.N_STATES:])
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        b_s.to(device)
        b_a.to(device)
        b_r.to(device)
        b_s_.to(device)
        self.eval_net.to(device)
        self.target_net.to(device)

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a-35)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + constants.GAMMA * q_next.max(1)[0].view(constants.BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        if self.learn_step_counter != 0 : # update min loss
            if loss[0].data.numpy() < self.min:
                self.min = loss[0].data.numpy()
                # save and print current loss
                print(f'{self.learn_step_counter} -- min = {self.min}')
            x.append(self.learn_step_counter)
            y.append(loss[0].data.numpy())

        # loss plot
        if self.learn_step_counter % 10 == 0:
            plt.figure(1)
            plt.title(f'net = {Net()}, batch = {constants.BATCH_SIZE}, LR = {constants.LR}')
            plt.plot(x,y)
            if constants.SAVE_GRAPHS:
                plt.savefig('loss_graph.png', dpi=100)
            if constants.MACHINE == 'local':
                plt.show()

        # update weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
