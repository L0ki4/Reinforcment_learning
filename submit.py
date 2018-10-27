import torch
import gym
import gym_gpn
import constants
import model
from model import Net
from functions import get_state
import matplotlib
if constants.MACHINE == 'remote':
    matplotlib.use('agg')
import matplotlib.pyplot as plt

env = gym.make('gpn-v0')
env.create_thread(token=constants.TOKEN)

torch.set_num_threads(12)

x_r = []
train_r = []
test_r = []

x_week = []
y_week = []

x_day = []
y_day = []


def submit():
    dqn = model.DQN()
    dqn.eval_net.load_state_dict(torch.load('models/15948_11.model'))
    dqn.target_net.load_state_dict(torch.load('models/15948_11.model'))

    prev_price = 55
    done = env.done
    if constants.MODE == 'train':
        s = get_state({'date_time': '2018-07-26T23:00:00'}, prev_price)
    else:
        s = get_state({'date_time': '2018-06-01T00:00:00'})
    a = dqn.get_action(s)

    s_, r, done, _ = env.step(a)

    y_week.append(r)
    y_day.append(r)

    prev_price = a
    my_s = get_state(s_, prev_price)

    counter = 0
    while True:
        a = dqn.get_action(my_s)

        # take action
        s_, r, done, _ = env.step(a)
        y_week[counter // (7 * 24)] += r
        y_day[counter // 24] += r

        # update prevs
        prev_price = a

        # postproccess state to features
        my_s_ = get_state(s_, prev_price)

        dqn.store_transition(s, a, r, my_s_)

        if dqn.memory_counter > constants.MEMORY_CAPACITY:
            dqn.learn()

        if done:
            break

        print(str(s_['date_time']) + ',' + str(env.cum_reward) + ',' + str(a))

        counter += 1

        if counter % (7 * 24) == 0:
            x_week.append(counter // (7 * 24))
            y_week[counter // (7 * 24) - 1] /= 7 * 24
            plt.figure(3)
            plt.title('mean by week')
            plt.plot(x_week, y_week, 'r')
            if constants.SAVE_GRAPHS:
                plt.savefig('mean_week.png', dpi=100)
            y_week.append(0)

        if counter % 24 == 0:
            x_day.append(counter // 24)
            y_day[counter // 24 - 1] /= 24
            plt.figure(4)
            plt.title('mean by day')
            plt.plot(x_day, y_day, 'r')
            if constants.SAVE_GRAPHS:
                plt.savefig('mean_day.png', dpi=100)
            y_day.append(0)


if __name__ == '__main__':
    submit()
