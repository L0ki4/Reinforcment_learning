import torch
import gym
import gym_gpn
import matplotlib
import constants
import model
from functions import get_state
from model import Net

if constants.MACHINE == 'remote':
    matplotlib.use('agg')
import matplotlib.pyplot as plt

env = gym.make('gpn-v0')
env.create_thread(token=constants.TOKEN)

torch.set_num_threads(12)

x_r = []
train_r = []
test_r = []

# x_week = []
# y_week = []
#
# x_day = []
# y_day = []

def train():
    dqn = model.DQN()

    print('\nCollecting experience...')
    for i_episode in range(400):
        env.reset()
        ep_r_test = 0
        ep_r_train = 0
        done_train = False
        price_dict = dict()
        reward_dict = dict()

        s = get_state({'date_time': '2018-05-31T23:00:00'}, reward_dict, price_dict)

        a = dqn.choose_action(s)
        s_, r, done, _ = env.step(a)

        price_dict[s_['date_time']] = a
        reward_dict[s_['date_time']] = r

        my_s_ = get_state(s_, reward_dict, price_dict)
        s = my_s_
        ep_r_train += r

        # y_week.append(r)
        # y_day.append(r)

        counter = 1
        for hours in range(7*24):
            a = dqn.choose_action(s)
            s_, r, done, _ = env.step(a)

            price_dict[s_['date_time']] = a
            reward_dict[s_['date_time']] = r

            my_s_ = get_state(s_, reward_dict, price_dict)
            s = my_s_
            ep_r_train += r

            # y_week.append(r)
            # y_day.append(r)

            counter += 1
        while True:
            if not done_train:
                a = dqn.choose_action(s)
            else:
                a = dqn.get_action(s)

            # take action
            s_, r, done, _ = env.step(a)
            price_dict[s_['date_time']] = a
            reward_dict[s_['date_time']] = r

            # y_week[counter // (7*24)] += r
            # y_day[counter // 24] += r

            # update prevs
            date = s_['date_time']
            print(f'at {date} with price = {a} reward is {r}')

            # postproccess state to features
            get_state(s_, reward_dict, price_dict)

            if done_train:
                ep_r_test += r
            else:
                dqn.store_transition(s, a, r, my_s_)
                ep_r_train += r

            if dqn.memory_counter > constants.MEMORY_CAPACITY and not done_train:
                dqn.learn()

            if s_['date_time'] == '2018-07-26T23:00:00':
                done_train = True

            s = my_s_
            counter += 1

            # if counter % (7*24) == 0:
            #     x_week.append(counter // (7*24))
            #     y_week[counter // (7*24) - 1] /= 7*24
            #     plt.figure(3)
            #     plt.title('mean by week')
            #     plt.plot(x_week, y_week, 'r')
            #     if constants.SAVE_GRAPHS:
            #         plt.savefig('mean_week.png', dpi=100)
            #     y_week.append(0)

            # if counter % 24 == 0:
            #     x_day.append(counter // 24)
            #     y_day[counter // 24 - 1] /= 24
            #     plt.figure(4)
            #     plt.title('mean by day')
            #     plt.plot(x_day, y_day, 'r')
            #     if constants.SAVE_GRAPHS:
            #         plt.savefig('mean_day.png', dpi=100)
            #     y_day.append(0)

            if done:
                print('Ep: ', i_episode,
                      '| Ep_r_train: ', round(ep_r_train, 2),
                      '| Ep_r_test: ', round(ep_r_test, 2))
                if constants.SAVE_MODELS:  # model saving
                    model_name = f'models/{dqn.learn_step_counter}_{i_episode}.model'
                    torch.save(dqn.eval_net.state_dict(), model_name)

                x_r.append(i_episode)
                train_r.append(ep_r_train/1344)
                test_r.append(ep_r_test/672)
                if i_episode != 0 and constants.SAVE_GRAPHS:
                    plt.figure(2)
                    plt.plot(x_r, train_r, 'r')
                    plt.plot(x_r, test_r, 'b')
                    plt.legend(['train', 'test'])
                    plt.savefig('cum_rews.png', dpi=100)
                    if constants.MACHINE == 'local':
                        plt.show()
                break


if __name__ == '__main__':
    train()
