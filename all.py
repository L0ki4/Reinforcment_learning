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

def train():
    dqn = model.DQN()

    print('\nCollecting experience...')
    for i_episode in range(400):
        env.reset()
        ep_r_test = 0
        ep_r_train = 0
        done_train = False

        s = get_state({'date_time': '2018-06-01T00:00:00'})
        a = dqn.choose_action(s)
        s_, r, done, _ = env.step(a)
        prev_price = a
        my_s_ = get_state(s_, prev_price)
        s = my_s_
        ep_r_train += r

        counter = 0
        while True:
            a = dqn.choose_action(s)

            # take action
            s_, r, done, _ = env.step(a)

            # update prevs
            prev_price = a

            # postproccess state to features
            my_s_ = get_state(s_, prev_price)

            if done_train:
                ep_r_test += r
            else:
                dqn.store_transition(s, a, r, my_s_)
                ep_r_train += r

            if dqn.memory_counter > constants.MEMORY_CAPACITY and not done_train:
                dqn.learn()

            if done:
                print('Ep: ', i_episode,
                      '| Ep_r_train: ', round(ep_r_train, 2),
                      '| Ep_r_test: ', round(ep_r_test, 2))
                if constants.SAVE_MODELS:  # model saving
                    model_name = f'models/{dqn.learn_step_counter}_{i_episode}.model'
                    torch.save(dqn.eval_net.state_dict(), model_name)

                x_r.append(i_episode)
                train_r.append(ep_r_train)
                test_r.append(ep_r_test)
                if i_episode != 0 and constants.SAVE_GRAPHS:
                    plt.figure(2)
                    plt.plot(x_r, train_r, 'r')
                    plt.plot(x_r, test_r, 'b')
                    plt.legend(['train', 'test'])
                    plt.savefig('cum_rews.png', dpi=100)
                    if constants.MACHINE == 'local':
                        plt.show()
                break

            if s_['date_time'] == '2018-07-26T23:00:00':
                done_train = True

            s = my_s_
            counter += 1


if __name__ == '__main__':
    train()
