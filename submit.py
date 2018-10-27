import torch
import gym
import gym_gpn
import constants
import model
from model import Net
from functions import get_state

env = gym.make('gpn-v0')
env.create_thread(token=constants.TOKEN)

torch.set_num_threads(12)

x_r = []
train_r = []
test_r = []


def submit():
    dqn = model.DQN()
    dqn.eval_net.load_state_dict(torch.load('SELU/13262_9.model'))
    dqn.target_net.load_state_dict(torch.load('SELU/13262_9.model'))

    prev_price = 55
    done = env.done
    s = get_state({'date_time': '2018-07-26T23:00:00'}, prev_price)
    a = dqn.get_action(s)

    s_, r, done, _ = env.step(a)

    prev_price = a
    my_s = get_state(s_, prev_price)

    while True:
        a = dqn.get_action(my_s)

        # take action
        s_, r, done, _ = env.step(a)

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


if __name__ == '__main__':
    submit()
