import numpy as np
import pandas as pd


def get_state(s, reward_dict, price_dict):
    dates = np.datetime64(s['date_time'])
    dates = pd.DatetimeIndex([dates])

    if dates.dayofweek[0] < 6:
        holiday = 0
    else:
        holiday = 1

    day_of_week_vec = np.zeros(7)
    day_of_week_vec[dates.dayofweek[0]] = 1

    hour_vec = np.zeros(24)
    hour_vec[dates.hour] = 1

    try:
        hour_reward = reward_dict[s['date_time']]
        hour_price = price_dict[s['date_time']]
    except:
        hour_reward = 0
        hour_price = 0

    try:
        day_reward = reward_dict[str(np.datetime64(s['date_time']) - np.timedelta64(23, 'h'))]
        day_price = price_dict[str(np.datetime64(s['date_time']) - np.timedelta64(23, 'h'))]
    except:
        day_reward = 0
        day_price = 0

    try:
        week_reward = reward_dict[str(np.datetime64(s['date_time']) - np.timedelta64(6, 'D') - np.timedelta64(23, 'h'))]
        week_price = price_dict[str(np.datetime64(s['date_time']) - np.timedelta64(6, 'D') - np.timedelta64(23, 'h'))]
    except:
        week_reward = 0
        week_price = 0

    return np.concatenate((day_of_week_vec, hour_vec, np.array([holiday, hour_reward, hour_price, day_reward,
                            day_price, week_reward, week_price])))
