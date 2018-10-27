import numpy as np
import pandas as pd


def get_state(s, prev_pr=-1):
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
