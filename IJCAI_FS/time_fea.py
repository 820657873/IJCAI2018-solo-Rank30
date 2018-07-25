import pandas as pd
import time
from collections import defaultdict
import matplotlib.pyplot as plt
import math
import numpy as np
import matplotlib.pyplot as plt



def get_time_fea(datax):
    print('开始统计时间差特征:')


    begin = time.time()
    # 用户每天此次点击距上一次点击，下一次点击的时间差
    print('耗时间的:')
    x = datax.sort_values(['user_id', 'day', 'context_timestamp'])
    x['former_time_cha'] = x.groupby(['user_id', 'day'])['context_timestamp'].diff(periods=1)
    x['later_time_cha'] = x.groupby(['user_id', 'day'])['context_timestamp'].diff(periods=-1)
    datax['former_time_cha'] = x['former_time_cha'].map(lambda x: x.seconds)
    datax['later_time_cha'] = x['later_time_cha'].map(lambda x: -x.seconds)
    datax['if_czxtdj'] = ((datax['former_time_cha'] == 0) | (datax['later_time_cha'] == 0)).map(
        lambda x: 1 if x else 0)
    print((time.time() - begin) / 60, 'duojiu')



    # 距用户每天第一次点击的时间差
    x = datax.groupby(['user_id', 'day'])['context_timestamp'].min().reset_index().rename(
        columns={'context_timestamp': 'user_day_min_time'})
    datax = pd.merge(datax, x, on=['user_id', 'day'], how='left')
    datax['user_day_first_time_cha'] = (datax['context_timestamp'] - datax['user_day_min_time']).map(
        lambda x: x.seconds)

    # 距用户每天最后一次点击的时间差
    x = datax.groupby(['user_id', 'day'])['context_timestamp'].max().reset_index().rename(
        columns={'context_timestamp': 'user_day_max_time'})
    datax = pd.merge(datax, x, on=['user_id', 'day'], how='left')
    datax['user_day_last_time_cha'] = (datax['user_day_max_time'] - datax['context_timestamp']).map(
        lambda x: x.seconds)


    # 用户每天浏览的总时长
    datax['user_day_time_duration'] = (datax['user_day_max_time'] - datax['user_day_min_time']).map(
        lambda x: x.seconds)



    # 用户每天距离点击同一个商品第一次的时间差
    x = datax.groupby(['user_id', 'day', 'item_id'])['context_timestamp'].min().reset_index().rename(
        columns={'context_timestamp': 'same_min_time'})
    datax = pd.merge(datax, x, on=['user_id', 'day', 'item_id'], how='left')
    datax['same_item_first_time_cha'] = (datax['context_timestamp'] - datax['same_min_time']).map(lambda x: x.seconds)

    # 用户每天距离点击同一个商品最后一次的时间差
    x = datax.groupby(['user_id', 'day', 'item_id'])['context_timestamp'].max().reset_index().rename(
        columns={'context_timestamp': 'same_max_time'})
    datax = pd.merge(datax, x, on=['user_id', 'day', 'item_id'], how='left')
    datax['same_item_last_time_cha'] = (datax['same_max_time'] - datax['context_timestamp']).map(lambda x: x.seconds)




    return datax