import pandas as pd
import time
from collections import defaultdict
import matplotlib.pyplot as plt
import math
import numpy as np
import matplotlib.pyplot as plt


def get_trick_fea(datax):
    print('开始统计trick特征:')


    ######################################  trick   ######################################

    # 用户每天点击的第一条
    x = datax.groupby(['user_id'])['context_timestamp'].min().reset_index().rename(
        columns={'context_timestamp': 'user_day_first_look_time'})
    datax = pd.merge(datax, x, on=['user_id'], how='left')
    datax['if_user_day_first_look'] = (datax['context_timestamp'] == datax['user_day_first_look_time']).map(
        lambda x: 1 if x else 0)


    # 用户每天点击的最后一条，强特
    x = datax.groupby(['user_id'])['context_timestamp'].max().reset_index().rename(
        columns={'context_timestamp': 'user_day_last_look_time'})
    datax = pd.merge(datax, x, on=['user_id'], how='left')
    datax['if_user_day_last_look'] = (datax['context_timestamp'] == datax['user_day_last_look_time']).map(
        lambda x: 1 if x else 0)


    # 用户每天点击同一商品的第一条
    x = datax.groupby(['user_id', 'item_id'])['context_timestamp'].min().reset_index().rename(
        columns={'context_timestamp': 'min_time'})
    datax = pd.merge(datax, x, on=['user_id', 'item_id'], how='left')
    datax['if_first'] = (datax['context_timestamp'] == datax['min_time']).map(lambda x: 1 if x else 0)


    # 用户每天点击同一商品的最后一条
    x = datax.groupby(['user_id', 'item_id'])['context_timestamp'].max().reset_index().rename(
        columns={'context_timestamp': 'max_time'})
    datax = pd.merge(datax, x, on=['user_id', 'item_id'], how='left')
    datax['if_last'] = (datax['context_timestamp'] == datax['max_time']).map(lambda x: 1 if x else 0)


    # 用户每天点击同一家商店的最后一次
    x = datax.groupby(['user_id', 'shop_id'])['context_timestamp'].max().reset_index().rename(
        columns={'context_timestamp': 'user_shop_last_time'})
    datax = pd.merge(datax, x, on=['user_id', 'shop_id'], how='left')
    datax['if_user_shop_last'] = (datax['context_timestamp'] == datax['user_shop_last_time']).map(lambda x: 1 if x else 0)




    ######

    begin = time.time()
    datax['user_rank'] = datax.groupby(['user_id'])['context_timestamp'].rank(method='min')
    datax['user_rest_clk'] = datax['user_id_all'] - datax['user_rank']
    print((time.time() - begin) / 60, 'user_rank')


    begin = time.time()
    datax['user_item_rank'] = datax.groupby(['user_id', 'item_id'])['context_timestamp'].rank(method='min')
    datax['user_item_rest_clk'] = datax['user_id_item_id_all'] - datax['user_item_rank']
    print((time.time() - begin)/60, 'item_rank')


    '''
    # 暂时不用
    begin = time.time()
    datax['user_cate_rank'] = datax.groupby(['user_id', 'cate_1'])['context_timestamp'].rank(method='min')
    datax['user_cate_rest_clk'] = datax['user_id_cate_1_all'] - datax['user_cate_rank']
    print((time.time() - begin) / 60, 'cate_rank')
    '''

    begin = time.time()
    datax['user_shop_rank'] = datax.groupby(['user_id', 'shop_id'])['context_timestamp'].rank(method='min')
    datax['user_shop_rest_clk'] = datax['user_id_shop_id_all'] - datax['user_shop_rank']
    print((time.time() - begin)/60, 'shop_rank')

    begin = time.time()
    datax['user_brand_rank'] = datax.groupby(['user_id', 'item_brand_id'])['context_timestamp'].rank(method='min')
    datax['user_brand_rest_clk'] = datax['user_id_item_brand_id_all'] - datax['user_brand_rank']
    print((time.time() - begin)/60, 'brand_rank')


    return datax