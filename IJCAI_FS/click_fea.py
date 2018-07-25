import pandas as pd
import time
from collections import defaultdict
import matplotlib.pyplot as plt
import math
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import log_loss


def chuli(x, num):
    res = x.hour * num
    res += x.minute // (60 / num)
    return  res


def get_click_fea(datax):
    print('开始统计点击特征:')

    ####################################    分箱统计点击量    ####################################

    datax['hour_2'] = datax['context_timestamp'].map(lambda x: chuli(x, 2))
    datax['hour_4'] = datax['context_timestamp'].map(lambda x: chuli(x, 4))
    datax['hour_6'] = datax['context_timestamp'].map(lambda x: chuli(x, 6))
    datax['hour_12'] = datax['context_timestamp'].map(lambda x: chuli(x, 12))


    a = ['user_id', 'item_id', 'item_price_level', 'item_sales_level', 'user_occupation_id']
    for i in a:
        x = datax.groupby([i, 'hour_2']).size().reset_index().rename(columns={0: i + '_hour_2_all'})
        datax = pd.merge(datax, x, on=[i, 'hour_2'], how='left')


    a = ['user_id', 'item_id', 'item_price_level', 'item_sales_level', 'user_occupation_id']
    for i in a:
        x = datax.groupby([i, 'hour_4']).size().reset_index().rename(columns={0: i + '_hour_4_all'})
        datax = pd.merge(datax, x, on=[i, 'hour_4'], how='left')


    a = ['user_id', 'item_id', 'item_price_level', 'item_sales_level', 'user_occupation_id']
    for i in a:
        x = datax.groupby([i, 'hour_6']).size().reset_index().rename(columns={0: i + '_hour_6_all'})
        datax = pd.merge(datax, x, on=[i, 'hour_6'], how='left')


    a = ['user_id', 'item_id', 'item_price_level', 'item_sales_level', 'user_occupation_id']
    for i in a:
        x = datax.groupby([i, 'hour_12']).size().reset_index().rename(columns={0: i + '_hour_12_all'})
        datax = pd.merge(datax, x, on=[i, 'hour_12'], how='left')


    a = ['user_id', 'item_id', 'item_price_level', 'item_sales_level', 'user_occupation_id']
    for i in a:
        x = datax.groupby([i, 'hour']).size().reset_index().rename(columns={0: i + '_hour_all'})
        datax = pd.merge(datax, x, on=[i, 'hour'], how='left')



    ####################################    商品    ####################################

    a = ['item_id', 'item_brand_id', 'item_price_level', 'item_sales_level']
    for i in a:
        x = datax.groupby(i).size().reset_index().rename(columns={0: i + '_all'})
        datax = pd.merge(datax, x, on=i, how='left')



    ####################################    用户    ####################################

    a = ['user_id', 'user_gender_id']
    for i in a:
        x = datax.groupby(i).size().reset_index().rename(columns={0: i + '_all'})
        datax = pd.merge(datax, x, on=i, how='left')

    a = ['user_id']
    b = ['item_id', 'item_brand_id', 'shop_id', 'item_price_level', 'item_sales_level']
    for i in a:
        for j in b:
            x = datax.groupby([i, j]).size().reset_index().rename(columns={0: i + '_' + j + '_all'})
            datax = pd.merge(datax, x, on=[i, j], how='left')




    ####################################    商店    ####################################

    a = ['shop_id', 'shop_review_num_level', 'shop_star_level']
    for i in a:
        x = datax.groupby(i).size().reset_index().rename(columns={0: i + '_all'})
        datax = pd.merge(datax, x, on=i, how='left')




    return  datax