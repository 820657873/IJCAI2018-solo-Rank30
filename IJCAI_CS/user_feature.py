import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss
import time
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt
import math
import numpy as np
from Bayesi import BayesianSmoothing
from sklearn.preprocessing import LabelEncoder
from ad_feature import get_cvr, get_zuhe_cvr
from ad_feature import get_score

# ------------------------------------用户相关特征------------------------------------


def user_fea_extract(datax, startday, endday):
    print('开始统计用户特征:')

    ####################################    每天点击量 单特征    ####################################

    # 用户每天的点击量
    x = datax.groupby(['user_id', 'day']).size().reset_index().rename(columns={0: 'user_day_sum'})
    datax = pd.merge(datax, x, on=['user_id', 'day'], how='left')

    # 用户每天每小时的点击量
    x = datax.groupby(['user_id', 'day', 'hour']).size().reset_index().rename(columns={0: 'user_day_hour_sum'})
    datax = pd.merge(datax, x, on=['user_id', 'day', 'hour'], how='left')

    # 用户每天每小时每分钟的点击量
    x = datax.groupby(['user_id', 'day', 'hour', 'minute']).size().reset_index().rename(columns={0: 'user_day_hour_minute_sum'})
    datax = pd.merge(datax, x, on=['user_id', 'day', 'hour', 'minute'], how='left')

    a = ['user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level']
    for i in a:
        x = datax.groupby([i, 'day']).size().reset_index().rename(columns={0: i + '_day_sum'})
        datax = pd.merge(datax, x, on=[i, 'day'], how='left')
        x = datax.groupby([i, 'day', 'hour']).size().reset_index().rename(columns={0: i + '_day_hour_sum'})
        datax = pd.merge(datax, x, on=[i, 'day', 'hour'], how='left')

    
    ####################################    每天点击量 用户对商品特征    ####################################

    # 用户每天对每件商品的点击量
    x = datax.groupby(['user_id', 'day', 'item_id']).size().reset_index().rename(columns={0: 'user_day_item_sum'})
    datax = pd.merge(datax, x, on=['user_id', 'day', 'item_id'], how='left')

    # 用户每天每小时对每件商品的点击量
    x = datax.groupby(['user_id', 'day', 'hour', 'item_id']).size().reset_index().rename(columns={0: 'user_day_hour_item_sum'})
    datax = pd.merge(datax, x, on=['user_id', 'day', 'hour', 'item_id'], how='left')

    # 用户每天对品牌的点击量
    x = datax.groupby(['user_id', 'day', 'item_brand_id']).size().reset_index().rename(columns={0: 'user_day_brand_sum'})
    datax = pd.merge(datax, x, on=['user_id', 'day', 'item_brand_id'], how='left')

    # 用户每天对价格的点击量
    x = datax.groupby(['user_id', 'day', 'item_price_level']).size().reset_index().rename(columns={0: 'user_day_price_sum'})
    datax = pd.merge(datax, x, on=['user_id', 'day', 'item_price_level'], how='left')

    # 用户每天对销量的点击量
    x = datax.groupby(['user_id', 'day', 'item_sales_level']).size().reset_index().rename(columns={0: 'user_day_sales_sum'})
    datax = pd.merge(datax, x, on=['user_id', 'day', 'item_sales_level'], how='left')

    a = ['user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level']
    b = ['item_brand_id', 'item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level']
    for i in a:
        for j in b:
            x = datax.groupby([i, 'day', j]).size().reset_index().rename(columns={0: i + '_day_' + j + '_sum'})
            datax = pd.merge(datax, x, on=[i, 'day', j], how='left')


    ####################################    点击量 其他特征    ####################################

    # 用户每天对商店的点击量
    x = datax.groupby(['user_id', 'day', 'shop_id']).size().reset_index().rename(columns={0: 'user_day_shop_sum'})
    datax = pd.merge(datax, x, on=['user_id', 'day', 'shop_id'], how='left')

    # 用户每天点击的商品种类数
    x = datax.groupby(by=['user_id', 'day', 'item_id']).size().reset_index()
    x[0] = 1
    x = x.groupby(['user_id', 'day'])[0].sum().reset_index().rename(columns={0: 'user_day_item_kind'})
    datax = pd.merge(datax, x, on=['user_id', 'day'], how='left')

    # 用户每天点击的品牌种类数
    x = datax.groupby(by=['user_id', 'day', 'item_brand_id']).size().reset_index()
    x[0] = 1
    x = x.groupby(['user_id', 'day'])[0].sum().reset_index().rename(columns={0: 'user_day_brand_kind'})
    datax = pd.merge(datax, x, on=['user_id', 'day'], how='left')

    # 用户每天点击的价格种类数
    x = datax.groupby(by=['user_id', 'day', 'item_price_level']).size().reset_index()
    x[0] = 1
    x = x.groupby(['user_id', 'day'])[0].sum().reset_index().rename(columns={0: 'user_day_price_kind'})
    datax = pd.merge(datax, x, on=['user_id', 'day'], how='left')

    x = datax.groupby(['user_id', 'day', 'predict_category_property']).size().reset_index()
    x[0] = 1
    x = x.groupby(['user_id', 'day'])[0].sum().reset_index().rename(columns={0: 'user_day_que_sum'})
    datax = pd.merge(datax, x, on=['user_id', 'day'], how='left')

    #datax['item_property_list'] = datax['item_property_list'].map(lambda x: x.split(';'))
    #datax['predict_category_property'] = datax['predict_category_property'].map(lambda x: x.split(';'))
    #datax['pro_pre_rate'] = get_score(datax['item_property_list'], datax['predict_category_property'])
    #print(datax['pro_pre_rate'])


    ####################################    时间差     ####################################

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

    # 用户每天距离点击同一个类目第一次点击的时间差
    x = datax.groupby(['user_id', 'day', 'cate_1'])['context_timestamp'].min().reset_index().rename(
        columns={'context_timestamp': 'same_cate_min_time'})
    datax = pd.merge(datax, x, on=['user_id', 'day', 'cate_1'], how='left')
    datax['same_cate_first_time_cha'] = (datax['context_timestamp'] - datax['same_cate_min_time']).map(lambda x: x.seconds)

    # 用户每天距离点击同一家商店第一次的时间差
    x = datax.groupby(['user_id', 'day', 'shop_id'])['context_timestamp'].min().reset_index().rename(
        columns={'context_timestamp': 'user_shop_day_min_time'})
    datax = pd.merge(datax, x, on=['user_id', 'day', 'shop_id'], how='left')
    datax['same_shop_first_time_cha'] = (datax['context_timestamp'] - datax['user_shop_day_min_time']).map(
        lambda x: x.seconds)

    # 用户每天距离点击同一家商店最后一次的时间差
    x = datax.groupby(['user_id', 'day', 'shop_id'])['context_timestamp'].max().reset_index().rename(
        columns={'context_timestamp': 'user_shop_day_max_time'})
    datax = pd.merge(datax, x, on=['user_id', 'day', 'shop_id'], how='left')
    datax['same_shop_last_time_cha'] = (datax['user_shop_day_max_time'] - datax['context_timestamp']).map(
        lambda x: x.seconds)

    # 用户每天浏览的总时长
    datax['user_day_time_duration'] = (datax['user_day_max_time'] - datax['user_day_min_time']).map(
        lambda x: x.seconds)



    # 用户每天此次点击距上一次点击，下一次点击的时间差
    print('耗时间的:')
    x = datax.sort_values(['user_id', 'day', 'context_timestamp'])
    x['former_time_cha'] = x.groupby(['user_id', 'day'])['context_timestamp'].diff(periods=1)
    x['later_time_cha'] = x.groupby(['user_id', 'day'])['context_timestamp'].diff(periods=-1)
    datax['former_time_cha'] = x['former_time_cha'].map(lambda x: x.seconds)
    datax['later_time_cha'] = x['later_time_cha'].map(lambda x: -x.seconds)
    datax['if_czxtdj'] = ( (datax['former_time_cha'] == 0) | (datax['later_time_cha'] == 0) ).map(lambda x: 1 if x else 0)





    ####################################    历史点击量，转化量，转化率     ####################################

    a = ['user_id' , 'user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level']
    for i in a:
        datax = get_cvr(datax, startday, endday, i)



    # 用户和商品的组合特征转化率
    a = ['user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level']
    b = ['cate_1', 'item_brand_id', 'item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level']
    for i in a:
        for j in b:
            datax = get_zuhe_cvr(datax, startday, endday, i, j)



    ####################################    trick特征     ####################################
    
    # 用户每天点击的第一条
    x = datax.groupby(['user_id', 'day'])['context_timestamp'].min().reset_index().rename(
        columns={'context_timestamp': 'user_day_first_look_time'})
    datax = pd.merge(datax, x, on=['user_id', 'day'], how='left')
    datax['if_user_day_first_look'] = (datax['context_timestamp'] == datax['user_day_first_look_time']).map(
        lambda x: 1 if x else 0)

    # 用户每天点击的最后一条，强特
    x = datax.groupby(['user_id', 'day'])['context_timestamp'].max().reset_index().rename(
        columns={'context_timestamp': 'user_day_last_look_time'})
    datax = pd.merge(datax, x, on=['user_id', 'day'], how='left')
    datax['if_user_day_last_look'] = (datax['context_timestamp'] == datax['user_day_last_look_time']).map(
        lambda x: 1 if x else 0)

    # 用户每天点击统一商品的第一条
    x = datax.groupby(['user_id', 'day', 'item_id'])['context_timestamp'].min().reset_index().rename(
        columns={'context_timestamp': 'min_time'})
    datax = pd.merge(datax, x, on=['user_id', 'day', 'item_id'], how='left')
    datax['if_first'] = (datax['context_timestamp'] == datax['min_time']).map(lambda x: 1 if x else 0)

    # 用户每天点击同一商品的最后一条
    x = datax.groupby(['user_id', 'day', 'item_id'])['context_timestamp'].max().reset_index().rename(
        columns={'context_timestamp': 'max_time'})
    datax = pd.merge(datax, x, on=['user_id', 'day', 'item_id'], how='left')
    datax['if_last'] = (datax['context_timestamp'] == datax['max_time']).map(lambda x: 1 if x else 0)


    return datax