# -*- coding: utf-8 -*-#
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss
import time
import datetime
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
from user_feature import user_fea_extract
from ad_feature import ad_fea_extract
from shop_feature import shop_fea_extract
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel
from ad_feature import get_cvr, get_zuhe_cvr


def tim(value):
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(value))

def data_process(datax):
    # 广告类目
    x = list(datax['item_category_list'].map(lambda x: x.split(';')).values)
    x = pd.DataFrame(x, columns=['cate_A', 'cate_B', 'cate_C'])
    x.fillna(value=-1, inplace=True)
    x['cate_A'] = x['cate_A'].astype('int64')
    x['cate_B'] = x['cate_B'].astype('int64')
    x['cate_C'] = x['cate_C'].astype('int64')
    datax = pd.concat([datax, x], axis=1)

    # 广告展示的时间
    datax['context_timestamp'] = pd.to_datetime(datax['context_timestamp'].map(tim))
    datax['context_timestamp'] = datax['context_timestamp'].map(lambda x: x + datetime.timedelta(days = 1)) #这里加了一天，方便统计
    datax['day'] = datax['context_timestamp'].map(lambda x: x.day)
    datax['hour'] = datax['context_timestamp'].map(lambda x: x.hour)
    datax['minute'] = datax['context_timestamp'].map(lambda x: x.minute)
    print(datax['context_timestamp'].min(), datax['context_timestamp'].max())

    #for i in range(0, 6):
    #    datax['item_property_list_%s' % i] = datax['item_property_list'].map(lambda x: x.split(';')[i]).astype('int64')
    #datax['len'] = datax['item_property_list'].map(lambda x: len( x.split(';') ) )
    #print(datax['len'].value_counts())

    return datax



def run_cvr(startday, endday):
    train = pd.read_csv('/home/ubuntu/tianchi/IJCAI_FS/train.txt', sep=' ')
    print('read train ok:', train.shape)
    test_a = pd.read_csv('/home/ubuntu/tianchi/IJCAI_FS/test_a.txt', sep=' ')
    print('read test ok:', test_a.shape)

    train = data_process(train)
    test_a = data_process(test_a)
    test_a['is_trade'] = -1

    datax = pd.concat([train, test_a], axis=0, ignore_index=True)  # 拼在一起统计
    datax['is_trade'] = (datax['is_trade'].fillna(value=-1)).astype(int)

    datax = datax[ datax['day'] != 7 ]  #丢掉第7天的数据
    print(datax.shape)

    print(datax['context_timestamp'].min(), datax['context_timestamp'].max())

    print("去重前:", datax.shape)
    datax.drop_duplicates(inplace=True)  # 重复的数据都是标记为1的数据，可能是购买多件
    datax.reset_index(drop=True, inplace=True)
    print("去重后:", datax.shape)


    # 商品的历史被点击量和转化率
    datax = get_cvr(datax, startday, endday, 'item_id')

    a = ['item_brand_id', 'item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level']
    for i in a:
        datax = get_cvr(datax, startday, endday, i)


    # 用户的历史点击量和转化率
    a = ['user_id', 'user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level']
    for i in a:
        datax = get_cvr(datax, startday, endday, i)

    # 用户和商品的组合特征转化率
    a = ['user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level']
    b = ['cate_B', 'item_brand_id', 'item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level']
    for i in a:
        for j in b:
            datax = get_zuhe_cvr(datax, startday, endday, i, j)


    # 商品的历史被点击量和转化率
    a = ['shop_id', 'shop_review_num_level', 'shop_review_positive_rate', 'shop_star_level', 'shop_score_service',
         'shop_score_delivery', 'shop_score_description']
    for i in a:
        datax = get_cvr(datax, startday, endday, i)


    print(datax.shape)
    datax.to_csv('/home/ubuntu/tianchi/IJCAI_FS/cvr.csv', index=False)
    print('ok')



run_cvr(8, 8) # 时间做个偏移，方便统计
