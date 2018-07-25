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
import matplotlib.pyplot as plt


def tim(value):
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(value))

def smooth(datax, endday, obj):
    datax.index = datax['context_timestamp']
    x = datax['2018-09-21':'2018-09-%s' % endday]
    I = x.groupby(obj)['is_trade'].size().reset_index()
    I.columns = [obj, 'I']
    C = x.groupby(obj)['is_trade'].sum().reset_index()
    C.columns = [obj, 'C']
    x = pd.concat([I, C['C']], axis=1)
    print('开始平滑:')
    hyper = BayesianSmoothing(1, 1)
    hyper.update(x['I'].values, x['C'].values, 1000, 0.00000001)
    print('平滑结束')
    alpha = hyper.alpha
    beta = hyper.beta
    #alpha = 1.69798377038  # 训练的贝叶斯参数-广告
    #beta = 93.7414148792  # 训练的贝叶斯参数3-广告
    # alpha = 1.71664839346  # 测试的贝叶斯参数-广告
    # beta = 96.0191282133  # 测试的贝叶斯参数-广告

    print('alpha:', alpha)
    print('beta:', beta)
    print('x', x.shape)
    x[obj + '_buy_smooth'] = (x['C'] + alpha) / (x['I'] + alpha + beta)

    filn = alpha / (alpha + beta)
    datax = pd.merge(datax, x, on=obj, how='left')
    datax[obj + '_buy_smooth'].fillna(value=filn, inplace=True)


    return datax

def get_score(pro, pre):
    res = []
    for (i, j) in zip( pro.values, pre.values ):
        s = set()
        s.update(i)
        s.discard(-1)
        if len(s) == 0:
            res.append(0)
            continue

        t = set()
        if j[0] == '-1':
            res.append(0)
            continue

        for k in j:
            hh = k.split(':')[1].split(',')
            t.update(hh)
        t.discard(-1)
        if len(t) == 0:
            res.append(0)
            continue

        res.append( len( s & t ) / len(t) )
    return res

def get_mean(datax, startday, endday, key1, key2):
    res = pd.DataFrame()
    for day in [1, 2, 7]:
        dfcvr = datax[(datax['day'] < day) | (datax['day'] == 31)]
        dfcvr = dfcvr.groupby(key1)[key2].mean().reset_index()
        dfcvr.columns = [key1, key1 + '_' + key2 + '_mean']

        sub_data = pd.merge(datax[datax['day'] == day], dfcvr, on=key1, how='left')
        res = pd.concat([res, sub_data], axis=0, ignore_index=True)
    x = datax[datax['day'] == 31]
    print(x.shape)
    print(res.shape)
    res = pd.concat([res, x], axis=0, ignore_index=True)
    print(res.shape)
    return res


def get_cvr(datax, startday, endday, key):
    res = pd.DataFrame()
    num = 0
    for day in [1, 2, 7]:
        num += 1
        dfcvr = datax[(datax['day'] < day) | (datax['day'] == 31)]
        print(dfcvr.shape, 'dfcvr')
        dfcvr = pd.get_dummies(dfcvr[ [key, 'is_trade'] ], columns=['is_trade'], prefix='label')
        dfcvr = dfcvr.groupby(key, as_index=False).sum()
        dfcvr[key + '_sum'] = (dfcvr['label_0'] + dfcvr['label_1']) / num
        dfcvr[key + '_buy'] = dfcvr['label_1'] / num
        dfcvr[key + '_cvr'] = (dfcvr['label_1']) / (dfcvr['label_0'] + dfcvr['label_1'])
        #print(dfcvr)
        sub_data = pd.merge(datax[datax['day'] == day], dfcvr[[key, key + '_sum', key + '_cvr']], on=key, how='left')
        res = pd.concat([res, sub_data], axis=0, ignore_index=True)
    x = datax[ datax['day'] == 31 ]
    res = pd.concat([res, x], axis=0, ignore_index=True)
    print(res.shape)
    return res

'''
def get_cvr_smooth(datax, startday, endday, key):
    res = pd.DataFrame()
    for day in np.arange(startday, endday+1):
        dfcvr = datax[datax['day'] < day]
        dfcvr = pd.get_dummies(dfcvr, columns=['is_trade'], prefix='label')
        dfcvr = dfcvr.groupby(key, as_index=False).sum()
        dfcvr[key + '_sum'] = (dfcvr['label_0'] + dfcvr['label_1'])

        hyper = BayesianSmoothing(1, 1)
        hyper.update( (dfcvr['label_0'] + dfcvr['label_1']).values, dfcvr['label_1'].values, 10000, 0.00000001)
        alpha = hyper.alpha
        beta = hyper.beta
        filn = alpha / (alpha + beta)
        print('day:', day, alpha, beta)


        dfcvr[key + '_cvr'] = (dfcvr['label_1'] + alpha) / (dfcvr['label_0'] + dfcvr['label_1'] + alpha + beta)
        #print(dfcvr)
        sub_data = pd.merge(datax[datax['day'] == day], dfcvr[[key, key + '_sum', key + '_cvr']], on=key, how='left')
        sub_data[key + '_cvr'].fillna(value=filn, inplace=True)
        res = pd.concat([res, sub_data], axis=0, ignore_index=True)
    x = datax[ datax['day'] < startday ]
    res = pd.concat([res, x], axis=0, ignore_index=True)
    print(res.shape)
    return res
'''

def get_cvr_smooth(smo, datax, startday, endday, key):
    res = pd.DataFrame()
    for day in np.arange(startday, endday+1):
        dfcvr = datax[datax['day'] < day]
        dfcvr = pd.get_dummies(dfcvr, columns=['is_trade'], prefix='label')
        dfcvr = dfcvr.groupby(key, as_index=False).sum()
        dfcvr[key + '_sum'] = (dfcvr['label_0'] + dfcvr['label_1']) / (day - 1)  # 变成日均历史点击量，特征量级归一化
        dfcvr[key + '_buy'] = dfcvr['label_1'] / (day - 1)

        hyper = smo[day]
        alpha = hyper[0]
        beta = hyper[1]
        filn = alpha / (alpha + beta)
        print('day:', day, alpha, beta)


        dfcvr[key + '_cvr'] = (dfcvr['label_1'] + alpha) / (dfcvr['label_0'] + dfcvr['label_1'] + alpha + beta)
        #print(dfcvr)
        sub_data = pd.merge(datax[datax['day'] == day], dfcvr[[key, key + '_sum', key + '_cvr']], on=key, how='left')
        sub_data[key + '_cvr'].fillna(value=filn, inplace=True)
        res = pd.concat([res, sub_data], axis=0, ignore_index=True)
    x = datax[ datax['day'] < startday ]
    res = pd.concat([res, x], axis=0, ignore_index=True)
    print(res.shape)
    return res

def get_zuhe_cvr(datax, startday, endday, key1, key2):
    res = pd.DataFrame()
    num = 0
    for day in [1, 2, 7]:
        num += 1
        dfcvr = datax[(datax['day'] < day) | (datax['day'] == 31)]
        print(dfcvr.shape, 'dfcvr')
        dfcvr = pd.get_dummies(dfcvr[ [key1, key2, 'is_trade'] ], columns=['is_trade'], prefix='label')
        dfcvr = dfcvr.groupby([key1, key2], as_index=False).sum()
        dfcvr[key1 + '_' + key2 + '_sum'] = (dfcvr['label_0'] + dfcvr['label_1']) / num
        dfcvr[key1 + '_' + key2 + '_buy'] = dfcvr['label_1'] / num
        dfcvr[key1 + '_' + key2 + '_cvr'] = (dfcvr['label_1']) / (dfcvr['label_0'] + dfcvr['label_1'])
        sub_data = pd.merge(datax[datax['day'] == day], dfcvr[[key1, key2, key1 + '_' + key2 + '_sum',
                                                               key1 + '_' + key2 + '_cvr']], on=[key1,key2], how='left')
        res = pd.concat([res, sub_data], axis=0, ignore_index=True)
    x = datax[datax['day'] == 31]
    res = pd.concat([res, x], axis=0, ignore_index=True)
    print(res.shape)
    return res


def ad_fea_extract(datax, startday, endday):
    print('开始统计广告特征:')

    ####################################    每天的点击量    ####################################

    a = ['item_id', 'item_brand_id', 'item_city_id', 'item_price_level', 'item_sales_level', 'item_collected_level',
         'item_pv_level']
    for i in a:
        x = datax.groupby([i, 'day']).size().reset_index().rename(columns={0: i + '_day_sum'})
        datax = pd.merge(datax, x, on=[i, 'day'], how='left')
        x = datax.groupby([i, 'day', 'hour']).size().reset_index().rename(columns={0: i + '_day_hour_sum'})
        datax = pd.merge(datax, x, on=[i, 'day', 'hour'], how='left')



    ####################################    历史点击量，转化量，转化率    ####################################
    '''
    smo = {  19 : [1.95940572653, 100.451129071],
             20 : [1.77857883324, 93.2650068046],
             21 : [1.72837335685, 93.4783071989],
             22 : [1.74860144691, 94.809709777],
             23 : [1.69324834815, 92.2935348426],
             24 : [1.69798377038, 93.7414148792],
             25 : [1.71664839346, 96.0191282133]
             }
    '''
    # 商品的历史被点击量和转化率
    datax = get_cvr(datax, startday, endday, 'item_id')

    a = ['item_brand_id', 'item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level']
    for i in a:
        datax = get_cvr(datax, startday, endday, i)


    # 时间的历史点击量和转化率
    datax = get_cvr(datax, startday, endday, 'hour')


    # 商品的年龄均值
    datax = get_mean(datax, startday, endday, 'item_id', 'user_age_level')


    return datax