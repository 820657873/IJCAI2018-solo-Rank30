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
from fea_choose import feature_select
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel


def tim(value):
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(value))

def data_process(datax):
    print(datax.shape, 'chuliqian')

    begin = time.time()
    for i in range(3):
        datax['cate_%d' % (i)] = datax['item_category_list'].apply(
            lambda x: x.split(";")[i] if len(x.split(";")) > i else -1)
        datax['cate_%d' % (i)] = datax['cate_%d' % (i)].astype('int64')
    print((time.time() - begin) / 60)

    for i in range(3):
        datax['pre_%d' % (i)] = datax['predict_category_property'].apply(
            lambda x: str(x.split(";")[i]).split(":")[0] if len(x.split(";")) > i else -1)
        datax['pre_%d' % (i)] = datax['pre_%d' % (i)].astype('int64')
    print((time.time() - begin) / 60)

    datax['context_timestamp'] = datax['context_timestamp'].map(tim)
    datax['day'] = datax['context_timestamp'].map(lambda x: int(x[8:10]))
    datax['hour'] = datax['context_timestamp'].map(lambda x: int(x[11:13]))
    datax['minute'] = datax['context_timestamp'].map(lambda x: int(x[14:16]))
    print((time.time() - begin) / 60)

    print(datax.shape, 'chulihou')


    return datax


def feature_extract(datax, startday, endday):

    datax['context_timestamp'] = pd.to_datetime(datax['context_timestamp'])

    print("去重前:",datax.shape)
    datax.drop_duplicates(inplace=True)  # 重复的数据都是标记为1的数据，可能是购买多件
    datax.reset_index(drop=True, inplace=True)
    print("去重后:",datax.shape)

    datax = ad_fea_extract(datax, startday, endday)
    datax = user_fea_extract(datax, startday, endday)
    datax = shop_fea_extract(datax, startday, endday)


    print("总特征维度:\n",datax.shape)
    print("总特征:\n",datax.columns)


    return datax



def start_train(datax, startday, endday, cv):
    datax = data_process(datax)

    datax = feature_extract(datax, startday, endday)

    x = pd.DataFrame()
    x['hhd'] = datax.columns
    x['hhd'].to_csv('/home/ubuntu/tianchi/IJCAI/columns.csv',index=False)


    nofea = ['cate_A', 'context_timestamp', 'day', 'instance_id', 'is_trade', 'item_category_list',
             'item_property_list', 'predict_category_property', 'same_max_time', 'same_min_time',
             'user_day_max_time', 'user_day_min_time', 'user_id', 'user_shop_day_max_time',
             'user_shop_day_min_time', 'same_cate_min_time', 'user_day_first_look_time', 'user_day_last_look_time',
             'min_time', 'max_time']

    fea = [i for i in datax.columns if i not in nofea]

    print("训练用维度:", len(fea))

    # 'if_user_day_last_look', 'if_user_day_first_look', 'if_last', 'if_first',


    print('start:')

    if cv == 0:
        datax.fillna(value=-1, inplace=True)
        #datax.to_csv('/home/ubuntu/tianchi/IJCAI/last.csv', index=False)

        x_train = datax[ (datax['day'] >= startday) & (datax['day'] < endday) ]
        x_test = datax[ datax['day'] == endday ]
        y_train = x_train['is_trade']
        y_test = x_test['is_trade']

        print(x_train.shape)
        print(x_test.shape)


        xg = xgb.XGBClassifier(n_estimators=110, max_depth=4, learning_rate=0.1)
        xg.fit(x_train[fea], y_train)
        y_pre = xg.predict_proba(x_test[fea])[:, 1]
        y_pre_two = xg.predict(x_test[fea])
        #xgb.plot_importance(xg)
        #plt.show()
        print(log_loss(y_test, y_pre))
        print(classification_report(y_test, y_pre_two))
    else:
        # cv
        x_train = datax[ (datax['day'] >= startday) & (datax['day'] <= endday) ]
        y_train = x_train['is_trade']

        xg = xgb.XGBClassifier()
        score = cross_val_score(xg, x_train[fea], y_train, cv=5, scoring='log_loss')
        print(score)
        print(score.mean())


def train_parm(startday, endday, cv):
    datax = pd.read_csv('/home/ubuntu/tianchi/IJCAI/last.csv')

    nofea = ['cate_A', 'context_timestamp', 'day', 'instance_id', 'is_trade', 'item_category_list',
             'item_property_list', 'predict_category_property', 'same_max_time', 'same_min_time',
             'user_day_max_time', 'user_day_min_time', 'user_id', 'user_shop_day_max_time',
             'user_shop_day_min_time', 'same_cate_min_time']
    fea = [i for i in datax.columns if i not in nofea]

    print("训练用维度:", len(fea))

    # 'if_user_day_last_look', 'if_user_day_first_look', 'if_last', 'if_first',


    print('start:')

    if cv == 0:
        datax.fillna(value=-1, inplace=True)

        x_train = datax[(datax['day'] >= startday) & (datax['day'] < endday)]
        x_test = datax[datax['day'] == endday]
        y_train = x_train['is_trade']
        y_test = x_test['is_trade']

        print(x_train.shape)
        print(x_test.shape)

        min_los = 1
        bl = 1


        xg = xgb.XGBClassifier(n_estimators=110, max_depth=4, learning_rate=0.1)
        xg.fit(x_train[fea], y_train)
        y_pre_one = xg.predict_proba(x_test[fea])[:, 1]
        print(log_loss(y_test, y_pre_one))

        lg = lgb.LGBMClassifier(num_leaves=40, n_estimators=97, max_depth=5)
        lg.fit(x_train[fea], y_train)
        y_pre_two = lg.predict_proba(x_test[fea])[:, 1]
        print(log_loss(y_test, y_pre_two))

        for i in np.arange(0.1, 1, 0.1):
            y_pre_all = y_pre_one * i + y_pre_two * (1-i)
            print(i, log_loss(y_test, y_pre_all))

    else:
        # cv
        x_train = datax[(datax['day'] >= startday) & (datax['day'] <= endday)]
        y_train = x_train['is_trade']

        xg = xgb.XGBClassifier()
        score = cross_val_score(xg, x_train[fea], y_train, cv=5, scoring='log_loss')
        print(score)
        print(score.mean())


def start_test_one(datax, datay, dataz, startday, endday):
    datax = data_process(datax)
    datay = data_process(datay)
    dataz = data_process(dataz)
    datay['is_trade'] = -1
    dataz['is_trade'] = -2


    datax = pd.concat([datax, datay, dataz], axis=0, ignore_index=True)  # 拼在一起统计
    datax['is_trade'] = ( datax['is_trade'].fillna(value=-1) ).astype(int)
    print(datax.shape)

    datax = datax[ (datax['day'] == 31) | (datax['day'] == 7) | (datax['day'] <= 2) ]
    print(datax.shape, 'tiquhou')
    datax = feature_extract(datax, startday, endday)

    datax = datax[ datax['day'] >= 1 ]

    #print('kaishicunchu')
    #datax.to_csv('/home/ubuntu/tianchi/IJCAI_FS/cunchu_two.csv', index=False)
    #print('cunchujieshu')

    #x_train = datax[ datax['is_trade'] != -1 ]
    #x_test = datax[ datax['is_trade'] == -1 ]
    x_train = datax[ datax['day'] != 7 ]
    x_test = datax[ datax['day'] == 7 ]
    print(x_train.shape, 'xunlian')
    print(x_test.shape, 'ceshi')


    y_train = x_train['is_trade']

    nofea = ['cate_0', 'context_id', 'context_timestamp', 'day', 'instance_id', 'is_trade', 'item_category_list',
             'item_property_list', 'predict_category_property', 'same_max_time', 'same_min_time',
             'user_day_max_time', 'user_day_min_time', 'user_id', 'user_shop_day_max_time',
             'user_shop_day_min_time', 'same_cate_min_time', 'user_day_first_look_time', 'user_day_last_look_time',
             'min_time', 'max_time']
    fea = [i for i in datax.columns if i not in nofea]

    print('训练特征维度:', len(fea))
    print('start:')


    print('ok')

    xg = lgb.LGBMClassifier(num_leaves=120, n_estimators=1304, max_depth=5, objective='binary',
                            learning_rate=0.02, colsample_bytree=0.7)
    xg.fit(x_train[fea], y_train)
    y_pre = xg.predict_proba(x_test[fea])[:, 1]

    y_out = xg.predict(x_test[fea])
    print(pd.Series(y_out).value_counts())

    result = pd.DataFrame()
    result['instance_id'] = x_test['instance_id']
    result['predicted_score'] = y_pre

    print(result.shape)

    result.to_csv('/home/ubuntu/tianchi/IJCAI_FS/the_first_two.txt', index=False, sep=' ')


def start_test(datax, datay, dataz, startday, endday):
    datax = data_process(datax)
    datay = data_process(datay)
    dataz = data_process(dataz)
    need = dataz['instance_id'].values

    datax = pd.concat([datax, datay, dataz], axis=0, ignore_index=True)  # 拼在一起统计
    datax['is_trade'] = ( datax['is_trade'].fillna(value=-1) ).astype(int)
    datax = feature_extract(datax, startday, endday)

    x_train = datax[ (datax['day'] >= startday) & (datax['day'] < endday) ]
    x_test = datax[ datax['day'] == endday ]


    y_train = x_train['is_trade']
    '''
    nofea = ['cate_A', 'context_timestamp', 'day', 'instance_id', 'is_trade', 'item_category_list',
             'item_property_list', 'predict_category_property', 'same_max_time', 'same_min_time',
             'user_day_max_time', 'user_day_min_time', 'user_id', 'user_shop_day_max_time',
             'user_shop_day_min_time', 'same_cate_min_time']
    '''
    nofea = ['cate_A', 'context_timestamp', 'day', 'instance_id', 'is_trade', 'item_category_list',
             'item_property_list', 'predict_category_property', 'same_max_time', 'same_min_time',
             'user_day_max_time', 'user_day_min_time', 'user_id', 'user_shop_day_max_time',
             'user_shop_day_min_time', 'same_cate_min_time', 'user_day_first_look_time', 'user_day_last_look_time',
             'min_time', 'max_time']
    fea = [i for i in datax.columns if i not in nofea]

    print('训练特征维度:', len(fea))
    print('start:')

    #datax.to_csv('/home/ubuntu/tianchi/IJCAI/all_last.csv', index=False)
    print('ok')

    xg = xgb.XGBClassifier()
    xg.fit(x_train[fea], y_train)
    y_pre = xg.predict_proba(x_test[fea])[:, 1]

    y_out = xg.predict(x_test[fea])
    print(pd.Series(y_out).value_counts())

    result = pd.DataFrame()
    result['instance_id'] = x_test['instance_id']
    result['predicted_score'] = y_pre
    result = result[ result['instance_id'].isin(need) ]

    print(result.shape)

    result.to_csv('/home/ubuntu/tianchi/IJCAI/result.txt', index=False, sep=' ')


def test_last(datax, datay, dataz, startday, endday):
    datax = pd.read_csv('/home/ubuntu/tianchi/IJCAI/all_last.csv')
    need = dataz['instance_id'].values


    x_train = datax[ (datax['day'] >= startday) & (datax['day'] < endday) ]
    x_test = datax[ datax['day'] == endday ]


    y_train = x_train['is_trade']

    nofea = ['cate_A', 'context_timestamp', 'day', 'instance_id', 'is_trade', 'item_category_list',
             'item_property_list', 'predict_category_property', 'same_max_time', 'same_min_time',
             'user_day_max_time', 'user_day_min_time', 'user_id', 'user_shop_day_max_time',
             'user_shop_day_min_time', 'same_cate_min_time']
    fea = [i for i in datax.columns if i not in nofea]

    print('训练特征维度:', len(fea))
    print('start:')


    xg = xgb.XGBClassifier(n_estimators=110, max_depth=4, learning_rate=0.1)
    xg.fit(x_train[fea], y_train)
    y_pre_one = xg.predict_proba(x_test[fea])[:, 1]
    print(len(y_pre_one))

    lg = lgb.LGBMClassifier(num_leaves=40, n_estimators=97, max_depth=5)
    lg.fit(x_train[fea], y_train)
    y_pre_two = lg.predict_proba(x_test[fea])[:, 1]
    print(len(y_pre_two))

    y_pre = y_pre_one * 0.5 + y_pre_two * 0.5
    print(len(y_pre))

    result = pd.DataFrame()
    result['instance_id'] = x_test['instance_id']
    result['predicted_score'] = y_pre
    result = result[ result['instance_id'].isin(need) ]

    print(result.shape)

    result.to_csv('/home/ubuntu/tianchi/IJCAI/result.txt', index=False, sep=' ')



def ronghe():
    a = pd.read_csv('/home/ubuntu/tianchi/IJCAI/8466.txt', sep=' ')
    b = pd.read_csv('/home/ubuntu/tianchi/IJCAI/baseline11.txt', sep=' ')
    print(a.shape)
    print(b.shape)

    result = pd.DataFrame()
    result['instance_id'] = a['instance_id']
    result['predicted_score'] = (a['predicted_score'] + b['predicted_score']) / 2

    print(result.shape)
    result.to_csv('/home/ubuntu/tianchi/IJCAI/result.txt', index=False, sep=' ')





train = pd.read_csv('/home/ubuntu/tianchi/IJCAI_FS/train.txt', sep=' ')
test_a = pd.read_csv('/home/ubuntu/tianchi/IJCAI_FS/test_a.txt', sep=' ')
test_b = pd.read_csv('/home/ubuntu/tianchi/IJCAI_FS/test_b.txt', sep=' ')

print(train.shape, test_a.shape, test_b.shape)

start_test_one(train, test_a, test_b, 0, 0)

def hh():

    begin = time.time()
    datax = pd.read_csv('/home/ubuntu/tianchi/IJCAI_FS/cunchu.csv')
    print('time:', (time.time() - begin) / 60)
    print('size:', datax.shape)

    x_train = datax[datax['day'] != 7]
    x_test = datax[datax['day'] == 7]
    print(x_train.shape, 'xunlian')
    print(x_test.shape, 'ceshi')

    y_train = x_train['is_trade']

    nofea = ['cate_0', 'context_id', 'context_timestamp', 'day', 'instance_id', 'is_trade', 'item_category_list',
             'item_property_list', 'predict_category_property', 'user_id', 'same_max_time', 'same_min_time',
             'user_day_max_time', 'user_day_min_time', 'user_id', 'user_shop_day_max_time',
             'user_shop_day_min_time', 'same_cate_min_time', 'user_day_first_look_time', 'user_day_last_look_time',
             'min_time', 'max_time', 'pre_1', 'property_0', 'property_1', 'property_2']
    fea = [i for i in datax.columns if i not in nofea]

    print('训练特征维度:', len(fea))
    print('start:')

    print('ok')

    xg = lgb.LGBMClassifier(num_leaves=120, n_estimators=1304, max_depth=5, objective='binary',
                            learning_rate=0.02, colsample_bytree=0.7)
    xg.fit(x_train[fea], y_train)
    y_pre = xg.predict_proba(x_test[fea])[:, 1]

    y_out = xg.predict(x_test[fea])
    print(pd.Series(y_out).value_counts())

    result = pd.DataFrame()
    result['instance_id'] = x_test['instance_id']
    result['predicted_score'] = y_pre

    print(result.shape)

    result.to_csv('/home/ubuntu/tianchi/IJCAI_FS/the_first.txt', index=False, sep=' ')

    return 0
