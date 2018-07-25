# -*- coding: utf-8 -*-#
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss
import time
import datetime
from sklearn.metrics import classification_report
from collections import defaultdict
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt
import math
import numpy as np
import lightgbm as lgb
from sklearn.cross_validation import train_test_split
from click_fea import get_click_fea
from heat_fea import get_heat_fea
from time_fea import get_time_fea
from trick_fea import get_trick_fea
from Bayesi import BayesianSmoothing


def tim(value):
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(value))

def data_process(datax):
    '''
    # 广告类目
    begin = time.time()
    x = list(datax['item_category_list'].map(lambda x: x.split(';')).values)
    x = pd.DataFrame(x, columns=['cate_A', 'cate_B', 'cate_C'])
    x.fillna(value=-1, inplace=True)
    x['cate_A'] = x['cate_A'].astype('int64')
    x['cate_B'] = x['cate_B'].astype('int64')
    x['cate_C'] = x['cate_C'].astype('int64')
    datax = pd.concat([datax, x], axis=1)
    print((time.time() - begin)/60)

    # 广告展示的时间
    begin = time.time()
    datax['context_timestamp'] = pd.to_datetime(datax['context_timestamp'].map(tim))
    print((time.time() - begin) / 60)
    datax['context_timestamp'] = datax['context_timestamp'].map(lambda x: x + datetime.timedelta(days = 1)) #这里加了一天，方便统计
    print((time.time() - begin) / 60)
    datax['day'] = datax['context_timestamp'].map(lambda x: x.day)
    datax['hour'] = datax['context_timestamp'].map(lambda x: x.hour)
    datax['minute'] = datax['context_timestamp'].map(lambda x: x.minute)
    print(datax['context_timestamp'].min(), datax['context_timestamp'].max())
    print((time.time() - begin) / 60)
    '''

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
    print(datax[ ['context_timestamp', 'day', 'hour', 'minute'] ])


    return datax


def get_mean(his_data, now_data, key1, key2):

    dfcvr = his_data
    dfcvr = dfcvr.groupby(key1)[key2].mean().reset_index()
    dfcvr.columns = [key1, key1 + '_' + key2 + '_mean']

    res = pd.merge(now_data, dfcvr, on=key1, how='left')

    print(res.shape)
    return res

def get_cvr(his_data, now_data, key):

    dfcvr = his_data
    dfcvr = pd.get_dummies(dfcvr[ [key, 'is_trade'] ], columns=['is_trade'], prefix='label')
    dfcvr = dfcvr.groupby(key, as_index=False).sum()
    #print(dfcvr)

    dfcvr[key + '_sum'] = (dfcvr['label_0'] + dfcvr['label_1'])
    dfcvr[key + '_buy'] = dfcvr['label_1']
    dfcvr[key + '_cvr'] = (dfcvr['label_1']) / (dfcvr['label_0'] + dfcvr['label_1'])

    res = pd.merge(now_data, dfcvr[[key, key + '_sum', key + '_buy', key + '_cvr']], on=key, how='left')

    print(res.shape)

    return res

def get_zuhe_cvr(his_data, now_data, key1, key2):

    dfcvr = his_data
    dfcvr = pd.get_dummies(dfcvr[ [key1, key2, 'is_trade'] ], columns=['is_trade'], prefix='label')
    dfcvr = dfcvr.groupby([key1, key2], as_index=False).sum()

    dfcvr[key1 + '_' + key2 + '_sum'] = (dfcvr['label_0'] + dfcvr['label_1'])
    dfcvr[key1 + '_' + key2 + '_buy'] = dfcvr['label_1']
    dfcvr[key1 + '_' + key2 + '_cvr'] = (dfcvr['label_1']) / (dfcvr['label_0'] + dfcvr['label_1'])

    res = pd.merge(now_data, dfcvr[[key1, key2, key1 + '_' + key2 + '_sum',key1 + '_' + key2 + '_buy',
                                    key1 + '_' + key2 + '_cvr']], on=[key1,key2], how='left')
    print(res.shape)
    return res

def get_his(his_data, now_data):

    ####################################    历史转化率   ####################################

    # 商品的历史被点击量和转化率
    now_data = get_cvr(his_data, now_data, 'item_id')

    # 'item_collected_level',
    a = ['item_price_level', 'item_sales_level']
    for i in a:
        now_data = get_cvr(his_data, now_data, i)


    # 用户的历史点击量和转化率
    a = ['user_id']
    for i in a:
        now_data = get_cvr(his_data, now_data, i)

    # 用户和商品的组合特征转化率
    #a = ['user_id', 'user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level']
    #b = ['item_id', 'cate_B', 'item_brand_id', 'item_price_level', 'item_sales_level', 'item_collected_level',
    #     'item_pv_level']
    a = ['user_id']
    b = ['item_id', 'cate_1']
    for i in a:
        for j in b:
            now_data = get_zuhe_cvr(his_data, now_data, i, j)

    ####################################    均值特征   ####################################

    return now_data


def get_cvr_smooth(his_data, now_data, key):

    dfcvr = his_data
    dfcvr = pd.get_dummies(dfcvr[ [key, 'is_trade'] ], columns=['is_trade'], prefix='label')
    dfcvr = dfcvr.groupby(key, as_index=False).sum()
    #print(dfcvr)

    dfcvr[key + '_sum'] = (dfcvr['label_0'] + dfcvr['label_1'])
    dfcvr[key + '_buy'] = dfcvr['label_1']

    #hyper = BayesianSmoothing(1, 1)
    #hyper.update( dfcvr[key + '_sum'].values, dfcvr[key + '_buy'].values, 10000, 0.00000001 )
    #alpha = hyper.alpha
    #beta = hyper.beta
    alpha = 0.547384009831
    beta = 45.8073433366
    filn = alpha / (alpha + beta)
    print('alpha:', alpha, 'beta:', beta)

    dfcvr[key + '_cvr'] = (dfcvr['label_1'] + alpha) / (dfcvr['label_0'] + dfcvr['label_1'] + alpha + beta)

    res = pd.merge(now_data, dfcvr[[key, key + '_sum', key + '_buy', key + '_cvr']], on=key, how='left')
    res[key + '_cvr'].fillna(value=filn, inplace=True)

    print(res.shape)

    return res



def feature_extract(datax):

    datax['context_timestamp'] = pd.to_datetime(datax['context_timestamp'])

    print("去重前:",datax.shape)
    datax.drop_duplicates(inplace=True)  # 重复的数据都是标记为1的数据，可能是购买多件
    datax.reset_index(drop=True, inplace=True)
    print("去重后:",datax.shape)

    datax = get_click_fea(datax)
    datax = get_heat_fea(datax)
    datax = get_time_fea(datax)
    datax = get_trick_fea(datax)

    return datax




def get_data():

    begin = time.time()
    train = pd.read_csv('/home/ubuntu/tianchi/IJCAI_FS/train.txt', sep=' ')
    end = time.time()
    print((end - begin)/60)

    begin = time.time()
    test_a = pd.read_csv('/home/ubuntu/tianchi/IJCAI_FS/test_a.txt', sep=' ')
    end = time.time()
    print((end - begin) / 60)

    begin = time.time()
    test_b = pd.read_csv('/home/ubuntu/tianchi/IJCAI_FS/test_b.txt', sep=' ')
    end = time.time()
    print((end - begin) / 60)

    print(train.shape, test_a.shape, test_b.shape, 'shujuliang')

    train = data_process(train)
    test_a = data_process(test_a)
    test_b = data_process(test_b)
    test_a['is_trade'] = -1
    test_b['is_trade'] = -2

    all_data = pd.concat([train, test_a, test_b], axis=0, ignore_index=True)
    print(all_data.columns)

    his_data = all_data[ (all_data['day'] < 7) | (all_data['day'] == 31)  ]
    now_data = all_data[ all_data['day'] == 7 ]
    print(his_data.shape)
    print(now_data.shape)

    print('kaishi_train')
    his_data.to_csv('/home/ubuntu/tianchi/IJCAI_FS/his_data.csv', index=False)
    print('jieshu_train')

    print('kaishi_test')
    now_data.to_csv('/home/ubuntu/tianchi/IJCAI_FS/now_data.csv', index=False)
    print('jieshu_test')

    '''
    now_data = get_his(his_data, now_data)

    now_data = feature_extract(now_data)

    nofea = ['cate_A', 'context_timestamp', 'day', 'instance_id', 'is_trade', 'item_category_list',
             'item_property_list', 'predict_category_property', 'same_max_time', 'same_min_time',
             'user_day_max_time', 'user_day_min_time', 'user_id', 'user_shop_day_max_time',
             'user_shop_day_min_time', 'same_cate_min_time', 'user_day_first_look_time', 'user_day_last_look_time',
             'min_time', 'max_time', 'hour']

    fea = [i for i in now_data.columns if i not in nofea]

    print('训练维度:', len(fea))

    datax = now_data
    print('kaishi')
    #datax.to_csv('/home/ubuntu/tianchi/IJCAI_FS/only_7.csv', index=False)
    print('end')

    online = 0

    if online == 0:
        datax.fillna(value=-1, inplace=True)
        #datax.to_csv('/home/ubuntu/tianchi/IJCAI_FS/last.csv', index=False)

        datax = datax[ datax['is_trade'] != -1 ]
        print(datax.shape)


        x_train = datax[ datax['hour'] <= 7]
        x_test = datax[ datax['hour'] > 7 ]
        y_train = x_train['is_trade']
        y_test = x_test['is_trade']

        #x_train, x_test, y_train, y_test = train_test_split(datax, datax['is_trade'], test_size=0.30, random_state=2018)

        print(x_train.shape)
        print(x_test.shape)


        #xg = xgb.XGBClassifier(n_estimators=110, max_depth=4, learning_rate=0.1)
        xg = lgb.LGBMClassifier(num_leaves=120, n_estimators=10000, max_depth=5, objective='binary',
                                learning_rate=0.05, subsample=0.9, colsample_bytree=0.9)
        xg.fit(x_train[fea], y_train, eval_set=[(x_test[fea], y_test)], early_stopping_rounds=30, verbose=1)
        y_pre = xg.predict_proba(x_test[fea])[:, 1]
        y_pre_two = xg.predict(x_test[fea])
        #xgb.plot_importance(xg)
        #plt.show()
        print(log_loss(y_test, y_pre))
        print(classification_report(y_test, y_pre_two))

    else:
        datax.fillna(value=-1, inplace=True)
        # datax.to_csv('/home/ubuntu/tianchi/IJCAI/all_last.csv', index=False)

        print(datax.shape)

        x_train = datax[ datax['is_trade'] != -1 ]
        x_test = datax[ datax['is_trade'] == -1 ]

        y_train = x_train['is_trade']

        print(x_train.shape)
        print(x_test.shape)

        xg = lgb.LGBMClassifier(num_leaves=120, n_estimators=463, max_depth=5, objective='binary',
                                learning_rate=0.05, subsample=0.9, colsample_bytree=0.9)
        xg.fit(x_train[fea], y_train)
        y_pre = xg.predict_proba(x_test[fea])[:, 1]

        y_out = xg.predict(x_test[fea])
        print(pd.Series(y_out).value_counts())

        result = pd.DataFrame()
        result['instance_id'] = x_test['instance_id']
        result['predicted_score'] = y_pre

        print(result.shape)

        result.to_csv('/home/ubuntu/tianchi/IJCAI_FS/result.txt', index=False, sep=' ')
    '''


if __name__ == '__main__':

    begin = time.time()
    datax = pd.read_csv('/home/ubuntu/tianchi/IJCAI_FS/now_data.csv')
    end = time.time()
    print((end - begin) / 60)

    print(datax.shape)
    print(datax.columns)



    gailv = pd.read_csv('/home/ubuntu/tianchi/IJCAI_FS/the_first.txt', sep=' ')
    print(gailv.shape, 'gailv')
    datax = pd.merge(datax, gailv, on='instance_id', how='left')
    print(datax.shape, 'pingjiehou')



    ###
    begin = time.time()
    his_data = pd.read_csv('/home/ubuntu/tianchi/IJCAI_FS/his_data.csv')
    print(his_data.shape, 'lishishuju')
    print((time.time() - begin) / 60)


    datax = get_his(his_data, datax)


    datax['user_first_apper'] = datax['user_id'].isin(his_data['user_id'])
    datax['user_first_apper'] = datax['user_first_apper'].map(lambda x: 0 if x else 1)

    datax['item_first_apper'] = datax['item_id'].isin(his_data['item_id'])
    datax['item_first_apper'] = datax['item_first_apper'].map(lambda x: 0 if x else 1)

    datax = get_mean(his_data, datax, 'user_id', 'item_price_level')
    datax = get_mean(his_data, datax, 'user_id', 'item_sales_level')
    datax = get_mean(his_data, datax, 'user_id', 'item_collected_level')
    datax = get_mean(his_data, datax, 'user_id', 'item_pv_level')
    datax = get_mean(his_data, datax, 'user_id', 'shop_review_positive_rate')

    datax = get_mean(his_data, datax, 'item_id', 'user_gender_id')
    datax = get_mean(his_data, datax, 'item_id', 'user_age_level')
    datax = get_mean(his_data, datax, 'item_id', 'user_star_level')

    datax = get_mean(his_data, datax, 'item_brand_id', 'user_age_level')
    datax = get_mean(his_data, datax, 'item_brand_id', 'user_star_level')



    ###

    datax = feature_extract(datax)

    #######datax.to_csv('/home/ubuntu/tianchi/IJCAI_FS/xunlian.csv', index=False)



    nofea = ['cate_0', 'context_id', 'context_timestamp', 'day', 'instance_id', 'is_trade',
             'item_category_list',    'item_property_list', 'predict_category_property', 'user_id',
             'user_day_max_time', 'user_day_min_time', 'user_day_first_look_time', 'user_day_last_look_time',
             'min_time', 'max_time', 'user_shop_last_time', 'pre_1', 'property_0', 'property_1', 'property_2',
             'same_min_time', 'same_max_time']

    fea = [i for i in datax.columns if i not in nofea]

    print('训练维度:', len(fea))


    online = 1

    if online == 0:
        #datax.to_csv('/home/ubuntu/tianchi/IJCAI_FS/xunlian.csv', index=False)

        datax = datax[datax['is_trade'] >= 0]
        print(datax.shape)

        x_train = datax[datax['hour'] <= 7]
        x_test = datax[datax['hour'] > 7]
        y_train = x_train['is_trade']
        y_test = x_test['is_trade']



        print(x_train.shape)
        print(x_test.shape)


        xg = lgb.LGBMClassifier(num_leaves=120, n_estimators=10000, max_depth=5, objective='binary',
                                learning_rate=0.02, subsample=0.8, colsample_bytree=0.9)
        xg.fit(x_train[fea], y_train, eval_set=[(x_test[fea], y_test)], early_stopping_rounds=30, verbose=1)

        ff = fea
        im = xg.feature_importances_
        aa = {}
        for i, j in zip(ff, im):
            aa[i] = j
        print(sorted(aa.items(), key=lambda x: x[1], reverse=True))


        y_pre = xg.predict_proba(x_test[fea])[:, 1]
        y_pre_two = xg.predict(x_test[fea])


        print(log_loss(y_test, y_pre))
        print(classification_report(y_test, y_pre_two))

    else:
        # datax.to_csv('/home/ubuntu/tianchi/IJCAI/all_last.csv', index=False)

        print(datax.shape)

        x_train = datax[datax['is_trade'] >= 0]
        x_test = datax[datax['is_trade'] < 0]

        need = datax[ datax['is_trade'] == -2 ]['instance_id'].values
        print(len(need), 'need')

        y_train = x_train['is_trade']

        print(x_train.shape)
        print(x_test.shape)

        #xg = lgb.LGBMClassifier(num_leaves=120, n_estimators=1084, max_depth=5, objective='binary',
        #                        learning_rate=0.02, subsample=0.8, colsample_bytree=0.9)
        xg = xgb.XGBClassifier(max_depth=4, learning_rate=0.02, n_estimators=1397, colsample_bytree=0.6,
                               subsample=0.9)

        xg.fit(x_train[fea], y_train)
        y_pre = xg.predict_proba(x_test[fea])[:, 1]

        y_out = xg.predict(x_test[fea])
        print(pd.Series(y_out).value_counts())

        result = pd.DataFrame()
        result['instance_id'] = x_test['instance_id']
        result['predicted_score'] = y_pre
        result = result[result['instance_id'].isin(need)]

        print(result.shape)

        result.to_csv('/home/ubuntu/tianchi/IJCAI_FS/result.txt', index=False, sep=' ')

        print(result['predicted_score'].mean())



#get_data()
#x = pd.read_csv('/home/ubuntu/tianchi/IJCAI_FS/result.txt', sep=' ')
#print(x['predicted_score'].mean())


#x = pd.read_csv('/home/ubuntu/tianchi/IJCAI_FS/the_first.txt', sep=' ')
#print(x['instance_id'].value_counts())

'''
x = his_data[his_data['day'] == 5]
x = x[x['hour'] == 23]
y = x.groupby('minute').size().reset_index()
y.columns = ['minute', 'sum']
z = x.groupby('minute')['is_trade'].sum().reset_index()
z.columns = ['minute', 'gm']
y = pd.merge(y, z, on='minute', how='left')
y['cvr'] = y['gm'] / y['sum']
print(y)
plt.plot(y['minute'].values, y['cvr'].values, 'D-')
plt.xticks(np.arange(0, 60, 1))
plt.show()
'''