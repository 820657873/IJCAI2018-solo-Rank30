import time
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import log_loss
from sklearn import ensemble
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from scipy import sparse
import xgboost as xgb
import operator
import datetime
from BaysSmooth import BayesianSmoothing
from gensim.models import word2vec
import numpy as np
import lightgbm as lgb

extract_feature = ['is_last']


# 由于时间是偏移的这里，做一下时间的特征
def timestamp_datetime(value):
    format = '%Y-%m-%d %H:%M:%S'
    value = time.localtime(value)
    dt = time.strftime(format, value)
    return dt


# 数据预处理
def pre_process(data):
    for i in range(3):
        data['category_%d' % (i)] = data['item_category_list'].apply(
            lambda x: x.split(";")[i] if len(x.split(";")) > i else " ")
    del data['item_category_list']

    for i in range(3):
        data['property_%d' % (i)] = data['item_property_list'].apply(
            lambda x: x.split(";")[i] if len(x.split(";")) > i else " ")
    del data['item_property_list']

    for i in range(3):
        data['predict_category_%d' % (i)] = data['predict_category_property'].apply(
            lambda x: str(x.split(";")[i]).split(":")[0] if len(x.split(";")) > i else " ")

    # data['year'] = data['']
    data['time'] = data.context_timestamp.apply(timestamp_datetime)
    data['day'] = data.time.apply(lambda x: int(x[8:10]))
    data['hour'] = data.time.apply(lambda x: int(x[11:13]))

    return data


def extract_rate_feature(data, feat_rate, rate_name, train, test, fill_Value):
    feature_ = []
    feature_.extend(feat_rate)
    feature_.append(rate_name + '_count')
    feature_.append(rate_name + '_sum')
    feature_.append(rate_name + '_rate')
    extract_feature.append(rate_name + '_count')
    extract_feature.append(rate_name + '_sum')
    extract_feature.append(rate_name + '_rate')

    temp = data.groupby(feat_rate)['is_trade'].agg(
        {rate_name + '_count': 'count', rate_name + '_sum': 'sum'}).reset_index()
    temp[rate_name + '_rate'] = temp[rate_name + '_sum'] / temp[rate_name + '_count']
    train = pd.merge(train, temp[feature_], on=feat_rate, how='left')
    test = pd.merge(test, temp[feature_], on=feat_rate, how='left')
    train[rate_name + '_rate'] = train[rate_name + '_rate'].fillna(fill_Value)
    test[rate_name + '_rate'] = test[rate_name + '_rate'].fillna(fill_Value)
    train = train.fillna(0)
    test = test.fillna(0)
    return train, test


# 提取用户固有的特征
def extract_user_feature(data, train, test):
    # 提取用户的点击次数，购买次数，购买率
    train, test = extract_rate_feature(data, ['user_id'], 'user', train, test, 1)
    # 提取用户在不同时间的点击次数， 购买次数，购买率
    train, test = extract_rate_feature(data, ['user_id', 'hour'], 'user_hour', train, test, 1)
    # 提取用户的性别的点击次数，购买次数，购买率
    train, test = extract_rate_feature(data, ['user_gender_id'], 'gender', train, test, 1)
    # 提取性别在不同时间的点击次数，购买次数，购买率
    train, test = extract_rate_feature(data, ['user_gender_id', 'hour'], 'gender_hour', train, test, 1)
    # 提取年龄的点击次数，点击购买次数，购买率
    train, test = extract_rate_feature(data, ['user_age_level'], 'age', train, test, 1)
    # 提取年龄在不同时间的点击数，点击购买次数，购买率
    train, test = extract_rate_feature(data, ['user_age_level', 'hour'], 'age_hour', train, test, 1)
    # 提取职业的点击次数，点击购买次数，购买率
    train, test = extract_rate_feature(data, ['user_occupation_id'], 'occupation', train, test, 1)
    # 提取职业在不同时间的点击数，点击购买次数，购买率
    train, test = extract_rate_feature(data, ['user_occupation_id', 'hour'], 'occupation_hour', train, test, 1)
    # 提取用户星级的点击数，点击购买次数，购买率
    train, test = extract_rate_feature(data, ['user_star_level'], 'star', train, test, 1)
    # 提取用户星级在不同时间段的点击数，点击购买次数，购买率
    train, test = extract_rate_feature(data, ['user_star_level', 'hour'], 'star_hour', train, test, 1)
    return train, test


# 提取商品固有的特征
def extract_item_feature(data, train, test):
    # 提取商品的被点击的次数，购买次数，购买率
    train, test = extract_rate_feature(data, ['item_id'], 'item', train, test, 1)
    # 提取商品在不同时间段的被点击的次数，购买次数，购买率
    train, test = extract_rate_feature(data, ['item_id', 'hour'], 'item_hour', train, test, 1)
    # 提取不同品牌被点击的次数，购买次数，购买率
    train, test = extract_rate_feature(data, ['item_brand_id'], 'brand', train, test, 1)
    # 提取不同品牌在不同时间段被点击的次数，购买的次数，购买率
    train, test = extract_rate_feature(data, ['item_brand_id', 'hour'], 'brand_hour', train, test, 1)
    # 提取不同品牌在不同价位的点击次数，购买次数，购买率
    train, test = extract_rate_feature(data, ['item_brand_id', 'item_price_level'], 'brand_price', train, test, 1)
    # 提取不同品牌在不同的城市的点击次数，购买次数，购买率
    train, test = extract_rate_feature(data, ['item_brand_id', 'item_city_id'], 'brand_city', train, test, 1)
    # 提取不同价位的点击次数，购买次数，购买率
    train, test = extract_rate_feature(data, ['item_price_level'], 'price', train, test, 1)
    # 提取不同商品在不同城市的点击次数，购买次数，购买率
    train, test = extract_rate_feature(data, ['item_id', 'item_city_id'], 'item_city', train, test, 1)
    # 提取不同商品在不同城市不同时间段被点击的次数，购买次数，购买率
    train, test = extract_rate_feature(data, ['item_id', 'item_city_id', 'hour'], 'item_city_hour', train, test, 1)
    # 提取商品收藏等级的点击次数，购买次数，购买率
    train, test = extract_rate_feature(data, ['item_collected_level'], 'collected', train, test, 1)
    # 提取不同品牌被收藏的点击次数，购买次数，购买率
    train, test = extract_rate_feature(data, ['item_collected_level', 'item_brand_id'], 'collected_brand', train, test,
                                       1)
    # 提取商品销售的点击次数，购买次数，购买率
    train, test = extract_rate_feature(data, ['item_sales_level'], 'sales', train, test, 1)
    # 提取商铺展示次数与购买率之间的关系
    # train, test = extract_rate_feature(data, ['item_pv_level'], 'pv', train, test, 1)
    return train, test


# 提取商铺特有的特征
def extract_shop_feature(data, train, test):
    # 提取商铺的点击次数，购买次数，购买率
    # train, test = extract_rate_feature(data, ['shop_id'], 'shop', train, test, 1)
    # 提取商铺在不同时间段的点击次数，购买次数，购买率
    # train, test = extract_rate_feature(data, ['shop_id', 'hour'], 'shop_hour', train, test, 1)
    # 提取商铺星级的点击次数，购买次数，购买率
    # train, test = extract_rate_feature(data, ['shop_star_level'], 'shop_star', train, test, 1)
    # 提取商铺评价数量的点击次数，购买次数，购买率
    train, test = extract_rate_feature(data, ['shop_review_num_level'], 'review', train, test, 1)
    return train, test


# 提取用户和商品之间的特征
def extract_User_Item_Feature(data, train, test):
    # 用统计量来表示职业与商品类目（类目一）之间的关系
    train, test = extract_rate_feature(data, ['user_occupation_id', 'category_1'], 'occup_cate', train, test, 1)
    # 用统计量来表示职业与商铺品牌之间的关系
    train, test = extract_rate_feature(data, ['user_occupation_id', 'item_brand_id'], 'occup_brand', train, test, 1)
    # 职业和商品城市之间的关系
    train, test = extract_rate_feature(data, ['user_occupation_id', 'item_city_id'], 'occup_city', train, test, 1)
    # 提取职业和价格之间的关系
    train, test = extract_rate_feature(data, ['user_occupation_id', 'item_price_level'], 'occup_price', train, test, 1)
    # 提取职业和收藏之间的关系
    train, test = extract_rate_feature(data, ['user_occupation_id', 'item_collected_level'], 'occup_collected', train,
                                       test, 1)
    # 考虑用户年龄跟商商品的类目可能有关系
    train, test = extract_rate_feature(data, ['user_age_level', 'category_1'], 'age_cate', train, test, 1)
    # 考虑用户年龄和商铺品牌的关系
    train, test = extract_rate_feature(data, ['user_age_level', 'item_brand_id'], 'age_brand', train, test, 1)
    # 提取用户年龄和价格之间的关系
    train, test = extract_rate_feature(data, ['user_age_level', 'item_price_level'], 'age_price', train, test, 1)
    # 提取用户年龄和城市之间的关系
    train, test = extract_rate_feature(data, ['user_age_level', 'item_city_id'], 'age_city', train, test, 1)
    # 提取用户性别和商品之间的关系
    # train, test = extract_rate_feature(data, ['user_gender_id', 'item_id'], 'gender_item', train, test, 1)
    # train, test = extract_rate_feature(data, ['user_gender_id', 'item_brand_id'], 'gender_brand', train, test, 1)
    # train, test = extract_rate_feature(data, ['user_gender_id', 'item_city_id'], 'gender_city', train, test, 1)
    # 提取用户和商品之间的点击次数，购买次数，购买率
    train, test = extract_rate_feature(data, ['user_id', 'item_id'], 'user_item', train, test, 1)
    # 提取用户和商品在不同时间段的点击次数，购买次数，购买率
    train, test = extract_rate_feature(data, ['user_id', 'item_id', 'hour'], 'user_item_hour', train, test, 1)
    # 提取用户和品牌之间的点击次数，购买次数，购买率
    train, test = extract_rate_feature(data, ['user_id', 'item_brand_id'], 'user_brand', train, test, 1)
    # 提取用户和价格之间的点击次数，购买次数，购买率
    train, test = extract_rate_feature(data, ['user_id', 'item_price_level'], 'user_price', train, test, 1)
    # 提取用户和城市之间的点击次数，购买次数，购买率
    train, test = extract_rate_feature(data, ['user_id', 'item_city_id'], 'user_city', train, test, 1)
    # 提取用户和pv之间的点击次数，购买次数，购买率
    train, test = extract_rate_feature(data, ['user_id', 'item_pv_level'], 'user_pv', train, test, 1)
    # 提取用户跟收藏等级之间的关系
    train, test = extract_rate_feature(data, ['user_id', 'item_collected_level'], 'user_collected', train, test, 1)
    # 提取用户跟类目一之间的关系
    train, test = extract_rate_feature(data, ['user_id', 'category_1'], 'user_category', train, test, 1)
    return train, test


# 提取用户和商铺之间的关系
def extrac_user_shop_feature(data, train, test):
    # 提取用户和商铺之间的关系
    train, test = extract_rate_feature(data, ['user_id', 'shop_id'], 'user_shop', train, test, 1)
    # 提取用户和星级之间的关系
    train, test = extract_rate_feature(data, ['user_id', 'shop_star_level'], 'user_star', train, test, 1)
    return train, test


# 提取上下文的特有特征
def extract_context_feature(data, train, test):
    # 提取不同页面的关系
    train, test = extract_rate_feature(data, ['context_page_id'], 'page', train, test, 1)
    return train, test


def extract_today_count_feature(data, feature1, target, name):
    temp = data.groupby(feature1)[target].agg({name: 'count'}).reset_index()
    data = pd.merge(data, temp, on=feature1, how='left')
    return data


def extract_today_feature(data):
    # 把自己特征按照时间排序，获得排序特征
    data['today_time_rank'] = data['time'].groupby(data['user_id']).rank(ascending=True)
    # 提取这个商品是不是该用户最后一次点击
    x = data.groupby(['user_id', 'item_id'])['time'].max().reset_index().rename(columns={'time': 'max_time'})
    data = pd.merge(data, x, on=['user_id', 'item_id'], how='left')
    data['last_click'] = (data['time'] == data['max_time']).map(lambda x: 1 if x else 0)
    # 获取这个商品是不是该用户第一次点击，因为一般第一次都是不会去购买
    x1 = data.groupby(['user_id', 'item_id'])['time'].min().reset_index().rename(columns={'time': 'min_time'})
    data = pd.merge(data, x1, on=['user_id', 'item_id'], how='left')
    data['first_click'] = (data['time'] == data['min_time']).map(lambda x: 1 if x else 0)
    # 考虑到有可能第一次就是最后一次，所以这里做一个相减
    data['last_click'] = data['last_click'] - data['first_click']
    # 这次点击是不是用户商铺的最后一次点击
    x2 = data.groupby(['user_id'])['time'].min().reset_index().rename(columns={'time': 'all_max_time'})
    data = pd.merge(data, x2, on=['user_id'], how='left')
    data['all_last_click'] = (data['time'] == data['all_max_time']).map(lambda x: 1 if x else 0)

    # 用户今天所有的点击次数
    data = extract_today_count_feature(data, ['user_id'], 'item_id', 'user_day_click_count')
    # 用户今天点击这个商品的次数
    data = extract_today_count_feature(data, ['user_id', 'item_id'], 'is_trade', 'user_i_day_click_count')
    # 用户今天点击这个品牌的次数
    data = extract_today_count_feature(data, ['user_id', 'item_brand_id'], 'is_trade', 'user_b_day_click_count')
    # 用户今天点击这个商铺的次数
    data = extract_today_count_feature(data, ['user_id', 'shop_id'], 'is_trade', 'user_s_day_click_count')
    # 这个商品今天被浏览的次数
    data = extract_today_count_feature(data, ['item_id'], 'is_trade', 'i_day_click_count')
    # 这个品牌今天被浏览的次数
    data = extract_today_count_feature(data, ['item_brand_id'], 'is_trade', 'b_day_click_count')
    # 这个商铺今天被浏览的次数
    data = extract_today_count_feature(data, ['shop_id'], 'is_trade', 's_day_click_count')

    return data


# 提取用户点击时间差的特征
def extract_user_click_time(data):
    user_timestamp = data[['user_id', 'time']].drop_duplicates()
    # 按时间排序
    user_timestamp['time_rank'] = user_timestamp['time'].groupby(user_timestamp['user_id']).rank(ascending=True)

    # 把时间排序以后，然后在进行前后挪动
    user_timestamp_1 = user_timestamp.copy()
    user_timestamp_2 = user_timestamp.copy()
    user_timestamp_1['time_rank'] = user_timestamp_1['time_rank'] - 1
    user_timestamp_2['time_rank'] = user_timestamp_2['time_rank'] + 1
    user_timeall = pd.merge(user_timestamp_1, user_timestamp_2, on=['user_id', 'time_rank'], how='left')
    user_timeall = pd.merge(user_timeall, user_timestamp, on=['user_id', 'time_rank'], how='left')
    user_timeall = user_timeall.fillna('1900-01-01 00:00:00')
    user_timeall['diff1'] = user_timeall['time_x'].map(
        lambda x: datetime.datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S')) - user_timeall['time'].map(
        lambda x: datetime.datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S'))
    user_timeall['days_1'] = user_timeall['diff1'].map(lambda x: x.days)
    user_timeall['second_1'] = user_timeall['diff1'].map(lambda x: x.seconds)
    user_timeall['diff2'] = user_timeall['time'].map(
        lambda x: datetime.datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S')) - user_timeall['time_y'].map(
        lambda x: datetime.datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S'))
    user_timeall['days_2'] = user_timeall['diff2'].map(lambda x: x.days)
    user_timeall['second_2'] = user_timeall['diff2'].map(lambda x: x.seconds)
    user_timeall = user_timeall[['user_id', 'time', 'days_1', 'days_2', 'second_1', 'second_2', 'time_rank']]
    data = pd.merge(data, user_timeall, on=['user_id', 'time'], how='left')
    return data


def compare_cate(data):
    if (data['category_1'] == data['predict_category_0']) or (data['category_1'] == data['predict_category_1']) or (
        data['category_1'] == data['predict_category_2']):
        return 1
    else:
        return 0


# 只统计今天的用户点击这个商品之间的一些特征
def extarct_user_before_click_feature(data):
    # 统计用户搜索之前搜索了这个类目几次, 用rank代表次数
    user_cate_rank = data[['user_id', 'category_1', 'time']].drop_duplicates()
    user_cate_rank['cate_time_rank'] = user_cate_rank.groupby(['user_id', 'category_1'])['time'].rank(ascending=True)
    data = pd.merge(data, user_cate_rank, on=['user_id', 'category_1', 'time'], how='left')
    user_cate_count = data[['user_id', 'category_1']]
    user_cate_count = user_cate_count.groupby(['user_id', 'category_1']).size().reset_index().rename(
        columns={0: 'cate_today_count'})
    data = pd.merge(data, user_cate_count, on=['user_id', 'category_1'], how='left')
    data['cate_count_rank_cha'] = data['cate_today_count'] - data['cate_time_rank']

    # 统计用户搜索这个商铺几次了，用rank代表次数
    user_shop_rank = data[['user_id', 'shop_id', 'time']].drop_duplicates()
    user_shop_rank['shop_time_rank'] = user_shop_rank.groupby(['user_id', 'shop_id'])['time'].rank(ascending=True)
    data = pd.merge(data, user_shop_rank, on=['user_id', 'shop_id', 'time'], how='left')
    data['shop_count_rank_cha'] = data['user_s_day_click_count'] - data['shop_time_rank']

    # 统计用户搜索这个商品几次了，用rank代表次数
    user_item_rank = data[['user_id', 'item_id', 'time']].drop_duplicates()
    user_item_rank['item_time_rank'] = user_item_rank.groupby(['user_id', 'item_id'])['time'].rank(ascending=True)
    data = pd.merge(data, user_item_rank, on=['user_id', 'item_id', 'time'], how='left')
    data['item_count_rank_cha'] = data['user_i_day_click_count'] - data['item_time_rank']

    # 统计用户搜索这个品牌几次了，用rank代表次数
    user_brand_rank = data[['user_id', 'item_brand_id', 'time']].drop_duplicates()
    user_brand_rank['brand_time_rank'] = user_brand_rank.groupby(['user_id', 'item_brand_id'])['time'].rank(
        ascending=True)
    data = pd.merge(data, user_brand_rank, on=['user_id', 'item_brand_id', 'time'], how='left')
    data['brand_count_rank_cha'] = data['user_b_day_click_count'] - data['brand_time_rank']

    # 统计这个人今天点击的商品在这个人今天点击商品的价格排名
    # user_item_price_rank = data[['user_id', 'item_id', 'item_price_level']].drop_duplicates()
    # user_item_price_rank['item_price_rank'] = user_item_price_rank.groupby(['user_id', 'item_id'])['item_price_level'].rank(ascending = True)
    # data = pd.merge(data, user_item_price_rank, on = ['user_id', 'item_id', 'item_price_level'], how = 'left')

    # 统计这个人今天点击的商品在这个人今天点击商品的销量排名
    # user_item_sales_rank = data[['user_id', 'item_id', 'item_sales_level']].drop_duplicates()
    # user_item_sales_rank['item_sales_rank'] = user_item_sales_rank.groupby(['user_id', 'item_id'])['item_sales_level'].rank(ascending = True)
    # data = pd.merge(data, user_item_sales_rank, on = ['user_id', 'item_id', 'item_sales_level'], how = 'left')

    # 统计这个人今天点击的店铺的星级排名
    # user_shop_star_rank = data[['user_id','shop_id', 'shop_star_level']].drop_duplicates()
    # user_shop_star_rank['shop_star_rank'] = user_shop_star_rank.groupby(['user_id', 'shop_id'])['shop_star_level'].rank(ascending = True)
    # data = pd.merge(data, user_shop_star_rank, on = ['user_id', 'shop_id', 'shop_star_level'], how = 'left')

    data['shop_count_user_click_rate'] = data['shop_time_rank'] / data['time_rank']
    data['cate_count_user_click_rate'] = data['cate_time_rank'] / data['time_rank']
    # data['item_count_user_click_rate'] = data['item_time_rank'] / data['time_rank']
    # data['brand_count_user_click_rate'] = data['brand_time_rank'] / data['time_rank']
    # data['cate_is_in_pre_cate'] = data.apply(compare_cate, axis =1)
    # print (data['cate_is_in_pre_cate'])
    return data


# 统计今天这个商品的类目，用户昨天针对这个商品的情况
def extract_yestoday_cate_buy(yes_data, today_data):
    yes_data_cate = yes_data[['user_id', 'category_1', 'is_trade']]
    yes_data_cate = yes_data_cate.groupby(['user_id', 'category_1'])['is_trade'].agg(
        {'yes_cate_count': 'count', 'yes_cate_sum': 'sum'}).reset_index()

    today_data = pd.merge(today_data, yes_data_cate[['user_id', 'category_1', 'yes_cate_count', 'yes_cate_sum']],
                          on=['user_id', 'category_1'], how='left')
    today_data.fillna(0)

    yes_data_shop = yes_data[['user_id', 'shop_id', 'is_trade']]
    yes_data_shop = yes_data_shop.groupby(['user_id', 'shop_id'])['is_trade'].agg(
        {'yes_shop_count': 'count', 'yes_shop_sum': 'sum'}).reset_index()

    today_data = pd.merge(today_data, yes_data_shop[['user_id', 'shop_id', 'yes_shop_count', 'yes_shop_sum']],
                          on=['user_id', 'shop_id'], how='left')
    today_data.fillna(0)

    return today_data


def extract_user_item_click_time(data):
    user_timestamp = data[['user_id', 'item_id', 'time']].drop_duplicates()
    # 按时间排序
    user_timestamp['item_time_rank'] = user_timestamp.groupby(['user_id', 'item_id'])['time'].rank(ascending=True)
    # 把时间排序以后，然后在进行前后挪动
    user_timestamp_1 = user_timestamp.copy()
    user_timestamp_2 = user_timestamp.copy()
    user_timestamp_1['item_time_rank'] = user_timestamp_1['item_time_rank'] - 1
    user_timestamp_2['item_time_rank'] = user_timestamp_2['item_time_rank'] + 1
    user_timeall = pd.merge(user_timestamp_1, user_timestamp_2, on=['user_id', 'item_id', 'item_time_rank'], how='left')
    user_timeall = pd.merge(user_timeall, user_timestamp, on=['user_id', 'item_id', 'item_time_rank'], how='left')
    user_timeall = user_timeall.fillna('1900-01-01 00:00:00')
    user_timeall['item_diff1'] = user_timeall['time_x'].map(
        lambda x: datetime.datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S')) - user_timeall['time'].map(
        lambda x: datetime.datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S'))
    user_timeall['item_days_1'] = user_timeall['item_diff1'].map(lambda x: x.days)
    user_timeall['item_second_1'] = user_timeall['item_diff1'].map(lambda x: x.seconds)
    user_timeall['item_diff2'] = user_timeall['time'].map(
        lambda x: datetime.datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S')) - user_timeall['time_y'].map(
        lambda x: datetime.datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S'))
    user_timeall['item_days_2'] = user_timeall['item_diff2'].map(lambda x: x.days)
    user_timeall['item_second_2'] = user_timeall['item_diff2'].map(lambda x: x.seconds)
    user_timeall = user_timeall[
        ['user_id', 'item_id', 'time', 'item_days_1', 'item_days_2', 'item_second_1', 'item_second_2',
         'item_time_rank']]
    data = pd.merge(data, user_timeall, on=['user_id', 'item_id', 'time'], how='left')
    return data


def extract_first_apper_feature(f_train, data):
    data['user_first_apper'] = data['user_id'].isin(f_train['user_id'])
    data['user_first_apper'] = data['user_first_apper'].map(lambda x: 0 if x else 1)

    data['item_first_apper'] = data['item_id'].isin(f_train['item_id'])
    data['item_first_apper'] = data['item_first_apper'].map(lambda x: 0 if x else 1)

    return data


def extract_mean_feature(all_data, data, feature, target, name):
    temp = all_data.groupby(feature)[target].agg({name: 'mean'}).reset_index()
    data = pd.merge(data, temp, on=feature, how='left')
    return data


# 统计平均值特征32
def extract_all_mean_feature(f_data, data):
    # 统计用户的平均水平
    data = extract_mean_feature(f_data, data, ['user_id'], 'item_price_level', 'user_price_mean')
    data = extract_mean_feature(f_data, data, ['user_id'], 'item_sales_level', 'user_sales_mean')
    data = extract_mean_feature(f_data, data, ['user_id'], 'item_collected_level', 'user_collectes_mean')
    data = extract_mean_feature(f_data, data, ['user_id'], 'item_pv_level', 'user_pv_mean')
    data = extract_mean_feature(f_data, data, ['user_id'], 'shop_review_positive_rate', 'user_review_mean')
    # data = extract_mean_feature(f_data[f_data.is_trade == 1], data, ['user_id'], 'item_price_level','user_buy_price_mean')
    # data = extract_mean_feature(f_data[f_data.is_trade == 1], data, ['user_id'], 'item_sales_level','user_buy_sales_mean')
    # data = extract_mean_feature(f_data[f_data.is_trade == 1], data, ['user_id'], 'item_collected_level','user_buy_collected_mean')

    # 统计平均年龄相关特征
    data = extract_mean_feature(f_data, data, ['item_id'], 'user_age_level', 'item_age_mean')
    data = extract_mean_feature(f_data, data, ['item_brand_id'], 'user_age_level', 'brand_age_mean')
    data = extract_mean_feature(f_data, data, ['category_1'], 'user_age_level', 'cate1_age_mean')
    data = extract_mean_feature(f_data, data, ['shop_id'], 'user_age_level', 'shop_age_mean')

    # 统计性别平均值
    data = extract_mean_feature(f_data, data, ['item_id'], 'user_gender_id', 'item_sex_mean')
    data = extract_mean_feature(f_data, data, ['item_brand_id'], 'user_gender_id', 'brand_sex_mean')
    data = extract_mean_feature(f_data, data, ['category_1'], 'user_gender_id', 'cate1_sex_mean')
    data = extract_mean_feature(f_data, data, ['shop_id'], 'user_gender_id', 'shop_sex_mean')

    # 针对平均值做差

    data['item_age_cha'] = data['user_age_level'] - data['item_age_mean']
    data['brand_age_cha'] = data['user_age_level'] - data['brand_age_mean']
    data['category_1_age_cha'] = data['user_age_level'] - data['cate1_age_mean']
    data['shop_age_cha'] = data['user_age_level'] - data['shop_age_mean']
    data['item_price_cha'] = data['item_price_level'] - data['user_price_mean']
    data['item_sales_cha'] = data['item_sales_level'] - data['user_sales_mean']
    data['item_collected_cha'] = data['item_collected_level'] - data['user_collectes_mean']
    data['item_pv_cha'] = data['item_pv_level'] - data['user_pv_mean']

    data['item_age_abs'] = data['item_age_cha'].abs()
    data['brand_age_abs'] = data['brand_age_cha'].abs()
    data['category_1_age_abs'] = data['category_1_age_cha'].abs()
    data['shop_age_abs'] = data['shop_age_cha'].abs()
    data['item_price_abs'] = data['item_price_cha'].abs()
    data['item_sales_abs'] = data['item_sales_cha'].abs()
    data['item_collected_abs'] = data['item_collected_cha'].abs()
    data['item_pv_abs'] = data['item_pv_cha'].abs()

    data.fillna(-100)

    return data


def getFeature(data):
    not_used_feature = ['category_0', 'category_2', 'user_id', 'index', 'instance_id', 'category_1',
                        'is_trade', 'context_timestamp', 'user_gender_id', 'user_occupation_id',
                        'date', 'predict_category_0', 'time',
                        'predict_category_1', 'predict_category_2',
                        'predict_category_property', 'property_0', 'property_1', 'property_2',
                        'max_time', 'min_time', 'all_max_time']

    id_features = ['context_page_id', 'item_id', 'item_brand_id',
                   'item_city_id', 'user_gender_id',
                   'shop_id', 'user_id', 'category_0', 'category_1', 'category_2',
                   'property_0', 'property_1', 'property_2', 'predict_category_0',
                   'predict_category_1', 'predict_category_2']
    predictors = [f for f in data.columns if f not in (not_used_feature)]

    # predictors.extend(id_features)
    # predictors = list(set(predictors))
    return predictors, id_features


def xgboost_model(train, test, features, target):
    params = {'booster': 'gbtree',
              'objective': 'binary:logistic',
              'eval_metric': 'logloss',
              'max_depth': 7,
              'lambda': 100,
              'subsample': 0.7,
              'colsample_bytree': 0.7,
              'eta': 0.01,
              'seed': 1024,
              'nthread': 12,
              'silent': 1,
              }

    dtrain = xgb.DMatrix(train[features], label=train[target])
    dtest = xgb.DMatrix(test[features])
    clf = xgb.train(params, dtrain, num_boost_round=600)
    test['predicted_score'] = clf.predict(dtest)
    # test.to_csv('test.csv',index = None)
    return test


def one_host(data, x_columns):
    for i in x_columns:
        data[str(i) + str(data[i])] = 1
    return data


def createOneHot(X_train, X_test):
    for i in ['user_gender_id', 'user_occupation_id']:
        tool_set = set(X_train[i])
        for tool in tool_set:
            X_train[str(i) + str(tool)] = 0
            X_test[str(i) + str(tool)] = 0

    X_train = X_train.apply(lambda x: one_host(x, ['user_gender_id', 'user_occupation_id']), axis=1)
    X_test = X_test.apply(lambda x: one_host(x, ['user_gender_id', 'user_occupation_id']), axis=1)
    return X_train, X_test


def lgb_model(train_on, test_on, train_off, test_off, features, target):
    params = {
        'learning_rate': 0.02,
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'sub_feature': 0.7,
        'num_leaves': 120,
        'min_data': 100,
        'min_hessian': 1,
        'verbose': -1,
        'metric': 'binary_logloss',
        'max_depth': 5,

    }

    lgb_train_off = lgb.Dataset(train_off[features], train_off['is_trade'])
    lgb_test_off = lgb.Dataset(test_off[features], test_off['is_trade'])
    lgb_train_on = lgb.Dataset(train_on[features], train_on['is_trade'])
    mid = lgb.train(params,
                    lgb_train_off,
                    valid_sets=lgb_test_off,
                    num_boost_round=100000,
                    early_stopping_rounds=30
                    )

    gbm = lgb.train(params,
                    lgb_train_on,
                    num_boost_round=mid.best_iteration
                    )

    test_on['predicted_score'] = gbm.predict(test_on[features])
    return test_on


def create_data(data, time):
    train = data[data.day == time]
    test = data[data.day == time]
    f_train = data[(data.day > (time - 10)) & (data.day < time)]
    yes_data = data[data.day == (time - 1)]
    train = extract_today_feature(train)
    # train = extract_yestoday_cate_buy(yes_data,train)
    # train = extarct_user_before_click_feature(train)
    train = extract_all_mean_feature(f_train, train)
    train = extract_first_apper_feature(f_train, train)
    train, test = extract_user_feature(f_train, train, test)
    train, test = extract_item_feature(f_train, train, test)
    train, test = extract_context_feature(f_train, train, test)
    train, test = extract_shop_feature(f_train, train, test)
    train, test = extrac_user_shop_feature(f_train, train, test)
    train, test = extract_User_Item_Feature(f_train, train, test)
    return train


def extract_max_feature(f_data):
    item_zhiye = f_data.groupby(['item_id', 'user_occupation_id'])['is_trade'].agg(
        {'user_occupa_count': 'count'}).reset_index()
    x = item_zhiye.groupby(['item_id'])['user_occupa_count'].max().reset_index().rename(
        columns={'user_occupation_id': 'max_occup'})


if __name__ == "__main__":
    print("程序开始")
    online = True  # 这里用来标记是 线下验证 还是 在线提交
    train = pd.read_csv('round1_ijcai_18_train_20180301.txt', sep=' ')
    test_a = pd.read_csv('round1_ijcai_18_test_a_20180301.txt', sep=' ')
    train.drop_duplicates(inplace=True)
    train = pre_process(train)
    test_a = pre_process(test_a)
    test_a['is_trade'] = -1
    all_data = pd.concat([train, test_a])

    print("特征抽取开始")
    train_online = None
    train_offline = None
    test_online = None
    test_offline = None

    temp_data = []
    all_data = all_data.reset_index()

    '''
    #tt, id_features = getFeature(all_data)
    lb = LabelEncoder()
    #for feat in id_features:
    all_data['category_1'] = lb.fit_transform(list(all_data['category_1']))
    '''
    begin = time.time()
    all_data = extract_user_click_time(all_data)
    end = time.time()
    print(end - begin)
    # all_data = extract_user_item_click_time(all_data)
    for i in [21, 22, 23, 24, 25]:
        begin = time.time()
        train_mid = create_data(all_data, i)
        temp_data.append(train_mid)
        end = time.time()
        print(end - begin)

    del all_data
    # 前三个作为线下训练集
    for i in range(3):
        train_offline = pd.concat([train_offline, temp_data[i]], axis=0)

    test_offline = temp_data[3]

    if online == True:
        # 前四个作为线上训练集
        for i in range(4):
            train_online = pd.concat([train_online, temp_data[i]], axis=0)
        # train_online = train_offline
        test_online = temp_data[4]
    else:
        test_online = test_offline
        train_online = train_offline

    # test_online.to_csv('test_online.csv', index = None)
    # test_offline.to_csv('test_offline.csv', index = None)
    # train_online.to_csv('train_online.csv', index = None)
    # train_offline.to_csv('train_offline.csv', index = None)
    print("特征抽取结束")

    test_online = test_online.fillna(-1)
    test_offline = test_offline.fillna(-1)
    train_online = train_online.fillna(-1)
    train_offline = train_offline.fillna(-1)

    begin = time.time()
    train_offline, test_offline = createOneHot(train_offline, test_offline)
    train_online, test_online = createOneHot(train_online, test_online)
    end = time.time()
    print(end - begin)

    # posit_sample = train_offline[train_offline.is_trade == 1]
    # train_offline = pd.concat([train_offline, posit_sample])
    features, tt = getFeature(train_offline)
    print(features)
    # test = xgboost_model(train_offline, test_offline, features, 'is_trade')
    test = lgb_model(train_online, test_online, train_offline, test_offline, features, 'is_trade')

    if online == False:
        print(log_loss(test['is_trade'], test['predicted_score']))
    else:
        test[['instance_id', 'predicted_score']].to_csv('baseline1.txt', index=False, sep=' ')  # 保存在线提交结果