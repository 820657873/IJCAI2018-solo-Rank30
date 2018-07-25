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

# ------------------------------------商店相关特征------------------------------------

def shop_fea_extract(datax, startday, endday):
    print('开始统计商店特征:')

    ####################################    点击量     ####################################

    a = ['shop_id', 'shop_review_num_level', 'shop_star_level']
    for i in a:
        x = datax.groupby([i, 'day']).size().reset_index().rename(columns={0: i + '_day_sum'})
        datax = pd.merge(datax, x, on=[i, 'day'], how='left')
        x = datax.groupby([i, 'day', 'hour']).size().reset_index().rename(columns={0: i + '_day_hour_sum'})
        datax = pd.merge(datax, x, on=[i, 'day', 'hour'], how='left')


    ####################################    历史点击量，转化量，转化率     ####################################

    a = ['shop_id', 'shop_review_num_level', 'shop_review_positive_rate', 'shop_star_level', 'shop_score_service',
         'shop_score_delivery', 'shop_score_description']
    for i in a:
        datax = get_cvr(datax, startday, endday, i)



    return datax