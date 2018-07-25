import pandas as pd
import time
from collections import defaultdict
import matplotlib.pyplot as plt
import math
import numpy as np
import matplotlib.pyplot as plt


def get_heat_fea(datax):
    print('开始统计热度特征:')


    ####################################    用户    ####################################

    # 强特
    x = datax.groupby('user_id')['item_id'].nunique().reset_index()
    x.columns = ['user_id', 'user_item_kind']
    datax = pd.merge(datax, x, on='user_id', how='left')

    x = datax.groupby('user_id')['cate_1'].nunique().reset_index()
    x.columns = ['user_id', 'user_cate_kind']
    datax = pd.merge(datax, x, on='user_id', how='left')

    x = datax.groupby('user_id')['item_brand_id'].nunique().reset_index()
    x.columns = ['user_id', 'user_brand_kind']
    datax = pd.merge(datax, x, on='user_id', how='left')



    ####################################    商品    ####################################

    x = datax.groupby('item_id')['user_id'].nunique().reset_index()
    x.columns = ['item_id', 'item_user_kind']
    datax = pd.merge(datax, x, on='item_id', how='left')



    ####################################    商店    ####################################

    x = datax.groupby('shop_id')['user_id'].nunique().reset_index()
    x.columns = ['shop_id', 'shop_user_kind']
    datax = pd.merge(datax, x, on='shop_id', how='left')




    return datax

