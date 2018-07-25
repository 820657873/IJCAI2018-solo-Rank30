# -*- coding: utf-8 -*-#
from math import sqrt
import pandas as pd


def multipl(a, b):
    sumofab = 0.0
    for i in range(len(a)):
        temp = a[i] * b[i]
        sumofab += temp
    return sumofab


def corrcoef(x, y):
    n = len(x)
    # 求和
    sum1 = sum(x)
    sum2 = sum(y)
    # 求乘积之和
    sumofxy = multipl(x, y)
    # 求平方和
    sumofx2 = sum([pow(i, 2) for i in x])
    sumofy2 = sum([pow(j, 2) for j in y])
    num = sumofxy - (float(sum1) * float(sum2) / n)
    # 计算皮尔逊相关系数
    den = sqrt((sumofx2 - float(sum1 ** 2) / n) * (sumofy2 - float(sum2 ** 2) / n))
    return num / den



a = pd.read_csv('/home/ubuntu/tianchi/IJCAI_FS/13990.txt', sep=' ')
b = pd.read_csv('/home/ubuntu/tianchi/IJCAI_FS/result_a.txt', sep=' ')

print(a.shape, b.shape)

print(a['predicted_score'].mean())
print(b['predicted_score'].mean())

print(corrcoef(a['predicted_score'].values, b['predicted_score'].values))