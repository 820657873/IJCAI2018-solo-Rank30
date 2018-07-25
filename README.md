# IJCAI2018-solo-Rank30

IJCAI2018 阿里妈妈搜索广告预测 单人单模型 大家看看就好
特征主要包括几个部分:

1、点击特征
不同时间粒度(全局、小时、30、15、10、5分钟）统计用户、商品、价格、销量、商店等点击量

2、热度特征
统计用户、商品、类别、品牌、商店的热度

3、时间差特征
用户点击的时间差，用户此次点击距离上次点击，下次点击的时间差，用户此次点击距离第一次，最后一次点击的时间差，用户的浏览时长，用户点击同一个商品的时间差等等

4、trick特征
这部分特征通过观察数据得来，但可以通过时间差特征学出这些特征。标记用户每天的第一次点击，最后一次点击，对同一件商品的第一次点击，最后一次点击等等

5、rank特征
排序特征，这部分特征在电商环境中比较常见。商品的排序特征，商店的排序特征等等