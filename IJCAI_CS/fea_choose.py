import random
import xgboost as xgb
from sklearn.metrics import log_loss

def feature_select(fea, train, test):
    s = 0;
    feat = fea
    while (s < 10):
        print('---------------------第', s, '次-------------------')
        s += 1
        features = fea
        now_feature_2 = []
        check = 1
        le = len(features)
        m = le
        for i in range(le):
            n = random.randint(0, m - 1)
            if features[n] in now_feature_2:
                continue
            now_feature_2.append(features[n])

            xg = xgb.XGBClassifier()
            xg.fit( train[now_feature_2], train['is_trade'] )
            y_pre = xg.predict_proba( test[now_feature_2] )[:, 1]
            jj = log_loss( test['is_trade'], y_pre )

            if (jj < check) & (check - jj > 0.00003):
                print('目前特征长度为', len(now_feature_2), ' 目前帅气的logloss值是', jj, ' 成功加入第', i + 1, '个', '降低为', check - jj)
                check = jj
            else:
                now_feature_2.pop()
            m -= 1;
        for f in now_feature_2:
            if (f in feat):
                feat.remove(f)
        print('剩余特征长度:', len(feat))

    return [i for i in fea if i not in feat]