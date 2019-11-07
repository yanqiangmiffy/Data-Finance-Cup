import pandas as pd

lgb = pd.read_csv('result/0.7087926129573728.csv')
xgb = pd.read_csv('result/xgb.csv')
cnn = pd.read_csv('result/NN_EntityEmbed_10fold-sub.csv')

res = lgb.copy()
res['target'] = cnn['target'] * 0.1 + xgb['target'] * 0.9
# res=lgb*0.8+xgb*0.2   # 0.877785  xgb 原先
res.to_csv('result/en73.csv', index=None,float_format='%.6f')
