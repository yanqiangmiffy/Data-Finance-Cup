import pandas as pd

lgb = pd.read_csv('result/lgb_0.7210517164335803.csv')
xgb = pd.read_csv('result/xgb_0.716163731747427.csv')
cnn = pd.read_csv('result/w3000.6495996632995624_cnn.csv')

res = lgb.copy()
res['target'] = lgb['target'] * 0.2 + xgb['target'] * 0.8 + cnn['target'] * 0.1
# res=lgb*0.8+xgb*0.2   # 0.877785  xgb 原先
res.to_csv('result/en73.csv', index=None)
