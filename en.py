import pandas as pd

lgb = pd.read_csv('result/lgb_0.7253317057417796.csv')
xgb = pd.read_csv('result/xgb_0.7262252999368136.csv')
# cnn = pd.read_csv('result/w3000.6495996632995624_cnn.csv')
rcnn = pd.read_csv('result/04_rcnn.csv')

res = lgb.copy()
res['target'] =  xgb['target'] * 0.8 + rcnn['target'] * 0.2
# res=lgb*0.8+xgb*0.2   # 0.877785  xgb 原先
res.to_csv('result/en73.csv', index=None)
