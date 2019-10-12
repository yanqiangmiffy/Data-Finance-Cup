# Data-Finance-Cup
2019厦门国际银行“数创金融杯”数据建模大赛

## 结果
- lgb 0.733
- xgb 0.72
- cnn_em_dim_100:0.742


------

新数据

------
- lgb 0.775
- xgb 0.776
- rcnn 0.74
- cnn 0.70
- xgb原始特征cv_gbdt： 0.7157745924652787
- lr基于xgb的特征cv_lr_trans： 0.6550243546006284
- lr基于xgb特征个原始特征cv_lr_trans_raw： 0.5163804384024511
## 相关资料

1. https://github.com/TingNie/CreditForecast/blob/master/code.ipynb
2. https://blog.csdn.net/xckkcxxck/article/details/84533862
3. https://blog.csdn.net/huakai16/article/details/84099033
4. https://tianchi.aliyun.com/notebook-ai/detail?postId=41822
5. https://www.cnblogs.com/wkang/p/9657032.html GBDT + LR 的结构
6. https://blog.csdn.net/dengxing1234/article/details/73739836 Xgboost+lr
7. https://blog.csdn.net/luoyexuge/article/details/85001859 从0到1开始训练一个bert语言模型

```
$ git fetch --all
$ git reset --hard origin/master 
$ git pull
```