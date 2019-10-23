# Data-Finance-Cup
2019厦门国际银行“数创金融杯”数据建模大赛

## 特征工程：
certValidStop
bankCard
residentAddr
certValidBegin
certId
dist
### 重复列

- ['x_0', 'x_1', 'x_2', 'x_3', 'x_4', 'x_5', 'x_6', 'x_7', 'x_8', 'x_9', 'x_10', 'x_11', 'x_13', 'x_15', 'x_17', 'x_18', 'x_19', 'x_21', 'x_23', 'x_24', 'x_36', 'x_37', 'x_38', 'x_57', 'x_58', 'x_59', 'x_60', 'x_77', 'x_78']
- ['x_22', 'x_40', 'x_70']
- ['x_41', 'unpayNormalLoan']
- ['x_43', 'unpayIndvLoan']
- ['x_45', 'unpayOtherLoan']
- 
['x_61', 'ncloseCreditCard']
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
- lgb xgb gen_feas 0.78
- 10.entity_embedding 0.70
- 过采样没有效果
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

## Gini系数
```text
·GINI系数:也是用于模型风险区分能力进行评估。
GINI统计值衡量坏账户数在好账户数上的的累积分布与随机分布曲线之间的面积，好账户与坏账户分布之间的差异越大，GINI指标越高，表明模型的风险区分能力越强。

GINI系数的计算步骤如下：
1. 计算每个评分区间的好坏账户数。
2. 计算每个评分区间的累计好账户数占总好账户数比率（累计good%）和累计坏账户数占总坏账户数比率(累计bad%)。
3. 按照累计好账户占比和累计坏账户占比得出下图所示曲线ADC。
4. 计算出图中阴影部分面积，阴影面积占直角三角形ABC面积的百分比，即为GINI系数
————————————————
版权声明：本文为CSDN博主「小墨鱼~~」的原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/zwqjoy/article/details/84859405
```