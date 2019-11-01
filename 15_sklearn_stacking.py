# -*- coding: utf-8 -*-
from heamy.dataset import Dataset
from heamy.estimator import Regressor, Classifier
# ModelsPipeline：https://blog.csdn.net/qiqzhang/article/details/85477242 ； https://cloud.tencent.com/developer/article/1463294
from heamy.pipeline import ModelsPipeline
import pandas as pd
import xgboost as xgb
import datetime
from sklearn.metrics import roc_auc_score
# lightgbm安装：https://blog.csdn.net/weixin_41843918/article/details/85047492
# lgb样例：https://www.jianshu.com/p/c208cac3496f
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from pandas.core.frame import DataFrame
from gen_feas import load_data


def xgb_feature(X_train, y_train, X_test, y_test=None):
    other_params = {'learning_rate': 0.125, 'max_depth': 3}
    model = xgb.XGBClassifier(**other_params).fit(X_train, y_train)
    predict = model.predict_proba(X_test)[:,1]
    #minmin = min(predict)
    #maxmax = max(predict)
    #vfunc = np.vectorize(lambda x:(x-minmin)/(maxmax-minmin))
    #return vfunc(predict)
    return predict

def xgb_feature2(X_train, y_train, X_test, y_test=None):
    # , 'num_boost_round':12
    other_params = {'learning_rate': 0.1, 'max_depth': 3}
    model = xgb.XGBClassifier(**other_params).fit(X_train, y_train)
    predict = model.predict_proba(X_test)[:,1]
    #minmin = min(predict)
    #maxmax = max(predict)
    #vfunc = np.vectorize(lambda x:(x-minmin)/(maxmax-minmin))
    #return vfunc(predict)
    return predict

def xgb_feature3(X_train, y_train, X_test, y_test=None):
    # , 'num_boost_round':20
    other_params = {'learning_rate': 0.13, 'max_depth': 3}
    model = xgb.XGBClassifier(**other_params).fit(X_train, y_train)
    predict = model.predict_proba(X_test)[:,1]
    #minmin = min(predict)
    #maxmax = max(predict)
    #vfunc = np.vectorize(lambda x:(x-minmin)/(maxmax-minmin))
    #return vfunc(predict)
    return predict

def rf_model(X_train, y_train, X_test, y_test=None):
    # n_estimators = 100
    model = RandomForestClassifier(n_estimators=90, max_depth=4,random_state=10).fit(X_train,y_train)
    predict = model.predict_proba(X_test)[:,1]
    #minmin = min(predict)
    #maxmax = max(predict)
    #vfunc = np.vectorize(lambda x:(x-minmin)/(maxmax-minmin))
    #return vfunc(predict)
    return predict


def et_model(X_train, y_train, X_test, y_test=None):
    model = ExtraTreesClassifier(max_features = 'log2', n_estimators = 1000 , n_jobs = -1).fit(X_train,y_train)
    return model.predict_proba(X_test)[:,1]

def gbdt_model(X_train, y_train, X_test, y_test=None):
    # n_estimators = 700
    model = GradientBoostingClassifier(learning_rate = 0.02, max_features = 0.7, n_estimators = 100 , max_depth = 5).fit(X_train,y_train)
    predict = model.predict_proba(X_test)[:,1]
    #minmin = min(predict)
    #maxmax = max(predict)
    #vfunc = np.vectorize(lambda x:(x-minmin)/(maxmax-minmin))
    #return vfunc(predict)
    return predict


def logistic_model(X_train, y_train, X_test, y_test=None):
    model = LogisticRegression(penalty = 'l2').fit(X_train,y_train)
    return model.predict_proba(X_test)[:,1]

def lgb_feature(X_train, y_train, X_test, y_test=None):
    model = lgb.LGBMClassifier(boosting_type='gbdt',  min_data_in_leaf=5, max_bin=200, num_leaves=25, learning_rate=0.01).fit(X_train, y_train)
    predict = model.predict_proba(X_test)[:,1]
    #minmin = min(predict)
    #maxmax = max(predict)
    #vfunc = np.vectorize(lambda x:(x-minmin)/(maxmax-minmin))
    #return vfunc(predict)
    return predict


VAILD = False
if __name__ == '__main__':

    if VAILD == False:
        train, test, no_features, features = load_data()
        train_train = train[features].values
        y = train['target'].astype('int32')
        test_data = test[features].values
        print(train_train.shape)

        train_train_x = train_train
        test_test_x = test_data
        xgb_dataset = Dataset(X_train=train_train_x,y_train=train['target'],X_test=test_test_x,y_test=None,use_cache=False)
        #heamy
        print ("---------------------------------------------------------------------------------------)")
        print ("开始构建pipeline：ModelsPipeline(model_xgb,model_xgb2,model_xgb3,model_lgb,model_gbdt)")
        model_xgb = Regressor(dataset=xgb_dataset, estimator=xgb_feature,name='xgb',use_cache=False)
        model_xgb2 = Regressor(dataset=xgb_dataset, estimator=xgb_feature2,name='xgb2',use_cache=False)
        model_xgb3 = Regressor(dataset=xgb_dataset, estimator=xgb_feature3,name='xgb3',use_cache=False)
        model_gbdt = Regressor(dataset=xgb_dataset, estimator=gbdt_model,name='gbdt',use_cache=False)
        model_lgb = Regressor(dataset=xgb_dataset, estimator=lgb_feature,name='lgb',use_cache=False)
        model_rf = Regressor(dataset=xgb_dataset, estimator=rf_model,name='rf',use_cache=False)

        # pipeline = ModelsPipeline(model_xgb,model_xgb2,model_xgb3,model_lgb,model_gbdt, model_rf)
        pipeline = ModelsPipeline(model_xgb, model_xgb2, model_xgb3, model_lgb, model_rf)
        print ("---------------------------------------------------------------------------------------)")
        print ("开始训练pipeline：pipeline.stack(k=7, seed=111, add_diff=False, full_test=True)")
        stack_ds = pipeline.stack(k=7, seed=111, add_diff=False, full_test=True)
        # k = 7    model_xgb, model_xgb2, model_xgb3, model_lgb, model_rf :   AUC: 0.780043
        print ("stack_ds: ", stack_ds)
        print ("---------------------------------------------------------------------------------------)")
        print ("开始训练Regressor：Regressor(dataset=stack_ds, estimator=LinearRegression,parameters={'fit_intercept': False})")
        stacker = Regressor(dataset=stack_ds, estimator=LinearRegression,parameters={'fit_intercept': False})
        print ("---------------------------------------------------------------------------------------)")
        print ("开始预测：")
        predict_result = stacker.predict()

        id_list = test["id"].tolist()
        d ={ "id" : id_list, "target" : predict_result  }
        res = DataFrame(d)#将字典转换成为数据框
        print (">>>>", res)
        csv_file = 'result/res_stacking.csv'
        res.to_csv( csv_file,index=None )
