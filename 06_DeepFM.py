import os
import time
import numpy as np
import pandas as pd
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Reshape, Lambda, concatenate, dot, add
from keras.layers import Dropout, GaussianDropout, multiply, SpatialDropout1D, BatchNormalization
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.optimizers import Adam
from sklearn.utils import shuffle
from tqdm import tqdm
# https://github.com/xxxmin/ctr_Keras/blob/master/preprocess.py
# file_path="pnn3_10fold.h5"
batch_size = 64
epochs = 5

# 设置随机种子
SEED = 2019
np.random.seed(SEED)

train = pd.read_csv("new_data/train.csv")
train_target = pd.read_csv('new_data/train_target.csv')
train = train.merge(train_target, on='id')
train = shuffle(train, random_state=2019)  # 打乱数据集
test = pd.read_csv("new_data/test.csv")
train['id'] = [i for i in range(len(train))]
test['target'] = [-1 for i in range(len(test))]
y_train = train['target'].astype(int).values

no_feas = ['id', 'target'] + ['certValidStop', 'certValidBegin']
train['certPeriod'] = train['certValidStop'] - train['certValidBegin']
test['certPeriod'] = test['certValidStop'] - test['certValidBegin']
numer_columns = ['certValidStop', 'certValidBegin', 'lmt', 'age', 'certPeriod']
cat_columns = [fea for fea in train.columns if fea not in numer_columns + no_feas]

for col in numer_columns:
    train_test = pd.concat([train[col], test[col]])
    train_test = pd.cut(train_test.values, 5, labels=range(5))
    train[col] = train_test[:train.shape[0]]
    test[col] = train_test[train.shape[0]:]
for col in cat_columns:
    train_test = pd.concat([train[col], test[col]])
    train_test = pd.factorize(train_test)[0]
    train[col] = train_test[:train.shape[0]]
    test[col] = train_test[train.shape[0]:]

print(train)


# ----------------------------------model-------------
def base_model(cat_columns, train, test):
    cat_num = len(cat_columns)
    field_cnt = cat_num
    cat_field_input = []
    field_embedding = []
    lr_embedding = []
    for cat in tqdm(cat_columns):
        input = Input(shape=(1,))
        cat_field_input.append(input)
        nums = pd.concat((train[cat], test[cat])).nunique() + 1
        embed = Embedding(nums, 1, input_length=1, trainable=True)(input)
        reshape = Reshape((1,))(embed)
        lr_embedding.append(reshape)
        # ffm embeddings
        field = []
        for i in range(field_cnt):
            embed = Embedding(nums, 10, input_length=1, trainable=True)(input)
            reshape = Reshape((10,))(embed)
            field.append(reshape)
        field_embedding.append(field)
        # ffm embeddings
    #######ffm layer##########
    inner_product = []
    for i in tqdm(range(field_cnt)):
        for j in range(i + 1, field_cnt):
            tmp = multiply([field_embedding[i][j], field_embedding[j][i]])
            inner_product.append(tmp)
    embed_layer = concatenate(inner_product, axis=-1)
    #######dnn layer##########
    embed_layer = Dense(64)(embed_layer)
    embed_layer = BatchNormalization()(embed_layer)
    embed_layer = Activation('relu')(embed_layer)
    embed_layer = Dense(64, activation='relu')(embed_layer)
    embed_layer = BatchNormalization()(embed_layer)
    embed_layer = Activation('relu')(embed_layer)
    embed_layer = Dense(1)(embed_layer)
    ########linear layer##########
    lr_layer = add(lr_embedding + [embed_layer])
    preds = Activation('sigmoid')(lr_layer)
    opt = Adam(0.001)
    model = Model(inputs=cat_field_input, outputs=preds)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])
    return model


#################################################training########################################3
# checkpoint = ModelCheckpoint(file_path+str(i), save_weights_only=True, verbose=1, save_best_only=True)
cols = cat_columns + numer_columns
x_train = train[cols].values
x_test = test[cols].values
x_train = list(x_train.T)
x_test = list(x_test.T)
early = EarlyStopping(monitor="val_loss", patience=3)
callbacks_list = [early]  # early
model = base_model(cols, train, test)
model.summary()
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.3,
          verbose=1,
          callbacks=callbacks_list)
