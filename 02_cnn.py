from keras.layers import *
from tensorflow import set_random_seed
import jieba
import multiprocessing
import pandas as pd
from gensim.models import Word2Vec
import numpy as np
import keras.backend as K
from keras.callbacks import Callback, ModelCheckpoint
from keras.models import Model
from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import *
import ipykernel
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle
from utils import *

# 设置随机种子
SEED = 2019
np.random.seed(SEED)
set_random_seed(SEED)

train = pd.read_csv("new_data/train.csv")
train_target = pd.read_csv('new_data/train_target.csv')
train = train.merge(train_target, on='id')
train = shuffle(train, random_state=SEED)  # 打乱数据集
test = pd.read_csv("new_data/test.csv")
train['id'] = [i for i in range(len(train))]
test['target'] = [-1 for i in range(len(test))]
# 特征列
df = pd.concat([train, test], sort=False)
df['certPeriod'] = df['certValidStop'] - df['certValidBegin']
no_fea = ['id', 'target', 'certValidStop', 'certValidBegin']
feas = [fea for fea in df.columns if fea not in no_fea]
print(len(feas))

# 参数
is_Train_w2v = False  # 是否重新训练词向量
EMBEDDING_DIM = 100  # 词向量维度
MAX_SEQUENCE_LENGTH = len(feas)  # 序列最大长度 len(feas)
W2V_FILE = 'data/w2v.txt'

if is_Train_w2v:

    print("正在将生成文本..")
    df['token_text'] = df.apply(lambda row: to_text(row, feas), axis=1)
    df.to_csv('tmp/df.csv', index=None)
    texts = df['token_text'].values.tolist()
    print(texts[0])
    train_w2v(texts, W2V_FILE, EMBEDDING_DIM)
else:
    print("加载已生成的向量...")
    df = pd.read_csv('tmp/df.csv')
    texts = df['token_text'].values.tolist()
    print(texts[0])

# 构建词汇表
tokenizer = Tokenizer(filters='|')
tokenizer.fit_on_texts(texts)
word_index = tokenizer.word_index
print(word_index)
print("词语数量个数：{}".format(len(word_index)))

# MAX_SEQUENCE_LENGTH = len(feas)
print("最大长度", MAX_SEQUENCE_LENGTH)
sequences = tokenizer.texts_to_sequences(texts)
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

# 类别编码
x_train = data[:len(train)]
x_test = data[len(train):]
print(x_train.shape)
print(x_train)
# y_train = to_categorical(train['target'].values)
y_train = train['target'].values
y_train = y_train.astype(np.int32)
print(y_train)


def create_text_cnn():
    #
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedding_layer = create_embedding(word_index, W2V_FILE, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH)
    embedding_sequences = embedding_layer(sequence_input)
    # conv1 = Conv1D(128, 5, activation='relu', padding='same')(embedding_sequences)
    # pool1 = MaxPool1D(3)(conv1)
    # conv2 = Conv1D(128, 5, activation='relu', padding='same')(pool1)
    # pool2 = MaxPool1D(3)(conv2)
    # conv3 = Conv1D(128, 5, activation='relu', padding='same')(pool2)
    # pool3 = MaxPool1D(3)(conv3)
    # flat = Flatten()(pool3)
    # dense = Dense(128, activation='relu')(flat)

    convs = []
    for kernel_size in range(1, 5):
        conv = BatchNormalization()(embedding_sequences)
        conv = Conv1D(128, kernel_size, activation='relu')(conv)
        convs.append(conv)
    poolings = [GlobalMaxPooling1D()(conv) for conv in convs]
    x_concat = Concatenate()(poolings)
    dense = Dense(128, activation='relu')(x_concat)
    preds = Dense(1, activation='sigmoid')(dense)
    model = Model(sequence_input, preds)
    return model


train_pred = np.zeros((len(train), 1))
test_pred = np.zeros((len(test), 1))

cv_scores = []
skf = StratifiedKFold(n_splits=5, random_state=52, shuffle=True)
for i, (train_index, valid_index) in enumerate(skf.split(x_train, y_train)):
    print("n@:{}fold".format(i + 1))
    X_train = x_train[train_index]
    X_valid = x_train[valid_index]
    y_tr = y_train[train_index]
    y_val = y_train[valid_index]

    model = create_text_cnn()
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])

    model.summary()
    checkpoint = ModelCheckpoint(filepath='models/cnn_text_{}.h5'.format(i + 1),
                                 monitor='val_loss',
                                 verbose=1, save_best_only=True)
    history = model.fit(X_train, y_tr,
                        validation_data=(X_valid, y_val),
                        epochs=5, batch_size=32,
                        callbacks=[checkpoint,
                                   roc_auc_callback(training_data=(X_train, y_tr),
                                                    validation_data=(X_valid, y_val))])

    # model.load_weights('models/cnn_text.h5')
    yval_pred = model.predict(X_valid)
    train_pred[valid_index, :] = yval_pred
    cv_scores.append(roc_auc_score(y_val, yval_pred))
    test_pred += model.predict(x_test)
score = np.mean(cv_scores)
print("5折平均分数为：{}".format(score))

# 提交结果
test['target'] = test_pred / 5
test[['id', 'target']].to_csv('result/02_{}_cnn.csv'.format(score), index=None)

# 训练数据预测结果
train['pred'] = train_pred
# auc
train[['id', 'target', 'pred']].to_excel('result/train.xlsx', index=None)
print(roc_auc_score(train['target'].values, train['pred'].values))
