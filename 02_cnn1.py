from keras.layers import *
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
from tensorflow import set_random_seed

# 设置随机种子
SEED = 2019
np.random.seed(SEED)
set_random_seed(SEED)


def train_w2v(text_list=None, output_vector='data/w2v.txt'):
    """
    训练word2vec
    :param text_list:文本列表
    :param output_vector:词向量输出路径
    :return:
    """
    print("正在训练词向量。。。")
    corpus = [text.split() for text in text_list]
    model = Word2Vec(corpus, size=100, window=5, min_count=1, workers=multiprocessing.cpu_count())
    # 保存词向量
    model.wv.save_word2vec_format(output_vector, binary=False)


train = pd.read_csv("new_data/train.csv")
train_target = pd.read_csv('new_data/train_target.csv')
train = train.merge(train_target, on='id')
train = shuffle(train, random_state=2019)
test = pd.read_csv("new_data/test.csv")

# 全量数据
train['id'] = [i for i in range(len(train))]
test['target'] = [-1 for i in range(len(test))]
df = pd.concat([train, test], sort=False)
df['bankCard'] = df['bankCard'].fillna(value=999999999)  # bankCard存在空值

df['certId_first2'] = df['certId'].apply(lambda x: int(str(x)[:2]))  # 前两位
df['certId_middle2'] = df['certId'].apply(lambda x: int(str(x)[2:4]))  # 中间两位
df['certId_last2'] = df['certId'].apply(lambda x: int(str(x)[4:6]))  # 最后两位
df.drop(columns='certId', inplace=True)
# dist
df['dist_first2'] = df['dist'].apply(lambda x: int(str(x)[:2]))  # 前两位
df['dist_middle2'] = df['dist'].apply(lambda x: int(str(x)[2:4]))  # 中间两位
df['dist_last2'] = df['dist'].apply(lambda x: int(str(x)[4:6]))  # 最后两位
df.drop(columns='dist', inplace=True)

# residentAddr
df['residentAddr_first2'] = df['residentAddr'].apply(lambda x: int(str(x)[:2]) if x != -999 else -999)  # 前两位
df['residentAddr_middle2'] = df['residentAddr'].apply(lambda x: int(str(x)[2:4]) if x != -999 else -999)  # 中间两位
df['residentAddr_last2'] = df['residentAddr'].apply(lambda x: int(str(x)[4:6]) if x != -999 else -999)  # 最后两位
df.drop(columns='residentAddr', inplace=True)

# bankCard
df['bankCard'] = df['bankCard'].astype(int)
df['bankCard_first6'] = df['bankCard'].apply(lambda x: int(str(x)[:6]) if x != -999 else -999)
df['bankCard_last3'] = df['bankCard'].apply(lambda x: int(str(x)[6:].strip()) if x != -999 else -999)
df.drop(columns='bankCard', inplace=True)

df['certValidBegin_bin'] = pd.qcut(df['certValidBegin'], 20, labels=[i for i in range(20)])
df['certValidStop_bin'] = pd.qcut(df['certValidStop'], 20, labels=[i for i in range(20)])
df['lmt_bin'] = pd.qcut(df['lmt'], 20, labels=[i for i in range(20)])

no_fea = ['id', 'target', 'certValidStop', 'certValidBegin', 'lmt']

feas = [fea for fea in df.columns if fea not in no_fea]
print(len(feas))


def to_text(row):
    text = []
    for fea in feas:
        text.append(fea + '_' + str(row[fea]))
    return " ".join(text)


df['token_text'] = df.apply(lambda row: to_text(row), axis=1)
# df[['id', 'token_text']].to_csv('tmp/df.csv', index=None)
# df = pd.read_csv('tmp/df.csv')
texts = df['token_text'].values.tolist()
train_w2v(texts)

# 构建词汇表
tokenizer = Tokenizer(filters='|')
tokenizer.fit_on_texts(texts)
word_index = tokenizer.word_index
print(word_index)
print("词语数量个数：{}".format(len(word_index)))

# 数据
EMBEDDING_DIM = 100
MAX_SEQUENCE_LENGTH = len(feas)

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


# 创建embedding_layer
def create_embedding(word_index, w2v_file):
    """
    :param word_index: 词语索引字典
    :param w2v_file: 词向量文件
    :return:
    """
    embedding_index = {}
    f = open(w2v_file, 'r', encoding='utf-8')
    next(f)  # 下一行
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs
    f.close()
    print("Total %d word vectors in w2v_file" % len(embedding_index))

    embedding_matrix = np.random.random(size=(len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                input_length=MAX_SEQUENCE_LENGTH, trainable=False)
    return embedding_layer


def create_text_cnn():
    #
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedding_layer = create_embedding(word_index, 'data/w2v.txt')
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


class roc_auc_callback(Callback):
    def __init__(self, training_data, validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x, verbose=0)
        roc = roc_auc_score(self.y, y_pred)
        logs['roc_auc'] = roc_auc_score(self.y, y_pred)
        logs['norm_gini'] = (roc_auc_score(self.y, y_pred) * 2) - 1

        y_pred_val = self.model.predict(self.x_val, verbose=0)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        logs['roc_auc_val'] = roc_auc_score(self.y_val, y_pred_val)
        logs['norm_gini_val'] = (roc_auc_score(self.y_val, y_pred_val) * 2) - 1

        print('\rroc_auc: %s - roc_auc_val: %s - norm_gini: %s - norm_gini_val: %s' % (
            str(round(roc, 5)), str(round(roc_val, 5)), str(round((roc * 2 - 1), 5)), str(round((roc_val * 2 - 1), 5))),
              end=10 * ' ' + '\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


skf = StratifiedKFold(n_splits=5, random_state=52, shuffle=True)
cv_scores = []
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
                        epochs=5,
                        batch_size=64,
                        callbacks=[checkpoint, roc_auc_callback(training_data=(X_train, y_tr),
                                                                validation_data=(X_valid, y_val))])

    # model.load_weights('models/cnn_text.h5')
    train_pred[valid_index, :] = model.predict(X_valid)
    test_pred += model.predict(x_test)

    # 5折平均分数
    yval_pred = model.predict(X_valid)
    train_pred[valid_index, :] = yval_pred
    cv_scores.append(roc_auc_score(y_val, yval_pred))
    test_pred += model.predict(x_test)
score = np.mean(cv_scores)
print("5折平均分数为：{}".format(score))

test['target'] = test_pred / 5
test[['id', 'target']].to_csv('result/qiang_nn.csv', index=None)
# 提交结果
test['target'] = test_pred / 5
test[['id', 'target']].to_csv('result/02_cnn1{}_cnn.csv'.format(score), index=None)
# 训练数据预测结果
# 概率
# oof_df = pd.DataFrame(train_pred)
# train = pd.concat([train, oof_df], axis=1)
# # 标签
# targets = np.argmax(train_pred, axis=1)
train['pred'] = train_pred
# 分类报告
train[['id', 'target', 'pred']].to_excel('result/train.xlsx', index=None)
print(roc_auc_score(train['target'].values, train['pred'].values))
