import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.python.client import device_lib
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding, Bidirectional, BatchNormalization
from keras.optimizers import RMSprop, Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
import jieba

print(device_lib.list_local_devices())
# K.tensorflow_backend._get_available_gpus()

'''
df_test = pd.read_csv('./train_data.csv')
titles = df_test['title']

seg_list_all = []
for i in titles:
    word_list = jieba.cut(i)
    word_str = ' '.join(list(word_list))
    seg_list_all.append(word_str)
    # seg_list_all.append(len(word_list))
# seg_list_all
news_test = df_test.iloc[:, 0:2]
df_seg = pd.DataFrame(np.array(seg_list_all).reshape(df_test.shape[0], 1))
df_seg.columns = ['seg_word']
test_all = pd.concat([news_test, df_seg], axis=1)
test_all.to_csv('./train_jieba.csv', index=False)
'''

news_all = pd.read_csv('./train_jieba.csv', index_col=None)
news_all = shuffle(news_all)
test_all = pd.read_csv('./test_jieba.csv')

train_rate = 0.9
val_rate = 0.1
# test_rate =0.2
train_df = news_all.iloc[:int(train_rate*len(news_all))]
val_df = news_all.iloc[int(train_rate*len(news_all)):int((train_rate+val_rate)*len(news_all))]
test_df = test_all.iloc[:int(len(test_all))]

# sns.countplot(train_df.label)
## 對dataset label 做編碼
train_y = train_df.label
val_y = val_df.label
# test_y = test_df.label
le = LabelEncoder()
train_y = le.fit_transform(train_y).reshape(-1, 1)
val_y = le.transform(val_y).reshape(-1, 1)
# test_y = le.transform(test_y).reshape(-1,1)
## 對dataset label做one-hot encoding
ohe = OneHotEncoder()
train_y = ohe.fit_transform(train_y).toarray()
val_y = ohe.transform(val_y).toarray()
# test_y = ohe.transform(test_y).toarray()
## 使用Tokenizer對詞組進行編碼
## 當我們建立一個Tokenizer對象後，使用該對象的fit_on_texts()函数，以空格去便是每個詞,
## 可以將輸入的文本中的每個詞編號，編號是根據詞頻的，詞頻越大，編號越小。
max_words = 1000000
max_len = 20
tok = Tokenizer(num_words=max_words, filters=u'，？：“”《》（ ）！', lower=False, split=u' ')  ## 使用的最大詞語數为20000
tok.fit_on_texts(train_df.seg_word)
## 使用word_index屬性可以看到每個詞對應的編碼
## 使用word_counts屬性可以看到每個詞對應的頻數
# for ii,iterm in enumerate(tok.word_index.items()):
#     if ii < 10:
#         print(iterm)
#     else:
#         break
# print("===================")  
# for ii,iterm in enumerate(tok.word_counts.items()):
#     if ii < 10:
#         print(iterm)
#     else:
#         break

train_seq = tok.texts_to_sequences(train_df.seg_word)
val_seq = tok.texts_to_sequences(val_df.seg_word)
test_seq = tok.texts_to_sequences(test_df.seg_word)
## 將每個序列調整相同的長度
train_seq_mat = sequence.pad_sequences(train_seq, maxlen=max_len)
val_seq_mat = sequence.pad_sequences(val_seq, maxlen=max_len)
test_seq_mat = sequence.pad_sequences(test_seq, maxlen=max_len)
# print(train_seq_mat.shape)
# print(val_seq_mat.shape)
# print(test_seq_mat.shape)

## LSTM模型
inputs = Input(name='inputs', shape=[max_len])
## Embedding(詞彙表大小,batch大小,每個新聞的詞長)
layer = Embedding(max_words+1, 500, input_length=max_len)(inputs)
layer = Dropout(0.5)(layer)
layer = BatchNormalization()(layer)
layer = Bidirectional(LSTM(192))(layer)
layer = Dropout(0.5)(layer)
layer = BatchNormalization()(layer)
layer = Dense(128, activation="sigmoid", name="FC1")(layer)
layer = Dropout(0.5)(layer)
layer = BatchNormalization()(layer)
# layer = Dense(64, activation='sigmoid', name='FC2')(layer)
layer = Dense(10, activation="softmax", name="FC3")(layer)
model = Model(inputs=inputs, outputs=layer)
model.compile(loss="categorical_crossentropy", optimizer=Adam(), metrics=["accuracy"])
model_fit = model.fit(train_seq_mat, train_y, batch_size=128, epochs=3,
                      validation_data=(val_seq_mat, val_y),
                      # callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)] ## 當val-loss不再提升時停止訓練
                     )

test_pre = model.predict(test_seq_mat)
#print(metrics.classification_report(np.argmax(test_pre,axis=1),np.argmax(test_y,axis=1)))
test_ans = pd.concat([test_all['id'], pd.DataFrame(np.argmax(test_pre, axis=1))], axis=1)
test_ans.columns = ['id', 'label']
test_ans.to_csv('./answer6.csv', index=False)

