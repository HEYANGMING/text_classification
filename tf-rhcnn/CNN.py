# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 04:08:20 2019

@author: heyangm
"""

import os
import time
import re
import pandas as pd
import numpy as np
import tensorflow as tf 
from hsoftmax import h_softmax as hs

###########数据整理##########################################
"获取文件路径并统计文件数量"
def getFilePathList(rootDir):
    filePath_list = []
    for walk in os.walk(rootDir):
        part_filePath_list = [os.path.join(walk[0], file) for file in walk[2]]
        filePath_list.extend(part_filePath_list)
    return filePath_list
filePath_list = getFilePathList('THUCNews')
print ('文件总数： ', len(filePath_list))

"获取所有样本标签"
label_list = []
for filePath in filePath_list:
    label = filePath.split('\\')[1]
    label_list.append(label)
print('标签总数： ', len(label_list))


"计算各个类别的标签数"
print(pd.value_counts(label_list))

"调用pickle库保存标签列表label_list"
import pickle
with open('label_list.pickle', 'wb') as file:
    pickle.dump(label_list, file)
    

"获取所有样本内容、保存content_list二进制文件"
def getFile(filePath):
    with open(filePath, encoding='utf8') as file:
        fileStr = ''.join(file.readlines(1000))
    return fileStr
 
interval = 20000  #每20000个文件保存为一个pickle文件
n_samples = len(label_list)
startTime = time.time()
directory_name = 'content_list'
if not os.path.isdir(directory_name):
    os.mkdir(directory_name)
for i in range(0, n_samples, interval):
    startIndex = i
    endIndex = i + interval
    content_list = []
   
    for filePath in filePath_list[startIndex:endIndex]:
        fileStr = getFile(filePath)
        content = re.sub('\s+', ' ', fileStr)
        content_list.append(content)
    save_fileName = directory_name + '/%06d-%06d.pickle' %(startIndex, endIndex)
    with open(save_fileName, 'wb') as file:
        pickle.dump(content_list, file)
    used_time = time.time() - startTime
    
"加载数据"
def getFilePathList(rootDir):
    filePath_list = []
    for walk in os.walk(rootDir):
        part_filePath_list = [os.path.join(walk[0], file) for file in walk[2]]
        filePath_list.extend(part_filePath_list)
    return filePath_list
 
startTime = time.time()
contentListPath_list = getFilePathList('content_list')
content_list = []
for filePath in contentListPath_list:
    with open(filePath, 'rb') as file:
        part_content_list = pickle.load(file)
    content_list.extend(part_content_list)
with open('label_list.pickle', 'rb') as file:
    label_list = pickle.load(file)
used_time = time.time() - startTime
print('loading pickle data used time: %.2f seconds' %used_time)
sample_size = len(content_list)
print('length of content_list，mean sample size: %d' %sample_size)


"制作保存词汇表，前10000字"
from collections import Counter 
def getVocabularyList(content_list, vocabulary_size):
    allContent_str = ''.join(content_list)
    counter = Counter(allContent_str)
    vocabulary_list = [k[0] for k in counter.most_common(vocabulary_size)]
    return ['PAD'] + vocabulary_list
startTime = time.time()
vocabulary_list = getVocabularyList(content_list, 10000)
used_time = time.time() - startTime
print('make wordlist used time: %.2f seconds' %used_time)

with open('vocabulary_list.pickle', 'wb') as file:
    pickle.dump(vocabulary_list, file)
with open('vocabulary_list.pickle', 'rb') as file:
    vocabulary_list = pickle.load(file)

##########训练集和测试集划分#########################################
startTime = time.time()
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(content_list, label_list)
train_content_list = train_X
train_label_list = train_y
test_content_list = test_X
test_label_list = test_y
used_time = time.time() - startTime
print('train_test_split used time : %.2f seconds' %used_time)

#############定义模型参数####################################
vocabulary_size = 10000  # 词汇表大小
sequence_length = 600  # 文本长度
embedding_size = 100  # 词向量维度
num_filters = 256  # 卷积核数目 
filter_size = 4  # 卷积核尺寸
num_fc_units = 128  # 全连接层神经元
dropout_keep_probability = 0.5  # dropout保留比例
learning_rate = 1e-3  # 学习率
batch_size = 64  # 每批训练大小

'获取词汇列表的词及对应的id，制成字典'
word2id_dict = dict([(b, a) for a, b in enumerate(vocabulary_list)])
'字典元素的列表'
content2idList = lambda content : [word2id_dict[word] for word in content if word in word2id_dict]
train_idlist_list = [content2idList(content) for content in train_content_list]  #id形式的训练集
used_time = time.time() - startTime
print('content2idList used time : %.2f seconds' %used_time)

num_classes = np.unique(label_list).shape[0]
import tensorflow.contrib.keras as kr
train_X = kr.preprocessing.sequence.pad_sequences(train_idlist_list, sequence_length) #使每个样本有相同的长度
from sklearn.preprocessing import LabelEncoder
labelEncoder = LabelEncoder()   
train_y = labelEncoder.fit_transform(train_label_list)  #对标签编码
train_Y = kr.utils.to_categorical(train_y, num_classes) #做Ont-Hot编码
 
tf.reset_default_graph()  #重置tensorflow计算图，加强代码的健壮性
X_holder = tf.placeholder(tf.int32, [None, sequence_length]) #文本对应的输入矩阵
Y_holder = tf.placeholder(tf.float32, [None, num_classes])   #文本对应的标签值
used_time = time.time() - startTime
print('data preparation used time : %.2f seconds' %used_time)

#############模型搭建#########################################
embedding = tf.get_variable('embedding', [vocabulary_size, embedding_size]) #更新模型的嵌入层
embedding_inputs = tf.nn.embedding_lookup(embedding, X_holder)  #将输入数据做词嵌入
conv = tf.layers.conv1d(embedding_inputs, num_filters, filter_size) #卷积层
max_pooling = tf.reduce_max(conv, [1])  #最大采样
full_connect = tf.layers.dense(max_pooling, num_fc_units) #全连接层
full_connect_activate = tf.nn.relu(full_connect)  #激活函数
softmax_before = tf.layers.dense(full_connect_activate, num_classes) #生成特征向量
predict_Y = hs.get_predict(softmax_before, X_holder)  #预测概率值
full_connect_dropout = tf.contrib.layers.dropout(full_connect, 
                                                 keep_prob=dropout_keep_probability) #duopout正则化
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y_holder,
                                                           logits=softmax_before)  #定义交叉熵损失函数
loss = tf.reduce_mean(cross_entropy)  #计算损失值
optimizer = tf.train.AdamOptimizer(learning_rate)  #定义优化器
train = optimizer.minimize(loss)  #最小化损失值
isCorrect = tf.equal(tf.argmax(Y_holder, 1), tf.argmax(predict_Y, 1)) #计算准确率
accuracy = tf.reduce_mean(tf.cast(isCorrect, tf.float32))  #平均准确率


##########参数初始化#########################################
init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)

##########模型训练#########################################
import random
test_idlist_list = [content2idList(content) for content in test_content_list]
test_X = kr.preprocessing.sequence.pad_sequences(test_idlist_list, sequence_length)
test_y = labelEncoder.transform(test_label_list) #对标签进行编码
test_Y = kr.utils.to_categorical(test_y, num_classes) 
startTime = time.time()
for i in range(4000):
    selected_index = random.sample(list(range(len(train_y))), k=batch_size)
    batch_X = train_X[selected_index]
    batch_Y = train_Y[selected_index]
    session.run(train, {X_holder:batch_X, Y_holder:batch_Y})  #开始训练模型
##########模型测试#########################################
    step = i + 1 
    if step % 200 == 0:
        selected_index = random.sample(list(range(len(test_y))), k=200)  #从测试集中随机挑选200个样本进行测试
        batch_X = test_X[selected_index]
        batch_Y = test_Y[selected_index]
        loss_value, accuracy_value = session.run([loss, accuracy], {X_holder:batch_X, Y_holder:batch_Y})
        print('step:%d loss:%.4f accuracy:%.4f' %(step, loss_value, accuracy_value))
used_time = time.time() - startTime
print('训练模型花费时间 : %.2f seconds' %used_time)
      
##########生成数据混淆矩阵#########################################
from sklearn.metrics import confusion_matrix
 
def predictAll(test_X, batch_size=100):
    predict_value_list = []
    for i in range(0, len(test_X), batch_size):
        selected_X = test_X[i: i + batch_size]
        predict_value = session.run(predict_Y, {X_holder:selected_X})
        predict_value_list.extend(predict_value)
    return np.array(predict_value_list) 
Y = predictAll(test_X)
y = np.argmax(Y, axis=1)
predict_label_list = labelEncoder.inverse_transform(y)

pd.set_option('display.max_rows', None)

print('\n混淆矩阵：\n', pd.DataFrame(confusion_matrix(test_label_list, predict_label_list), 
             columns=labelEncoder.classes_,
             index=labelEncoder.classes_ ))


##########统计测试结果#########################################
from sklearn.metrics import precision_recall_fscore_support
def eval_model(y_true, y_pred, labels):
    # 计算每个类别的Precision, Recall, f1, support
    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred)
    # 计算总体的平均Precision, Recall, f1, support
    tot_p = np.average(p, weights=s)
    tot_r = np.average(r, weights=s)
    tot_f1 = np.average(f1, weights=s)
    tot_s = np.sum(s)
    res1 = pd.DataFrame({
        u'Label': labels,
        u'Precision': p,
        u'Recall': r,
        u'F1': f1,
        u'Support': s
    })
    res2 = pd.DataFrame({
        u'Label': ['总体'],
        u'Precision': [tot_p],
        u'Recall': [tot_r],
        u'F1': [tot_f1],
        u'Support': [tot_s]
    })
    res2.index = [14]
    res = pd.concat([res1, res2])
    return res[['Label', 'Precision', 'Recall', 'F1', 'Support']]
    
print('\n测试统计结果：\n', eval_model(test_label_list, predict_label_list, labelEncoder.classes_))
    