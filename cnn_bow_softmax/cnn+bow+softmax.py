# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 15:27:08 2019

@author: admin
"""
##############制作词汇表#################################################
from collections import Counter 
def getVocabularyText(content_list, size): 
    size = size - 1 
    allContent = ''.join(content_list) 
    #将内容列表中的所有文章合并起来变成字符串str形式 
    counter = Counter(allContent) 
    #将Counter对象实例化并传入字符串形式的内容 
    vocabulary = [] 
    vocabulary.append('<PAD>') 
    for i in counter.most_common(size): 
        vocabulary.append(i[0]) 
    with open('vocabulary.txt', 'w', encoding='utf8') as file: 
        for vocab in vocabulary: 
            file.write(vocab + '\n')

##################读取数据##################################################
with open('./cnews/cnews.vocab.txt', encoding='utf8') as file: 
    vocabulary_list = [k.strip() for k in file.readlines()]    #读取词表 
    
with open('./cnews/cnews.train.txt', encoding='utf8') as file: 
    line_list = [k.strip() for k in file.readlines()]      #读取每行数据
   
    train_label_list = [k.split()[0] for k in line_list]   #将标签依次取出 
    
    train_content_list = [k.split(maxsplit=1)[1] for k in line_list] 
    #将内容依次取出,此处注意split()选择最大分割次数为1,否则句子被打断. #同理读取test数据 
with open('./cnews/cnews.test.txt', encoding='utf8') as file: 
    line_list = [k.strip() for k in file.readlines()] 
    test_label_list = [k.split()[0] for k in line_list] 
    test_content_list = [k.split(maxsplit=1)[1] for k in line_list]
    
#############文本向量化###################################################
word2id_dict = dict(((b, a) for a, b in enumerate(vocabulary_list))) 
def content2vector(content_list): 
    content_vector_list = [] 
    for content in content_list: 
        content_vector = [] 
        for word in content: 
            if word in word2id_dict: 
                content_vector.append(word2id_dict[word]) 
            else: 
                content_vector.append(word2id_dict['<PAD>']) 
        content_vector_list.append(content_vector)
    return content_vector_list 
train_vector_list = content2vector(train_content_list) 
test_vector_list = content2vector(test_content_list)

vocab_size = 5000  # 词汇表达小
seq_length = 600  # 句子序列长度
num_classes = 10  # 类别数

#########用keras的预处理模块去规范化句子序列长度##############################
import tensorflow.contrib.keras as kr 
train_X = kr.preprocessing.sequence.pad_sequences(train_vector_list,600) 
test_X = kr.preprocessing.sequence.pad_sequences(test_vector_list,600) 

##############对label进行one-hot处理########################################
from sklearn.preprocessing import LabelEncoder 
label = LabelEncoder() 
train_Y = kr.utils.to_categorical(label.fit_transform(train_label_list),num_classes=num_classes) 
test_Y = kr.utils.to_categorical(label.fit_transform(test_label_list),num_classes=num_classes) 

##############构建CNN模型########################################
embedding_dim = 128 # 词向量维度 
num_filters = 256 # 卷积核数目 
kernel_size = 5 # 卷积核尺寸 
hidden_dim = 128 # 全连接层神经元 
dropout_keep_prob = 0.5 # dropout保留比例 
learning_rate = 1e-3 # 学习率 
batch_size = 64 # 每批训练大小
epoch = 3000  #训练次数

import tensorflow as tf
X_holder = tf.placeholder(tf.int32,[None,seq_length]) #X的占位符由于句子是由id组成的向量，而id为int类型， #所以定义x的传入数据为tf.int32类型，#None表示可以传任意组X 
Y_holder = tf.placeholder(tf.float32,[None,num_classes]) #同理Y占位符向量维度为10，也即num_classes。
embedding = tf.get_variable('embedding', [vocab_size, embedding_dim]) #embedding字典维度为5000*128,128为词向量维度 
embedding_inputs = tf.nn.embedding_lookup(embedding, X_holder) #embedding_inputs的维度为(batch_size)64*600*128

conv1 = tf.layers.conv1d(inputs=embedding_inputs,filters=num_filters,kernel_size=kernel_size) #卷积层
max_pool = tf.reduce_max(conv1,reduction_indices=[1])  #池化层
full_connect = tf.layers.dense(max_pool,hidden_dim)    #全链接层

full_connect_dropout = tf.contrib.layers.dropout(full_connect,keep_prob=0.8) #duopout正则化
full_connect_activate = tf.nn.relu(full_connect_dropout)  # 激活
full_connect_last = tf.layers.dense(full_connect_activate,num_classes) #全链接层
predict_y = tf.nn.softmax(full_connect_last)  #softmax计算

###############定义损失函数####################################################
cross_entry = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y_holder,logits=full_connect_last)
loss = tf.reduce_mean(cross_entry) 
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate) 
train = optimizer.minimize(loss)

#####################计算准确率#################################################
correct = tf.equal(tf.argmax(Y_holder,1),tf.argmax(predict_y,1))
accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))

######################参数初始化##############################################
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

######################开始训练和测试##############################################
import random 
import time
startime = time.time()
for i in range(epoch): 
    train_index = random.sample(list(range(len(train_Y))),k=batch_size) 
    X = train_X[train_index] 
    Y = train_Y[train_index] 
    sess.run(train,feed_dict={X_holder:X,Y_holder:Y}) 
    step = i + 1 
    if step % 100 == 0: 
        test_index = random.sample(list(range(len(test_Y))), k=200)  #每训练100次随机选200测试样本进行测试
        x = test_X[test_index] 
        y = test_Y[test_index] 
        loss_value, accuracy_value = sess.run([loss, accuracy], {X_holder:x, Y_holder:y}) 
        print('step:%d loss:%.4f accuracy:%.4f' %(step, loss_value, accuracy_value))
usetime = time.time() - startime
print("训练模型花费的时间： %.3f 秒" %usetime)

##################生成混淆矩阵########################################################
import numpy as np 
import pandas as pd 
from sklearn.metrics import confusion_matrix 

pd.set_option('display.width', 100)
pd.set_option('display.max_rows', None)

def predictAll(test_X, batch_size=100): 
    predict_value_list = [] 
    for i in range(0, len(test_X), batch_size): 
        X = test_X[i: i + batch_size] 
        predict_value = sess.run(predict_y, {X_holder:X}) 
        predict_value_list.extend(predict_value) 
    return np.array(predict_value_list) 
Y = predictAll(test_X) 
y = np.argmax(Y, axis=1) 
predict_label_list = label.inverse_transform(y) 
print('\n混淆矩阵：\n', pd.DataFrame(confusion_matrix(test_label_list, predict_label_list), 
             columns=label.classes_, index=label.classes_ ))

##############统计测试结果#####################################################
from sklearn.metrics import precision_recall_fscore_support
def eval_model(y_true, y_pred, labels): 
    # 计算每个分类的Precision, Recall, f1, support 
    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred) 
    # 计算总体的平均Precision, Recall, f1, support 
    tot_p = np.average(p, weights=s) 
    tot_r = np.average(r, weights=s) 
    tot_f1 = np.average(f1, weights=s) 
    tot_s = np.sum(s) 
    res1 = pd.DataFrame({ u'Label': labels, u'Precision': p, u'Recall': r, u'F1': f1, u'Support': s }) 
    res2 = pd.DataFrame({ u'Label': ['总体'], u'Precision': [tot_p], u'Recall': [tot_r], u'F1': [tot_f1], u'Support': [tot_s] }) 
    res2.index = [10] 
    res = pd.concat([res1, res2]) 
    return res[['Label', 'Precision', 'Recall', 'F1', 'Support']] 
print('\n测试统计结果：\n', eval_model(test_label_list, predict_label_list, label.classes_))