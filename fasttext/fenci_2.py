#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 17:03:36 2018

@author: heyangm
"""

import jieba
import os

# 分词处理
ftext_all = open('/home/heyangm/Desktop/分词及词向量/result/news_all.txt', 'w',encoding = 'utf-8') 
ftrain = open('/home/heyangm/Desktop/分词及词向量/result/news_train.txt', 'w',encoding = 'utf-8')
ftest = open('/home/heyangm/Desktop/分词及词向量/result/news_test.txt', 'w',encoding = 'utf-8')
fvec = open('/home/heyangm/Desktop/分词及词向量/result/text_vec.txt', 'w',encoding = 'utf-8')

basedir = '/home/heyangm/Desktop/分词及词向量/news/'  # 文件地址
dir_list = ['ent','game','house','lottery','sport','stock']  #类别标签

num = -1
for e in dir_list:
    num += 1
    indir = basedir + e + '/'
    files = os.listdir(indir) #返回文件夹名列表
    count = 0
    for fileName in files:
        count += 1            
        filepath = indir + fileName
        with open(filepath,'r') as fr:
            text = fr.read()
        text = text.encode('utf-8').decode('utf-8')  #可有可无
        seg_text = jieba.cut(text.replace("\t"," ").replace("\n"," ").replace('，','').replace('。','').replace('“','').replace('”','').replace(':','').replace('…','').replace('！','').replace('？','').replace('~','').replace('（','').replace('）','').replace('、','').replace('；','').replace('/',''))
        outline = " ".join(seg_text)
        outline = outline + '\t__label__' + e + '\n' #对每个文本写上标签
           
        ftext_all.write(outline) # 全部的文件,训练模型用
        ftext_all.flush()  

        if count < 6000:
            ftrain.write(outline)
            ftrain.flush()
            continue
        elif count  < 12000:
            ftest.write(outline)
            ftest.flush()
            continue
        else:
            break

# 训练word2vec模型，生成词向量
from gensim.models import word2vec
s = word2vec.LineSentence('/home/heyangm/Desktop/分词及词向量/result/news_all.txt')
model = word2vec.Word2Vec(s,size=30,window=5,min_count=3,workers=4)
model.save('/home/heyangm/Desktop/分词及词向量/result/vec_model.txt')
print('中国：\n', model['中国'])
print('少女与模特：\n', model.similarity('中国','模特'))
print('与中国相似的词：\n', model.most_similar('中国'))

# 下面代码是想把整个文件的词转换成向量，但有些问题，等弄好之后发你
#with open('/home/heyangm/Desktop/分词及词向量/result/news_train.txt', 'r') as oread:
    #tem = oread.read()
#fvec.write(model[tem])

fvec.close()
ftrain.close()
ftest.close()
ftext_all.close()

