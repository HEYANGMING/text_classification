#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#fvec = open('/home/heyangm/Desktop/分词及词向量/result/text_vec.txt', 'w',encoding = 'utf-8')
from gensim.models import word2vec
s = word2vec.LineSentence('/home/heyangm/Desktop/分词及词向量/result/news_all.txt')
model = word2vec.Word2Vec(s,size=30,window=5,min_count=3,workers=4)
model.save('/home/heyangm/Desktop/分词及词向量/result/vec_model.txt')
print('中国：\n', model['中国'])
print('少女与模特：\n', model.similarity('中国','模特'))
print('与中国相似的词：\n', model.most_similar('中国'))
#with open('/home/heyangm/Desktop/分词及词向量/result/news_train.txt', 'r') as oread:
    #tem = oread.read()
#for t in tem:
    #fvec.write(model[tem])

