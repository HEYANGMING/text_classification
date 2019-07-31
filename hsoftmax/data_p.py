# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 13:48:49 2019

@author: admin
"""

import jieba
import os

ftext_all = open('E:\\shujuji\\cnn_THUC\\cnn_hsoftmax\\news_all.txt', 'w', encoding = 'utf-8') 

basedir = 'E:\\shujuji\\cnn_THUC\\THUCNews\\'   # 文件地址
dir_list = ['娱乐','游戏','房产','彩票','体育','股票','家居','教育','军事','星座','科技','社会','财经','时尚']  #类别标签

num = -1
for e in dir_list:
    num += 1
    indir = basedir + e + '\\'
    files = os.listdir(indir) #返回文件列表
    count = 0
    for fileName in files:
        count += 1            
        filepath = indir + fileName
        with open(filepath,'r', encoding = 'utf-8') as fr:
            text = fr.read()
        seg_text = jieba.cut(text.replace("\t"," ").replace("\n"," ").replace('，','').replace
                             ('。','').replace('“','').replace('”','').replace(':','').replace
                             ('…','').replace('！','').replace('？','').replace('~','').replace
                             ('（','').replace('）','').replace('、','').replace('；','').replace('/',''))
        outline = " ".join(seg_text)
        outline = outline + '\t__label__' + e + '\n' #对每个文本写上标签
        
        if count < 5000:
            ftext_all.write(outline) 
            ftext_all.flush()
ftext_all.close()        