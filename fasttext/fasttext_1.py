#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 22:05:26 2018

@author: heyangm
"""

import logging
import fasttext

logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
classifier = fasttext.supervised("/home/heyangm/Desktop/分词及词向量/result/news_all.txt",
    "/home/heyangm/Desktop/分词及词向量/result/text.model", label_prefix="__label__")
classifier = fasttext.load_model('/home/heyangm/Desktop/分词及词向量/result/text.model.bin',
     label_prefix='__label__')
result_1 = classifier.test('/home/heyangm/Desktop/分词及词向量/result/news_test.txt')
print(result_1.precision)
print(result_1.recall)
