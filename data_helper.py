#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 10:12:03 2017

@author: yoran
"""

import re
import jieba
import numpy as np
import pandas as pd
from collections import Counter

def clean_str(s):
	"""Clean sentence"""
	s = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", s)
	s = re.sub(r"\'s", " \'s", s)
	s = re.sub(r"\'ve", " \'ve", s)
	s = re.sub(r"n\'t", " n\'t", s)
	s = re.sub(r"\'re", " \'re", s)
	s = re.sub(r"\'d", " \'d", s)
	s = re.sub(r"\'ll", " \'ll", s)
	s = re.sub(r",", " , ", s)
	s = re.sub(r"!", " ! ", s)
	s = re.sub(r"\(", " \( ", s)
	s = re.sub(r"\)", " \) ", s)
	s = re.sub(r"\?", " \? ", s)
	s = re.sub(r"\s{2,}", " ", s)
	s = re.sub(r'\S*(x{2,}|X{2,})\S*',"xxx", s)
	s = re.sub(r'[^\x00-\x7F]+', "", s)
	return s.strip().lower()

def process(s):
    print('-------')
    s = s.replace('\n','')
    s = re.sub(r"\'s", " \'s", s)
    s = re.sub(r"\'ve", " \'ve", s)
    s = re.sub(r"n\'t", " n\'t", s)
    s = re.sub(r"\'re", " \'re", s)
    s = re.sub(r"\'d", " \'d", s)
    s = re.sub(r"\'ll", " \'ll", s)
    s = re.sub(r",", " , ", s)
    s = re.sub(r"!", " ! ", s)
    s = re.sub(r"\(", " \( ", s)
    s = re.sub(r"\)", " \) ", s)
    s = re.sub(r"\?", " \? ", s)
    s = re.sub(r"\s{2,}", " ", s)
    s = re.sub(r'\S*(x{2,}|X{2,})\S*',"xxx", s)
    #s = re.sub(r'[^\x00-\x7F]+', "", s)    
    print(s)
    seq = jieba.cut(s,cut_all = True)
    rts = ''
    for word in seq:
        if(len(word)>=1):
            #word = word.strip()
            #print(word)
            rts = rts + word + ' '
    return rts[:-1]

def load_data_and_labels1(filename):
    df = pd.read_excel(filename)
    selected = ['投诉问题类别','投诉问题内容']
    non_selected = list(set(df.columns)-set(selected))
    
    df = df.drop(non_selected,axis=1)
    df = df.dropna(axis=0,how='any',subset=selected)
    df = df.reindex(np.random.permutation(df.index))
    
    labels = sorted(list(set(df[selected[0]].tolist())))
    one_hot = np.zeros((len(labels),len(labels)),int)
    np.fill_diagonal(one_hot,1)
    label_dict = dict(zip(labels,one_hot))
    
    x_raw = df[selected[1]].apply(lambda x: process(x)).tolist()
    y_raw = df[selected[0]].apply(lambda y: label_dict[y]).tolist()
    
    return x_raw, y_raw, df, labels
    

def load_data_and_labels(filename):
    """load sentences and labels"""
    df = pd.read_csv(filename,compression='zip',dtype={'consumer_complaint_narrative':object})
    selected = ['product','consumer_complaint_narrative']
    non_selected = list(set(df.columns)-set(selected))
    
    df = df.drop(non_selected,axis=1)
    df = df.dropna(axis=0,how='any',subset=selected)
    df = df.reindex(np.random.permutation(df.index))
    
    labels = sorted(list(set(df[selected[0]].tolist())))
    one_hot = np.zeros((len(labels),len(labels)),int)
    np.fill_diagonal(one_hot,1)
    label_dict = dict(zip(labels,one_hot))
    
    x_raw = df[selected[1]].apply(lambda x: clean_str(x)).tolist()
    y_raw = df[selected[0]].apply(lambda y: label_dict[y]).tolist()
    
    return x_raw, y_raw, df, labels

def batch_iter(data,batch_size,num_epochs,shuffle=True):
    """iterate the data batch by batch"""
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(data_size/batch_size) + 1
    
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
            
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
            


if __name__ == '__main__':
    input_file = 'comments.xlsx'
    x_raw1, y_raw1, df1, labels1 = load_data_and_labels1(input_file)
    '''input_file = 'consumer_complaints.csv.zip'
    x_raw, y_raw, df, labels = load_data_and_labels(input_file)'''