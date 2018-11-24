import re
import textblob as tb
from  nltk.corpus import stopwords as pw
import numpy as np
import math
import os
import nltk


#读取文档字典
def doc_load():
    dirs = os.listdir("C:\\Users\\Donna\\PycharmProjects\\201834862DouJinfeng\\Project1\\20news-18828") #列出所在文件夹中的所有文件夹的名字，返回一个文件夹名组成的数组
    docs_list = [] #创建一个空数组,存储文档
    labels_list = [] #创建一个空数组,存储文档类，别相当于标签
    label = 0  #列出labeles_lists数组的值从0开始

    for dir in dirs : #循环，循环遍历每一个文件夹，新生成的dir为一个标记dirs的变量
         docs = os.listdir("C:\\Users\\Donna\\PycharmProjects\\201834862DouJinfeng\\Project1\\20news-18828"+'\\'+dir)
         for doc in docs :
             with open ("C:\\Users\\Donna\\PycharmProjects\\201834862DouJinfeng\\Project1\\20news-18828"+'\\'+dir+'\\'+doc,'rb') as data:  #打开文档
                 docs_list.append(data.read()) #取读文档
             labels_list.append(label)

         label = label+1  #对文档进行数值标记

    return docs_list,labels_list

#对于一个文档内容进行分割
#可将text="I went to  Shanghai last monday and brought mang eggs!"转化为<class 'list'>: ['go', 'shanghai', 'last', 'monday', 'and', 'bring', 'mang', 'egg']
def string_split(text):
     lowers = text.lower() #大小写
     words = re.split('[^a-z]*',lowers) #以所有字母以外的符号作为分隔符进行分词
     words = [word for word in words if len(word) >=3 ] #去掉长度小于三的词
     words = tb.WordList(words).singularize()  # 复数变单数
     words = [word.lemmatize('v') for word in words ]  #过去式、进行时变一般形式
     stoplists = pw.words("english") #得到stopwords相应的stoplists
     words = [word for word in words if word not in stoplists] #删除stoplists当中的单词
     return words

#对每个文档进行分割
def doc_split(docs_list):
     words_list = []
     for doc in docs_list:
          words_list.append(string_split(str(doc))) ####此处写的不一样
     return  words_list

#将文档中单词频率低于frequency的单词去掉，并返回去掉低频词的new_words_list
def words_freq_proc(words_list,frequency):
     word_frequency = dict()            #####what is mean it
     for doc in words_list:
          for word in doc:
               word_frequency[word] = word_frequency.get(word,0)+1
     new_words_list = []
     for doc in words_list:
          new_list = []
          for word in doc:
               if(word_frequency[word]>=frequency):
                    new_list.append(word)
          new_words_list.append(new_list)
     return new_words_list

#检查词频创建词典 the more important things in this codes
def words_statistics(words_list):
     index = 0
     word_df = dict() #每个单词的df
     word_doc_tf = dict() #每个单词在哪个文档中出现过
     doc_word_tf = [] #每个文档中包含了哪些单词

     #统计word_doc_tf,doc_word_tf
     for doc in words_list:
          word_tf = dict()
          for word in doc:
               word_tf[word] =word_tf.get(word,0)+1
               if(word_doc_tf.__contains__(word)):
                    word_doc_tf[word][words_list.index(doc)] = word_doc_tf[word].get(words_list.index(doc),0)+1
               else:
                    word_doc_tf[word] = {words_list.index(doc):1}
          doc_word_tf.append(word_tf)

     #统计word_df
     for doc in doc_word_tf:
          for word in doc.keys(): #############
               word_df[word] = word_df.get(word,0)+1

     #通过word_df计算word_idf
     num_docs = len(words_list)
     word_idf = dict()
     for word in word_df:
          word_idf[word] = math.log(num_docs/word_df[word])


     return word_doc_tf,word_idf,doc_word_tf
