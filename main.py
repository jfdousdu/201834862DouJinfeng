import math
from  201834862DouJinfeng import knn,data_process as dp
from sklearn.model_selection import  train_test_split

FREQUENCY=15



def _tf_idf(word_tf_test,word_doc_tf_train,word_idf_train,num_doc_train,doc_word_tf_train):
    '''
    计算word_tf对应的tf-idf向量，以及其对应的tf-idf_train向量数组
    :param word_tf:
    :param word_doc_tf:
    :param word_df:
    :return:
    '''

    word_in_test=[word for word in word_tf_test if word in list(word_doc_tf_train.keys())]#去掉test文档中没有在训练数据集中出现过的单词
    x_test=[]
    x_train=[]
    index_train=[]
    doc_index=set()
    for word in word_in_test:
        #idf=math.log(num_doc_train/word_df_train.get(word))#其实都是根据训练数据集来进行计算的，可以先计算清楚。
        x_test.append((1+math.log(word_tf_test[word]))*word_idf_train[word])
        doc_index=doc_index|(word_doc_tf_train[word].keys())###合并所有在训练数据集中出现过的文档的index
    for index in doc_index:
        tmp=[]
        for word in word_in_test:
            tf=doc_word_tf_train[index].get(word,0)
            if(tf!=0):
                tf=1+math.log(tf)
            tmp.append(tf*word_idf_train[word])
        x_train.append(tmp)
        index_train.append(index)
    return x_test,x_train,index_train

