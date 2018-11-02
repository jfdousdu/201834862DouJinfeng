import math
from  201834862DouJinfeng import knn,data_process as dp
from sklearn.model_selection import  train_test_split

FREQUENCY=15

#计算word_tf对应的tf-idf向量，以及其相应的tf-idf_train向量数组
def tf_idf(word_tf_test,word_doc_tf_train,word_idf_train,num_doc_train,doc_word_tf_train):
    word_in_test = [word for word in word_tf_test if word in list(word_doc_tf_train.keys())] #去掉test文档中没有在训练集中出现过的单词
    x_test = []
    x_train = []
    index_train = [] ############################
    doc_index = set()
    for word in word_in_test:
        x_test.append(1+math.log(word_tf_test[word])*word_idf_train[word]) #其实就是根据训练数据集来计算的，可以先计算清楚
        doc_index  =  doc_index|(word_doc_tf_train[word].keys()) #合并所有在训练数据集中出现过的文档index
    for index in doc_index:
        tmp = []
        for word in word_in_test:
            tf = doc_word_tf_train[index].get(word,0)
            if(tf!=0):
                tf = 1+ math.log(tf)
             tmp.append(tf*word_idf_train[word])
        x_train.append(tmp)
        index_train.append(index)
     return  x_test,x_train,x_train

#返回准确率
def compute_acc():
    index = 0
    docs_list,labels_list = dp.doc_load()
    words_list = dp.doc_split(docs_list)
    words_list = dp.words_freq_proc(words_list,FREQUENCY)
    X_train,X_test,Y_train,Y_test = train_test_split(words_list,labels_list,test_size = 0.2,random_state = 42)
    word_doc_tf_train,word_idf_train,doc_word_tf_train = dp.words_statistics(X_train)
    word_doc_tf_test,word_idf_test,doc_word_tf_test = dp.words_statistics(X_test)

    num_test = len(doc_word_tf_test)
    num_docs_train = len(doc_word_tf_train)
    num_right = 0
    print('Prepare is raady!')
    for i in range(num_test):
        x_test,x_train,index_train = tf_idf(doc_word_tf_test[i],word_doc_tf_train,word_idf_train,num_docs_train,doc_word_tf_train)
        y_train = [Y_train[j] for j in index_train]
        y_eval = knn.knn_cal(x_test,x_train,y_train,5)
        if(y_eval ==Y_test[i]):
            num_right= num_right+1
        index += 1
        if(index%10==0)
            print(index,' ',num_right/i)
    return num_right/num_test

def data_save():
     docs_list,labels_list = dp.doc_load()
     words_list = dp.doc_split(docs_list)
     words_list = dp.words_freq_proc(words_list,FREQUENCY)
     X_train, X_test, Y_train, Y_test = train_test_split(words_list, labels_list, test_size=0.2, random_state=42)
     word_doc_tf_train, word_idf_train, doc_word_tf_train = dp.words_statistics(X_train)
     word_doc_tf_test, word_idf_test, doc_word_tf_test = dp.words_statistics(X_test)

     with open('.\\tmp\\word_doc_tf_train.txt','w') as data:
         data.write(str(word_doc_tf_train))
     with open('.\\tmp\\word_idf_train.txt','w') as data:
         data.write(str(word_idf_train))
     with open('.\\tmp\\doc_word_tf_train.txt','w') as data:
         data.write(str(doc_word_tf_train))
     with open('.\\tmp\\doc_word_tf_test.txt','w') as data:
         data.write(str(doc_word_tf_test))
     with open ('.\\tmp\\Y_train.txt','w') as data:
         data.write(str(Y_train))
     with open('.\\tmp\\Y_test.txt','w') as data:
         data.write(str(Y_test))


def data_load():
    with open('.\\tmp\\word_doc_tf_train.txt', 'w') as data:
        tmp = data.read()
        word_doc_tf_train = eval(tmp)
    with open('.\\tmp\\word_idf_train.txt', 'w') as data:
        data.write(str(word_idf_train))
    with open('.\\tmp\\doc_word_tf_train.txt', 'w') as data:
        data.write(str(doc_word_tf_train))
    with open('.\\tmp\\doc_word_tf_test.txt', 'w') as data:
        data.write(str(doc_word_tf_test))
    with open('.\\tmp\\Y_train.txt', 'w') as data:
        data.write(str(Y_train))
    with open('.\\tmp\\Y_test.txt', 'w') as data:
        data.write(str(Y_test))
