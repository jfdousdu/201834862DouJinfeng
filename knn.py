import numpy as np
import operator

#分别计算x_test和x_train的cos值，用于比较两者的相似性，如果越接近1则表明结果越相近
def cos_compute(x_test,x_train):
    len_test  = (x_test**2).sum()**0.5
    len_train = (x_train**2).sum()**0.5
    num_train = len(x_train)
    cos = []
    for i in range(num_train):
        cos.append(np.dot(x_test,x_train[i]))
    cos = cos/(len_test*len_train)
    return cos

#计算欧几里得距离，用于比较相似，其值越小越相似
def euclidean_compute(x_test,x_train):
     num_train = len(x_train)
     x_test = np.tile(x_test,(num_train,1))
     test_minus_train = x_test - x_test
     square = test_minus_train**2
     sum = square.sum(axis=1)
     distance = sum**0.5
     return distance

#knn方法
#'''
##x_train 训练数据
#y_traun 训练数据标签
#k
#返回x_test属于哪一类
#'''
def knn_cal(x_test,x_trai,y_train,k):
    distance = enclidean_compute（x_test,x_train)
    sortedDistIndecies = distance.argsort()
    classCount = {}

    for i in range(k):
        voteIlabel = y_train[sortedDistIndecies[i]]
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

