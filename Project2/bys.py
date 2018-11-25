from Project2 import dp as dm
from sklearn.model_selection import train_test_split
import numpy as np
import math
NUM_CLASS=20# 类的个数

'''
the vocabulary of train 
'''
def calc_x_y_prob(x_train,y_train):
    res = [{} for i  in range(NUM_CLASS)] #########
    num_all = [0]*NUM_CLASS
    num_docs = len(x_train)
    for i in range(num_docs):
        for word in x_train[i]:
            res[y_train[i]][word] = (res[y_train[i]].get(word,0)+1)
            num_all[y_train[i]]+=1
    for i in range(NUM_CLASS):
        for key in res[i]:
            res[i][key] = (res[i][key]+1)/(num_all[i][key]+len(list(res[i].keys())))
    return res,num_all

def calc_y_prob(Y_train):
    y_prob = [1]*NUM_CLASS
    for label in Y_train:
        y_prob[label]+=1
    for i in range(NUM_CLASS):
        y_prob[i] = y_prob[i]/(len(Y_train)+NUM_CLASS)
    return y_prob

def naive_bayes(x_test,x_y_prob,y_prob,num_all):
    res_prob = [0]*NUM_CLASS
    for i in range(NUM_CLASS):
        for word in x_test:
            tmp_prob = x_y_prob[i].get(word,0)
            if(tmp_prob!=0):
                res_prob[i] += math.log(tmp_prob)
            else:
                res_prob[i] += math.log(1/(num_all[i]+len(list(x_y_prob[i].keys))))
        res_prob[i]+=math.log(y_prob[i])
    return np.array(res_prob).argmax()


if __name__=="__main__":
    print("begin")
    flag=False
    if(flag):
        main()
    else:
        data_save()
        main()

'''
if __name__=="__main__":
    print("begin")
    flag=True
    if(flag):
        main()
    else:
        data_save()
        main()
'''




