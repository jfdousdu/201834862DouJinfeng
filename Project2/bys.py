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
            ##res[i][key] = (res[i][key]+1)/(num_all[i]+len(list(res[i].keys())))
            res[i][key] = res[i][key] / num_all[i]

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
                res_prob[i] += math.log(1/(num_all[i]+len(list(x_y_prob[i].keys()))))
        res_prob[i]+=math.log(y_prob[i])
    return np.array(res_prob).argmax()


'''
Calculate the exact value
'''
def main():
    X_train,Y_train,X_test,Y_test = data_load()
    y_prob = calc_y_prob(Y_train)
    x_y_prob,num_all = calc_x_y_prob(X_train,Y_train)
    num_right = 0
    for i in range(len(X_test)):
        label = naive_bayes(X_test[i],x_y_prob,y_prob,num_all)
        if(label == Y_test[i]):
            num_right += 1
        if((i+1)%10 ==0):
            print(i+1,num_right/(i+1))
    return num_right/len(X_test)





def data_load():
    with open('.\\tmp\\X_train.txt','r') as data:
        tmp = data.read()
        X_train = eval(tmp)
    with open('.\\tmp\\Y_train.txt', 'r') as data:
        tmp = data.read()
        Y_train = eval(tmp)
    with open('.\\tmp\\X_test.txt', 'r') as data:
        tmp = data.read()
        X_test = eval(tmp)
    with open('.\\tmp\\Y_test.txt', 'r') as data:
        tmp = data.read()
        Y_test = eval(tmp)
    return X_train,Y_train,X_test,Y_test


def data_save():
    docs_list,labels_list = dm.doc_load()
    words_list = dm.doc_split(docs_list)
    X_train, X_test, Y_train, Y_test = train_test_split(words_list, labels_list, test_size=0.2, random_state=42)

    with open('.\\tmp\\X_train.txt','w') as data:
         data.write(str(X_train))
    with open('.\\tmp\\Y_train.txt', 'w') as data:
         data.write(str(Y_train))
    with open('.\\tmp\\X_test.txt', 'w') as data:
         data.write(str(X_test))
    with open('.\\tmp\\Y_test.txt', 'w') as data:
         data.write(str(Y_test))


'''
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





