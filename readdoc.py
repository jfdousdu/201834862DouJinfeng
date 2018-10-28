#读取文档内容
import os

def doc_load():
    dirs = os.listdir("C:\\Users\\Donna\\PycharmProjects\\201834862DouJinfeng\\20news-18828") #列出所在文件夹中的所有文件夹的名字，返回一个文件夹名组成的数组
    docs_list = [] #创建一个空数组,存储文档
    labels_list = [] #创建一个空数组,存储文档类，别相当于标签
    label = 0  #列出labeles_lists数组的值从0开始

    for dir in dirs : #循环，循环遍历每一个文件夹，新生成的dir为一个标记dirs的变量
         docs = os.listdir("C:\\Users\\Donna\\PycharmProjects\\201834862DouJinfeng\\20news-18828"+'\\'+dir)
         for doc in docs :
             with open ("C:\\Users\\Donna\\PycharmProjects\\201834862DouJinfeng\\20news-18828"+'\\'+dir+'\\'+doc,'rb') as data:  #打开文档
                 docs_list.append(data.read()) #取读文档
                 labels_list.append(label)

         label = label+1  #对文档进行数值标记

    return docs_list,labels_list
#return docs_list,labels_list