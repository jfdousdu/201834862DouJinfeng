import os
import re
import textblob as tb
import math
from nltk.corpus import stopwords as pw

##the path of the data article
newsPath = 'C:\\Users\\Donna\\PycharmProjects\\201834862DouJinfeng\\Project1\\20news-18828'

'''
docs_list,labels_list  to discovery these article and labels
'''
def doc_load():
    dirs =  os.listdir(newsPath)
    docs_list = []
    labels_list = []
    label = 0
    for dir in dirs:
        docs = os.listdir(newsPath+'\\'+dir)
        for doc in docs:
            with open(newsPath+'\\'+dir+'\\'+doc,'rb') as data:
                docs_list.append(data.read())
            labels_list.append(label)
        label = label+1
    return docs_list,labels_list

'''
cut the str into words
'''
def _string_split(str):
    str = str.lower()
    words = re.split('[^a-z]*',str)  ##split into words
    words = [word for word in words if len(word)>=3]
    words = tb.WordList(words).singularize()
    words = [word.lemmatize('v') for word in words]
    catcheStopWords = pw.words("english")
    words = [word for word in words if word not in catcheStopWords]
    return words
'''
cut docs_list into doc_list 
'''
def doc_split(docs_list):
    words_list = []
    for doc in docs_list:
        words_list.append(_string_split(str(doc)))
    return words_list

'''
built vocabulary dictionary
'''
def get_vocab(words_list):
    vocab = set()
    for doc in words_list:
        vocab = vocab|set(doc)   ##Let's take the union of the two
    return list(vocab)


def words_freq_proc(words_list,frequency):
    word_frequency = dict()

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

################################################
def words_statistics(words_list):
    index = 0
    word_df = dict()
    word_doc_tf = dict()
    doc_word_tf = []

    for doc in words_list:
        word_tf = dict()
        for word in doc:
            word_tf[word] = word_tf.get(word,0)+1
            if(word_doc_tf.__contains__(word)):
                word_doc_tf[word][words_list.index(doc)] = word_doc_tf[word].get(words_list.index(doc),0)+1
            else:
                word_doc_tf[word] = {words_list.index(doc):1}
    doc_word_tf.append(word_tf)


    for doc in doc_word_tf:
        for word in doc.keys():
            word_df[word] = word_df.get(word,0)+1


    num_docs = len(words_list)
    word_idf = dict()
    for word in word_df:
        word_idf[word] = math.log(num_docs/word_df[word])

    return word_doc_tf, word_idf, doc_word_tf