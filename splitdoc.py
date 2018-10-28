import nltk
from nltk.corpus import stopwords   #使用nltk提供的的stopwords
from nltk.stem.porter import PorterStemmer  # 提取词干
from nltk.stem import WordNetLemmatizer

#去除给定text的太小写以及标点符号

def get_token(text):
     lowers = text.lower() #大小写
    # remove_punctuation_map = dict((ord(char),None) for char in string.punctuation )