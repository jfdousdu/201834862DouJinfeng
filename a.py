def _string_split(str):
    '''
    internal
    :param str:
    :return: 把str分割成的单词数组
    '''
    #tb=TextBlob(bigString)
    #words=tb.words.singularize().lemmatize().lower()##
    #bigString=TextBlob(bigString)
    #bigString.correct()
    str.lower()
    #words=nltk.wordpunct_tokenize(bigString)
    words=re.split('[^a-z]*',str)## 以所有除字母以外的符号作为分隔符进行分词
    words = [word for word in words if len(word) >= 3]  ##去掉长度小于三的词
    words=tb.WordList(words).singularize()#复数变单数
    words=[word.lemmatize('v') for word in words]#过去式、进行时变一般形式
    cacheStopWords=pw.words("english")#得到stopwords
    words=[word for word in words if word not in cacheStopWords]## remove stopwords
    return words