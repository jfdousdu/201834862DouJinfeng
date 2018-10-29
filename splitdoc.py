

import re
import textblob as tb

text ="I went to  Shanghai last monday and brought mang eggs!"

def string_split(text):
     lowers = text.lower() #大小写
     words = re.split('[^a-z]*',lowers) #以所有字母以外的符号作为分隔符进行分词
     words = [word for word in words if len(word) >=3 ] #去掉长度小于三的词
     words = tb.WordList(words).singularize()  # 复数变单数
     words = [word.lemmatize('v') for word in words ]  #过去式、进行时变一般形式
     return words