# -*- coding: utf-8 -*-
"""
Information Retrieval Assignment 2

"""
import pandas as pd
import numpy as np
import re
import math
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
from nltk.stem import WordNetLemmatizer 
import matplotlib.pyplot as plt

#read the dataset
dataset = pd.read_csv("winemag-data.csv")
# only take the description column 
dataset = dataset.iloc[:,2]
#stop words 
stop_words = set(stopwords.words('english'))

# tokenize the words of each row
token_list = []
for row in dataset:
    token_list.append(word_tokenize(row))

tokens_list_ws = [] # tokens list without special characters and stop words 
token_stemmed_list = [] #stemmed tokens list
ps = PorterStemmer()
# removing single character tokens and special characters 
for row in token_list:
    tokens_ws = []
    tokens_stemmed = []
    for token in row:
        match = re.search('([a-z])\w+',token.lower()) # removing special character words
        if len(token)>1 and match and token.lower() not in stop_words:
            tokens_ws.append(token.lower())
            tokens_stemmed.append(ps.stem(token.lower()))
    tokens_list_ws.append(tokens_ws)
    token_stemmed_list.append(tokens_stemmed)


    
    
#     qw_frq = dict.fromkeys(myQuery,0)
#     for word in tokens_lematized:
#         if word in myQuery:
#             qw_frq[word]+=1
    
#     for word in myQuery:
#         qw_frq[word]= qw_frq[word]/len(tokens_lematized)
#     list_qwf.append(qw_frq)
    
# #plt.show()

# # document frequency df term came in how many documents
# df = dict.fromkeys(myQuery,0)
# for doc in list_qwf:
#     for word in myQuery:
#         if doc[word]>0:
#             df[word]+=1  

# idf = dict.fromkeys(myQuery,0)
# for word in myQuery:
#    df[word] = df[word] if df[word]>0 else 1
#    idf[word]= math.log(10/df[word])


# list_tf_idf = []
# for doc in list_qwf:
#     for word in myQuery:
#         doc[word]= doc[word]*idf[word]
#     list_tf_idf.append(list(doc.values()))

# vector = np.matrix(list_tf_idf)

# plt.plot(list_tf_idf[0],'g',list_tf_idf[1],'r',list_tf_idf[2],'b',list_tf_idf[3],'k',list_tf_idf[4],'c',list_tf_idf[5],'m',list_tf_idf[6],'tan',list_tf_idf[7],'violet',list_tf_idf[8],'peru',list_tf_idf[9],'teal')
# plt.savefig('vector_final.jpg')

