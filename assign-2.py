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
import matplotlib.cm as cm

#read the dataset
dataset = pd.read_csv("winemag-data.csv")
# only take the description column 
dataset = dataset.iloc[:,2]
#stop words 
stop_words = set(stopwords.words('english'))
stop_words.add("'ll")
stop_words.add("'re")
stop_words.add("'ve")

#Query
myQuery = set(['aroma', 'fruit','herb','orang','blackberri','juici','cranberri','zin','worth','spice','peach','pineappl'])

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
#create a map to store frequency of my query words
term_freq_list = []
for tokens in token_stemmed_list:
    freq_vector = dict.fromkeys(myQuery,0)
    for word in tokens:
        if word in myQuery:
            freq_vector[word]+=1
    term_freq_list.append(freq_vector)


# compute weighted term freq i.e term_freq>0 should be 1 + log(term_freq) base 10
for freq_dict in term_freq_list:
    for key in freq_dict:
        val = freq_dict[key]
        if val > 0:
            freq_dict[key] = 1 + math.log10(val)
          
        
# compute document frequency of query words
doc_freq = dict.fromkeys(myQuery, 0)
# iterate through every document and find out document frequency of every query word
for word in myQuery:
    for freq_dict in term_freq_list:
        if freq_dict[word]>0:
            doc_freq[word]+=1
# compute idf inverse document frequency log10 N/d_ft N is number of documents which is 500 for now
idf = dict.fromkeys(myQuery, 0)
for word in myQuery:
    val = 500/doc_freq[word]
    idf[word] = math.log10(val)
 

# compute TF_IDF iterate through all term frequency list and multiply the idf value respectively 
tf_idf_list = []
for freq_dict in term_freq_list:
    tf_idf = []
    for num1,num2 in zip(freq_dict.values(), idf.values()): 
        tf_idf.append(num1 * num2)
    tf_idf_list.append(tf_idf)

vector = np.matrix(tf_idf_list)

# creating a heatmap using tf_idf vector
fig, ax = plt.subplots(1,1)
img = ax.imshow(vector,extent=[-1,500,-1,500])
ax.set_xticklabels(myQuery)
fig.colorbar(img)



