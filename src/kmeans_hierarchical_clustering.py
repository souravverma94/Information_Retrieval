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
import matplotlib.patheffects as PathEffects
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.cm as cm
import scipy.cluster.hierarchy as shc

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
    
# convert the list of vectors to a matrix
tf_idf_matrix = np.matrix(tf_idf_list)

# # creating a heatmap of tf_idf matrix
# fig, ax = plt.subplots(figsize=(15,10)) 
# sns.heatmap(tf_idf_matrix)

# # find the cosine similarity between query vector and every document 
# query_vector = np.array([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])
# query_vector = query_vector.reshape(1,-1)
# cossim_with_query = cosine_similarity(query_vector, vector)
# sns.heatmap(cossim_with_query,0,1)

# #cosine similarity bw documents
# fig, ax = plt.subplots(figsize=(20,20)) 
# cossim_bw_docs = cosine_similarity(tf_idf_matrix[:30])
# sns.heatmap(cossim_bw_docs, linewidths=0.5)

# # K-means clustering
# no_of_clusters = 4
# # get the clusters
# kmeans = KMeans(n_clusters=no_of_clusters, random_state=0).fit(tf_idf_matrix)
# # We can extract labels from k-cluster solution and store is to a vector
# Y = kmeans.labels_ # a vector
# z = pd.DataFrame(Y.tolist()) # a list
# # Random state we define this random state to use this value in TSNE which is a randmized algo.
# RS = 25111993
# # Fit the model using t-SNE randomized algorithm
# digits_proj = TSNE(random_state=RS).fit_transform(tf_idf_matrix)

# # An user defined function to create scatter plot of vectors
# def scatter(x, colors):
#     # We choose a color palette with seaborn.
#     palette = np.array(sns.color_palette("hls", no_of_clusters))

#     # We create a scatter plot.
#     f = plt.figure(figsize=(18, 18))
#     ax = plt.subplot(aspect='equal')
#     sc = ax.scatter(x[:,0], x[:,1], lw=0, s=120,
#                     c=palette[colors.astype(np.int)])
#     #plt.xlim(-25, 25)
#     #plt.ylim(-25, 25)
#     ax.axis('off')
#     ax.axis('tight')

#     # We add the labels for each cluster.
#     txts = []
#     for i in range(no_of_clusters):
#         # Position of each label.
#         xtext, ytext = np.median(x[colors == i, :], axis=0)
#         txt = ax.text(xtext, ytext, str(i), fontsize=50)
#         txt.set_path_effects([
#             PathEffects.Stroke(linewidth=5, foreground="w"),
#             PathEffects.Normal()])
#         txts.append(txt)

#     return f, ax, sc, txts
# # Draw the scatter plot
# print(list(range(0, no_of_clusters)))
# sns.palplot(np.array(sns.color_palette("hls", no_of_clusters)))
# scatter(digits_proj, Y)
# plt.savefig('tsne-generated_'+ str(no_of_clusters)+'cluster.png', dpi=120)

# Hierachical clustering using scipy library
plt.figure(figsize=(30,20),)
plt.title("Winereview Hierarchical Clustering Dendograms")
dend = shc.dendrogram(shc.linkage(tf_idf_matrix, method='average', metric='euclidean'))
# dend = shc.dendrogram(shc.linkage(tf_idf_matrix, method='single', metric='euclidean'))
# dend = shc.dendrogram(shc.linkage(tf_idf_matrix, method='complete', metric='euclidean'))
plt.autoscale()
plt.show()