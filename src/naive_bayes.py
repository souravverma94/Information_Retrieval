# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 19:11:27 2020

Information Retrieval Assignment 3

"""
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns


#read the dataset
dataset = pd.read_csv("winemag-data.csv")
# only take the description column and points column 
dataset = dataset.iloc[:,[2,4]]
# method to label the wine according to its points 
def label_wines(points):
    if points >= 95:
        return "Classic"
    elif points>=90:
        return "Outstanding"
    elif points >=85:
        return "Very Good"
    elif points>=80:
        return "Good"
    else: 
        return "Mediocre"

wine_score_list = []
for score in dataset.iloc[:,1]:
    wine_score_list.append(label_wines(score))

dataset = dataset.drop(columns=['points'])
dataset.insert(1, 'class', wine_score_list)

stop_words = set(stopwords.words('english'))
stop_words.add("'ll")
stop_words.add("'re")
stop_words.add("'ve")
stop_words.add("19th")

#tokenize the description to make feature set
descriptions = dataset.iloc[:,0]
# tokenize the words of each row
tokens_list = []

for row in descriptions:
    tokens_list.append(word_tokenize(row))
    
tokens_list_ws = [] # tokens list after data preprocessing (stemming, remove stopwords)
possible_words = set() #set to store all possible words
ps = PorterStemmer() #initialize the stemmer
# stemming and removing single character tokens and special characters 
for row in tokens_list:
    tokens_ws = []
    for token in row:
        match = re.search('([a-z])\w+',token.lower()) # removing special character words
        if len(token)>1 and match and token.lower() not in stop_words:
            stemmed_token = ps.stem(token.lower())
            tokens_ws.append(stemmed_token)
            possible_words.add(stemmed_token)
    tokens_list_ws.append(tokens_ws)

feature_sets=[]
for row in tokens_list_ws:
    feature_set = dict.fromkeys(possible_words, 0)
    for token in row:
        feature_set[token]+=1
    feature_sets.append(list(feature_set.values()))


final_dataframe = pd.DataFrame(feature_sets, columns=possible_words)
final_dataframe['CLASS'] = wine_score_list
print(final_dataframe)

#CLASS is the label to predict classs of wine so we drop it from dataset
y = final_dataframe.CLASS
x = final_dataframe
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

# probability of each class in training dataset Pcj values
Pcj_dict = {}
classes = ['Classic','Outstanding','Very Good','Good'] 
class_dict = dict.fromkeys(classes, 0.0)
for row in y_train:
    class_dict[row]+=1
for claass in classes:
    pcj = class_dict[claass]/len(y_train)
    # print('PCj({})={}'.format(claass,pcj))
    Pcj_dict[claass] = pcj

# Compute p(w|class) using Laplace smoothing method 
Pwc_dict = {}
for e_class in classes:
    fil_df = x_train.loc[lambda df: df['CLASS']== e_class]
    wk_dict = dict.fromkeys(possible_words, 0.0)
    n=0 # total number of words in class
    for i in range(0,len(fil_df)):
        feature_dict = fil_df.iloc[i,:len(possible_words)+1]
        for key in possible_words:
            wk_dict[key]+= feature_dict[key]
            n+=feature_dict[key]
    for word in possible_words:
        nk = wk_dict[word]
        Pwc = (nk + 1)/ (n+len(possible_words))
        # print('P({}|{}) = {}'.format(word,e_class,Pwc))
        wk_dict[word] = Pwc
    Pwc_dict[e_class] = wk_dict

# list of posterior probability
posterior_prob_list = []
#calcualte posterior probability for each class
for idx,val in y_test.iteritems():
    post_prob_dict = dict.fromkeys(['id','Classic','Outstanding','Very Good','Good'])
    post_prob_dict['id']=idx
    token_list = tokens_list_ws[idx]
    for e_class in classes:
        post_prob = Pcj_dict[e_class]
        for token in token_list:
            post_prob*= Pwc_dict[e_class][token]
        post_prob_dict[e_class] = post_prob
    posterior_prob_list.append(post_prob_dict)
    # print('{} : {}'.format(idx, token_list))

# function to find maximum posterior probability among 4 classes and return the label of the class
def find_max_post_prob(data):
    prob=data['Classic']
    label='Classic'
    for lbl in ['Outstanding','Very Good','Good']:
        if data[lbl]>prob:
            prob = data[lbl]
            label=lbl
    return(label,prob)
# compute the results of classifier
predicted_results = {}
for row in posterior_prob_list:
    claass,max_prob = find_max_post_prob(row)
    predicted_results.update({ row['id'] : claass })

# Classifier results
y_predicted = pd.Series(predicted_results, name='CLASS')
conf_mat = confusion_matrix(y_test, y_predicted, labels=classes)
print(conf_mat)
print('\n\n')
sns.heatmap(conf_mat)

from sklearn.metrics import classification_report 

print(classification_report(y_test, y_predicted))

# word cloud 
out = []
gd =[]
vegd = []
clasic = []
for idx,val in y_test.iteritems():
    token_list = tokens_list_ws[idx]
    if val=='Classic':
        clasic+= token_list
    elif val=='Good':
        gd+=token_list
    elif val=='Very Good':
        vegd+=token_list
    elif val=='Outstanding':
        out+=token_list



wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                min_font_size = 10).generate(' '.join(clasic))

plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 

    
    
    
