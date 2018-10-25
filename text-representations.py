# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 12:53:48 2018

@author: faizankhan2
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import math
import re
import os
#os.chdir('C:\development\Text-Mining')
from nltk import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(ngram_range=(2,2))
file=open('data/test.ft.txt','r',encoding="utf-8")

g=file.read()

sentences=sent_tokenize(g)

wn=WordNetLemmatizer()


def preprocess_text(text):
    tokens = word_tokenize(text)
    stop_words=stopwords.words('english')
    stop_words.extend(['__label__1','__label__2'])
    tokens=[token.lower() for token in tokens if token not in stop_words]
    tokens=[re.sub(r'[^A-Za-z]+','',token) for token in tokens]
    tokens=[wn.lemmatize(token) for token in tokens]
    return tokens  


text_tokens=[]
for item in sentences[0:1000]:
    tokens = preprocess_text(item)
    temp = " ".join(tokens)
    text_tokens.append(temp)    

from nltk import FreqDist

word_dist = FreqDist()
for s in text_tokens:
    word_dist.update(s.split())

########################################################################################
from nltk.util import ngrams
from collections import Counter


text=''
for sent in text_tokens:
    text=text+sent

tokens=word_tokenize(text)    
bigrams = ngrams(tokens,2)

bigram_dict=dict(Counter(bigrams))

final_bigram_dict={}
for key,value in bigram_dict.items():
    new_key=" ".join(key)
    final_bigram_dict[new_key]=value

unigram_index= CountVectorizer(ngram_range=(1,1))
unigram_index.fit_transform(text_tokens)
unigram_dist = unigram_index.vocabulary_


def pmi(word1, word2 ,unigram_freq, bigram_freq):
    #print(word1,word2)
    prob_word1 = unigram_freq[word1]/float(sum(unigram_freq.values()))
    #print(prob_word1)
    prob_word2 = unigram_freq[word2]/float(sum(unigram_freq.values()))
    #print(prob_word2)
    prob_word1_word2 = bigram_freq[" ".join([word1,word2])]/float(sum(bigram_freq.values()))
    #print(prob_word1_word2)
    ratio = prob_word1_word2/float(prob_word1*prob_word2)
    #print(word1,word2,prob_word1,prob_word2)
    if ratio==0:
        return 0
    else:
        return math.log(ratio,2)

pmi_dict={}
for key in final_bigram_dict.keys():
    first_word = key.split()[0]
    second_word = key.split()[1]
    if (first_word in word_dist.keys()) and (second_word in word_dist.keys()):
        pmi_dict[key]=pmi(key.split()[0],key.split()[1],word_dist,final_bigram_dict)
    else:
        pmi_dict[key]=0


start = '\s'
end= '\e'

    
context_word_pairs={}
for story_id in range(0,len(text_tokens)):
    text_tokens[story_id] = start +' '+ text_tokens[story_id] +' '+ end
    list_of_words=text_tokens[story_id].split()
    context_word_pairs[story_id] = {}
    for word_index in range(1,len(list_of_words)-1):
        context_word_pairs[story_id][list_of_words[word_index]]=[list_of_words[word_index-1],list_of_words[word_index+1]]
#    
#tokens=[]
#for s in text_tokens:
#    tokens.extend(word_tokenize(s))
#
#unique_words=set(tokens)
#
list_cw_pairs=[]
for i in range(0,len(context_word_pairs)):
    list_cw_pairs.append(context_word_pairs[i])
#
#for word in unique_words:
#    print(context_word_pairs.get(word))
    

from collections import defaultdict
from functools import reduce
def foo(r, d):
    for k in d:
        r[k].append(d[k])
    
d = reduce(lambda r, d: foo(r, d) or r, list_cw_pairs, defaultdict(list))    
    
    
final_dict={}
for k,v in d.items():
    tmp_list_before=[]
    tmp_list_after=[]
    for x in range(0,len(v)):
        tmp_list_before.append(v[x][0])
        tmp_list_after.append(v[x][1])
    final_dict[k]=[tmp_list_before,tmp_list_after]
    
def create_vectors(word,context):
    #print(word)
    #print(context)
    vector = np.zeros((len(word_dist.keys()),)) 
    for x in range(0,len(context[0])):
        temp_word_1 = context[0][x]+" " + word
        if temp_word_1 in pmi_dict.keys():
            vector[word_dist[context[0][x]]] = pmi_dict[temp_word_1]
    for y in range(0,len(context[1])):
        temp_word_2 = word + " " + context[1][y]
        if temp_word_2 in pmi_dict.keys():
            vector[word_dist[context[1][y]]] = pmi_dict[temp_word_2]
    return vector

word_vectors=[]
word_list=[]
for w,v in final_dict.items():
    if w!="":
        word_vectors.append(create_vectors(w,v))
        word_list.append(w)
 
 
        
        
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

pca=PCA(n_components=300)

principalComponents=pca.fit_transform(word_vectors)

'''
new_pca = PCA(n_components = 2)
n=new_pca.fit_transform(principalComponents)
plt.scatter(n[:,0],n[:,1])

for i,word in enumerate(word_list[0:300]):
    plt.annotate(word,xy=(n[i,0],n[i,1]))
plt.show()

'''
labels = []
tokens = []

for word in range(0,len(word_list)):
    tokens.append(word_vectors[word])
    labels.append(word_list[word])

tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
new_values = tsne_model.fit_transform(tokens)

x = []
y = []
for value in new_values:
    x.append(value[0])
    y.append(value[1])
    
plt.figure(figsize=(16, 16)) 
for i in range(len(x)):
    plt.scatter(x[i],y[i])
    plt.annotate(labels[i],
                 xy=(x[i], y[i]),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')
plt.show()
'''
from annoy import AnnoyIndex

num=300
t = AnnoyIndex(num)

for i in range(0,len(principalComponents)):
    t.add_item(i,principalComponents[i])
    
t.build(10)
    
print(word_list[0])
#print(t.get_nns_by_item(0,5))
for i in t.get_nns_by_item(0,5):
    print(word_list[i])
#print(t.get_distance(1,4))

#print(t.get_distance(18,24))
