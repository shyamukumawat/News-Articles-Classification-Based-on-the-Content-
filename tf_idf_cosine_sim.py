
# coding: utf-8

# In[42]:


import os
import numpy as np
import nltk
import scipy
from nltk import download
from nltk.corpus import stopwords
from nltk import word_tokenize


# In[43]:


download('stopwords')
stop_words=stopwords.words('english')


# In[44]:


def preprocess(file_name):
    file_content=file_name.read()
    text =file_content.lower()
    doc = word_tokenize(text)
    doc = [word for word in doc if word not in stop_words]
    doc = [word for word in doc if word.isalpha()]
    return doc


# In[62]:


def computeTF(tkn_frq,n):
    #unq_tkn=set(tokens)
    #tkn_frq=dict.fromkeys(unq_tkn,0)
    #for words in tokens:
    #    tkn_frq[words]+=1
    tfDict={}
    for word,count in tkn_frq.items():
        tfDict[word]=count/float(n)
    return tfDict


# In[103]:


def computeIDF(docList):
    import math
    N=len(docList)
    x=[]
    
    for doc in docList:
        for word, val in doc.items():
            if(val>=0):
                x.append(word)
    idfDict = dict.fromkeys(set(x), 0)
    for doc in docList:
        for word, val in doc.items():
            if val > 0 :
                idfDict[word] += 1
    
    for word, val in idfDict.items():
        idfDict[word] = math.log10(N / float(val))
        
    return idfDict


# In[104]:


def computeTFIDF(tfBow, idfs):
    tfidf = {}
    for word, val in tfBow.items():
        tfidf[word] = val*idfs[word]
    return tfidf


# In[112]:


TEXT_TFIDF=[] ######2d list
i=0
for directory in sorted(os.listdir('News Articles/')):
    dir_tf=[]
    dir_tkn=[]
    dir_TFIDF=[]
    i+=1
    print(i)
    for file in sorted(os.listdir('News Articles/'+directory)):
        file_name=open('News Articles/'+directory+'/'+file)
        tokens=preprocess(file_name)
        dir_tkn.append(tokens)
        doc_tkn_frq=dict.fromkeys(set(tokens),0)
        for word in tokens:
            doc_tkn_frq[word]+=1
        
        doc_TF=computeTF(doc_tkn_frq,len(tokens))
        #print(doc_TF)
        dir_tf.append(doc_TF)
        dir_tkn_frq.append(doc_tkn_frq)
    ### idf of each directory
    dir_tkn_set=set(dir_tkn_set)
    
    idf=computeIDF(dir_tkn_frq)
    #print(idf)
    for doc_tf in dir_tf:
        dir_TFIDF.append(computeTFIDF(doc_tf,idf))
    TEXT_TFIDF.append(dir_TFIDF)
print('Executed Successfully')


# In[126]:


def cosine_similarity(a,b):
    dotProdct=np.dot(a,b)
    norm_a=np.linalg.norm(a)
    norm_b=np.linalg.norm(b)
    c=dotProdct/(norm_a*norm_b)
    return c;


# In[164]:


def getsimilarity(d1,d2):
    d1_d2_words=[]
    for word,key in d1.items():
        d1_d2_words.append(word)
    for word,key in d2.items():
        d1_d2_words.append(word)
    d1_d2_words=set(d1_d2_words)
    v1=np.zeros(np.array(len(d1_d2_words),int))
    v2=np.zeros(np.array(len(d1_d2_words),int))
    i=0
    for w in d1_d2_words:
        if w in d1:
            v1[i]=d1[w]
        if w in d2:
            v2[i]=d2[w]
        i+=1
    return cosine_similarity(v1,v2)
            
        


# In[166]:


for dirt in TEXT_TFIDF:
    for x in dirt:
        for y in dirt:
            print('similarity value :', 10*getsimilarity(x,y))
        print()
        print()
        print()
    break
        
        
        

