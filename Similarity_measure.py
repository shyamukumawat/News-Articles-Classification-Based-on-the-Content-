
# coding: utf-8

# In[2]:


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import nltk


# In[3]:


def process(file):
    raw=open(file).read()
    tokens=word_tokenize(raw)
    words=[w.lower() for w in tokens]
    #print("Tokens:-",words)
    porter=nltk.PorterStemmer()
    stemmed_tokens=[porter.stem(t) for t in words]
    
    
    ####--------Remove Stopwords-------####
    stop_words=set(stopwords.words('english'))
    filter_tokens=[w for w in stemmed_tokens if not w in stop_words]
    #print("Tokens after stopwords removing", filter_tokens)
    
    ####--------counting of words-----####
    count=nltk.defaultdict(int)
    for word in filter_tokens:
        count[word]+=1
    return count


# In[4]:


def cosine_similarity(a,b):
    dotProdct=np.dot(a,b)
    norm_a=np.linalg.norm(a)
    norm_b=np.linalg.norm(b)
    c=dotProdct/(norm_a*norm_b)
    return c;


# In[10]:


def getsimilarity(dict1,dict2):
    all_word_list=[]
    for key in dict1:
        all_word_list.append(key)
    for key in dict2:
        all_word_list.append(key)
    v1=np.zeros(len(all_word_list),dtype=int)
    v2=np.zeros(len(all_word_list),dtype=int)
    i=0
    for (key) in all_word_list:
        v1[i]=dict1.get(key,0)
        v2[i]=dict2.get(key,0)
        i=i+1
    '''i=0;
    for key in dict1:
        print("[", key, dict1[key],"], ",end=" ");
        i+=1;
        if i<100:
            continue
        else:
            break '''

 
    return cosine_similarity(v1,v2);
                                                                                          


# In[6]:


if __name__=="__main__":
    dict1=process('Benjamin.txt')
    dict2=process('Humpries.txt')
    dict3=process('wheaton.txt')
    
    print('similarity between Benjamin.txt and Humpries.txt is', getsimilarity(dict1,dict2))
    print('similarity between Benjamin.txt and wheaton.txt is', getsimilarity(dict1,dict3))
    print('similarity between Humpries.txt and wheaton.txt is', getsimilarity(dict2,dict3))
    dict1=process('Dying to live.txt')
    dict2=process('Humpries.txt')
    dict3=process('Karen Kaiser.txt')
    print('similarity between Dying to live.txt and Humpries.txt is', getsimilarity(dict1,dict2))
    print('similarity between Dying to live.txt and Karen Kaiser.txt is', getsimilarity(dict1,dict3))
    print('similarity between Humpries.txt and Karen Kaiser.txt is', getsimilarity(dict2,dict3))
    dict1=process('Dying to live.txt')
    dict2=process('wearing a mask.txt')
    dict3=process('Sarah Fader.txt')
    print('similarity between Dying to live.txt and wearing a mask.txt is', getsimilarity(dict1,dict2))
    print('similarity between Dying to live.txt and Sarah Fader.txt is', getsimilarity(dict1,dict3))
    print('similarity between wearing a mask.txt and Sarah Fader.txt is', getsimilarity(dict2,dict3))


# In[11]:


print('similarity between Benjamin.txt and Humpries.txt is', getsimilarity(dict1,dict2))

