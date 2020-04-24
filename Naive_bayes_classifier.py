
# coding: utf-8

# In[1]:


## Naive classification based categorization of news article
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
import pandas, numpy, os


# In[2]:


import nltk


# In[3]:


##### raw data framming using pandas
labels, texts=[],[]
trainDF=pandas.DataFrame()
trainDF['label']=labels
trainDF['text']=texts


# In[4]:


##########  Loading of raw data into pandas with labeling 
print('Categories of different NEWS articles are as following:')
i=1
for directory in sorted(os.listdir('News Articles/')):
    for file in sorted(os.listdir('News Articles/'+directory)):
        file_name=open('News Articles/'+directory+'/'+file)
        file_content=file_name.read()
        text =file_content.lower()
        #texts.append(texts)
        if(directory=='business'):
            labels.append(1)
            trainDF.loc[i]=['1',text]
        elif (directory=='entertainment'):
            labels.append(2)
            trainDF.loc[i]=['2',text]
        elif (directory=='politics'):
            labels.append(3)
            trainDF.loc[i]=['3',text]
        elif (directory=='sport'):
            labels.append(4)
            trainDF.loc[i]=['4',text]
        elif (directory=='tech'):
            labels.append(5)
            trainDF.loc[i]=['5',text]
        i=i+1
    print(directory)


# In[5]:


print('Total No of NEWS Articles in BBC News Dataset:', len(trainDF))


# In[6]:


print('Splitting the dataset into Training and Testing Data with ratio 70:30')
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'],test_size=.30)


# In[7]:


print('Size of training and testing dataset size :', len(train_x),len(valid_x))


# In[8]:


#########   Preprocessing   ###########
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(trainDF['text'])
# transform the training and validation data using count vectorizer object
xtrain_count =  count_vect.transform(train_x)
xvalid_count =  count_vect.transform(valid_x)


# In[9]:


# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(trainDF['text'])
xtrain_tfidf =  tfidf_vect.transform(train_x)
xvalid_tfidf =  tfidf_vect.transform(valid_x)


# In[10]:


# ngram level tf-idf 
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram.fit(trainDF['text'])
xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)


# In[11]:


# characters level tf-idf
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram_chars.fit(trainDF['text'])
xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x) 
xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x)


# In[12]:


import string
trainDF['char_count'] = trainDF['text'].apply(len)
trainDF['word_count'] = trainDF['text'].apply(lambda x: len(x.split()))
trainDF['word_density'] = trainDF['char_count'] / (trainDF['word_count']+1)
trainDF['punctuation_count'] = trainDF['text'].apply(lambda x: len("".join(_ for _ in x if _ in string.punctuation))) 
trainDF['title_word_count'] = trainDF['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))
trainDF['upper_case_word_count'] = trainDF['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))


# In[13]:


pos_family = {
    'noun' : ['NN','NNS','NNP','NNPS'],
    'pron' : ['PRP','PRP$','WP','WP$'],
    'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'],
    'adj' :  ['JJ','JJR','JJS'],
    'adv' : ['RB','RBR','RBS','WRB']
}


# In[14]:


# function to check and get the part of speech tag count of a words in a given sentence
def check_pos_tag(x, flag):
    cnt = 0
    try:
        wiki = textblob.TextBlob(x)
        for tup in wiki.tags:
            ppo = list(tup)[1]
            if ppo in pos_family[flag]:
                cnt += 1
    except:
        pass
    return cnt


# In[15]:


trainDF['noun_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'noun'))
trainDF['verb_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'verb'))
trainDF['adj_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'adj'))
trainDF['adv_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'adv'))
trainDF['pron_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'pron'))


# In[16]:


def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    
    
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    
    return metrics.accuracy_score(predictions, valid_y)


# In[17]:


#####  Classifier training with different feature
NB_Classifier_tfidf=naive_bayes.MultinomialNB()
NB_Classifier_ngram=naive_bayes.MultinomialNB()


# In[18]:


accuracy = train_model(NB_Classifier_tfidf, xtrain_tfidf, train_y, xvalid_tfidf)
print ("Accuracy of Naive Bayes classifier using WordLevel TF-IDF features : ", accuracy)


# In[19]:


accuracy = train_model(NB_Classifier_ngram, xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print ("Accuracy of Naive Bayes classifier using n-gram at n=2 TF-IDF features: ", accuracy)


# In[20]:





# In[21]:


print("Validation of Trained Models for different Articles :")


# In[22]:


flg=1
while(flg):
    print('Enter article no to predict category of it\'s from which it belongs: ')
    art_no=int(input())
    print('predicted Class Type of selected NEWS Article using word level tf-idf features : ',end=' ')
    cls_no=NB_Classifier_tfidf.predict(xvalid_tfidf[art_no])
    if(cls_no==1):
        print('Class No',cls_no,'Bussiness')
        print('Labeled class of document',art_no,' is ',valid_y[art_no])
    elif(cls_no==2):
        print('Class No',cls_no,'Entertainment')
        print('Labeled class of document',art_no,' is ',valid_y[art_no])
    elif(cls_no==3):
        print('Class No',cls_no,'Politices')
        print('Labeled class of document',art_no,' is ',valid_y[art_no])
    elif(cls_no==4):
        print('Class No',cls_no,'Technology')
        print('Labeled class of document',art_no,' is ',valid_y[art_no])
    elif(cls_no==5):
        print('Class No',cls_no,'Sports')
        print('Labeled class of document',art_no,' is ',valid_y[art_no])
    else:
        print(cls_no)
        print('Prediction is wrong')
    print('predicted Class Type of selected NEWS Article using n-gram tf-idf features : ',end=' ')
    cls_no=NB_Classifier_ngram.predict(xvalid_tfidf_ngram[art_no])
    if(cls_no==1):
        print('Class No',cls_no,'Bussiness')
        print('Labeled class of document',art_no,' is ',valid_y[art_no])
    elif(cls_no==2):
        print('Class No',cls_no,'Entertainment')
        print('Labeled class of document',art_no,' is ',valid_y[art_no])
    elif(cls_no==3):
        print('Class No',cls_no,'Politices')
        print('Labeled class of document',art_no,' is ',valid_y[art_no])
    elif(cls_no==4):
        print('Class No',cls_no,'Technology')
        print('Labeled class of document',art_no,' is ',valid_y[art_no])
    elif(cls_no==5):
        print('Class No',cls_no,'Sports')
        print('Labeled class of document',art_no,' is ',valid_y[art_no])
    else:
        print(cls_no)
        print('Prediction is wrong')
    print('Enter 1 to continue and 0 to exit:')
    flg=int(input())
    
        
    
    


# In[23]:




