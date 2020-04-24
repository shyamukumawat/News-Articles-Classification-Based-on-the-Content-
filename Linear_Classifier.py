
# coding: utf-8

# In[4]:


## Naive classification based categorization of news article
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
import pandas, numpy, os


# In[5]:


import nltk
labels, texts=[],[]
trainDF=pandas.DataFrame()
trainDF['label']=labels
trainDF['text']=texts


# In[6]:


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


# In[7]:


print('Total No of NEWS Articles in BBC News Dataset:', len(trainDF))


# In[9]:


print('Splitting the dataset into Training and Testing Data with ratio 75:25')
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'],test_size=.25)


# In[10]:


print('Size of training and testing dataset size :', len(train_x),len(valid_x))


# In[11]:


#########   Preprocessing   ###########
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(trainDF['text'])
# transform the training and validation data using count vectorizer object
xtrain_count =  count_vect.transform(train_x)
xvalid_count =  count_vect.transform(valid_x)


# In[12]:


# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(trainDF['text'])
xtrain_tfidf =  tfidf_vect.transform(train_x)
xvalid_tfidf =  tfidf_vect.transform(valid_x)


# In[15]:


# ngram level tf-idf 
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram.fit(trainDF['text'])
xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)


# In[16]:


# characters level tf-idf
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram_chars.fit(trainDF['text'])
xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x) 
xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x)


# In[17]:


import string
trainDF['char_count'] = trainDF['text'].apply(len)
trainDF['word_count'] = trainDF['text'].apply(lambda x: len(x.split()))
trainDF['word_density'] = trainDF['char_count'] / (trainDF['word_count']+1)
trainDF['punctuation_count'] = trainDF['text'].apply(lambda x: len("".join(_ for _ in x if _ in string.punctuation))) 
trainDF['title_word_count'] = trainDF['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))
trainDF['upper_case_word_count'] = trainDF['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))


# In[18]:


pos_family = {
    'noun' : ['NN','NNS','NNP','NNPS'],
    'pron' : ['PRP','PRP$','WP','WP$'],
    'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'],
    'adj' :  ['JJ','JJR','JJS'],
    'adv' : ['RB','RBR','RBS','WRB']
}


# In[19]:


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


# In[20]:


trainDF['noun_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'noun'))
trainDF['verb_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'verb'))
trainDF['adj_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'adj'))
trainDF['adv_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'adv'))
trainDF['pron_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'pron'))


# In[21]:


def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    
    
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    
    return metrics.accuracy_score(predictions, valid_y)


# In[22]:


#Linear Classifier Training with three different features 
Linear_Classifier_tfidf=linear_model.LogisticRegression()
Linear_Classifier_ngram=linear_model.LogisticRegression()
Linear_Classifier_char_tfidf=linear_model.LogisticRegression()


# In[23]:


accuracy = train_model(Linear_Classifier_tfidf, xtrain_tfidf, train_y, xvalid_tfidf)
print ("Accuracy of Linear classifier using WordLevel TF-IDF features : ", accuracy)


# In[24]:


accuracy = train_model(Linear_Classifier_ngram, xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print ("Accuracy of Naive Bayes classifier using n-gram at n=2 TF-IDF features: ", accuracy)


# In[22]:





# In[25]:


print("Validation of Trained Models for different Articles :")


# In[ ]:


flg=1
while(flg):
    print('Enter article no to predict category of it\'s from which it belongs: ')
    art_no=int(input())
    print('predicted Class Type of selected NEWS Article using word level tf-idf features : ',end=' ')
    cls_no=Linear_Classifier_tfidf.predict(xvalid_tfidf[art_no])
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
    cls_no=Linear_Classifier_ngram.predict(xvalid_tfidf_ngram[art_no])
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

