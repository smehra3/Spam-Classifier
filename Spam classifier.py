
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.1** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-text-mining/resources/d9pwm) course resource._
# 
# ---

# # Assignment 3
# 
# In this assignment you will explore text message data and create models to predict if a message is spam or not. 

# In[3]:


import pandas as pd
import numpy as np

spam_data = pd.read_csv('spam.csv')

spam_data['target'] = np.where(spam_data['target']=='spam',1,0)
spam_data.head(10)


# In[4]:


from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(spam_data['text'], 
                                                    spam_data['target'], 
                                                    random_state=0)


# ### Question 1
# What percentage of the documents in `spam_data` are spam?
# 
# *This function should return a float, the percent value (i.e. $ratio * 100$).*

# In[3]:


def answer_one():
    return spam_data.sum(axis = 0, skipna = True)['target']*100/len(spam_data) #Your answer here


# In[4]:


answer_one()


# ### Question 2
# 
# Fit the training data `X_train` using a Count Vectorizer with default parameters.
# 
# What is the longest token in the vocabulary?
# 
# *This function should return a string.*

# In[7]:


from sklearn.feature_extraction.text import CountVectorizer

def answer_two():
    
    vectorizer = CountVectorizer()
    x_vectorized = vectorizer.fit(X_train)
    #print sorted([word, len(word) for word in vectorizer.get_feature_names()], key=lambda x: x[1], reverse=True)[0][0]
    return sorted([(word, len(word)) for word in vectorizer.get_feature_names()], key=lambda x: x[1], reverse=True)[0][0]#Your answer here


# In[8]:


answer_two()


# ### Question 3
# 
# Fit and transform the training data `X_train` using a Count Vectorizer with default parameters.
# 
# Next, fit a fit a multinomial Naive Bayes classifier model with smoothing `alpha=0.1`. Find the area under the curve (AUC) score using the transformed test data.
# 
# *This function should return the AUC score as a float.*

# In[11]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score

def answer_three():
    vectorizer = CountVectorizer()
    nbc = MultinomialNB(alpha=0.1)
    nbc.fit(vectorizer.fit_transform(X_train),y_train)
    y_predicted = nbc.predict(vectorizer.transform(X_test)) 
    
    return roc_auc_score(y_test, y_predicted)#Your answer here


# In[12]:


answer_three()


# ### Question 4
# 
# Fit and transform the training data `X_train` using a Tfidf Vectorizer with default parameters.
# 
# What 20 features have the smallest tf-idf and what 20 have the largest tf-idf?
# 
# Put these features in a two series where each series is sorted by tf-idf value and then alphabetically by feature name. The index of the series should be the feature name, and the data should be the tf-idf.
# 
# The series of 20 features with smallest tf-idfs should be sorted smallest tfidf first, the list of 20 features with largest tf-idfs should be sorted largest first. 
# 
# *This function should return a tuple of two series
# `(smallest tf-idfs series, largest tf-idfs series)`.*

# In[25]:


from sklearn.feature_extraction.text import TfidfVectorizer

def answer_four():
    vectorizer = TfidfVectorizer()
    x_train_vectorized = vectorizer.fit_transform(X_train)
    names_idfs =  list(zip(vectorizer.get_feature_names(), vectorizer.idf_))
    #print(x_train_vectorized.max(0).toarray()[0].argsort())
    smallest_20_tfidfs = sorted(names_idfs, key=lambda x: x[1])[:20]
    largest_20_tfidfs = sorted(names_idfs, key=lambda x: x[1],reverse=True)[:20]
    return (pd.Series([f[1] for f in smallest_20_tfidfs],index=[f[0] for f in smallest_20_tfidfs]),
            pd.Series([f[1] for f in largest_20_tfidfs],index=[f[0] for f in largest_20_tfidfs]))


# In[26]:


answer_four()


# ### Question 5
# 
# Fit and transform the training data `X_train` using a Tfidf Vectorizer ignoring terms that have a document frequency strictly lower than **3**.
# 
# Then fit a multinomial Naive Bayes classifier model with smoothing `alpha=0.1` and compute the area under the curve (AUC) score using the transformed test data.
# 
# *This function should return the AUC score as a float.*

# In[9]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score

def answer_five():
    vectorizer = TfidfVectorizer(min_df=3)
    nbc = MultinomialNB(alpha=0.1)
    nbc.fit(vectorizer.fit_transform(X_train),y_train)
    y_predicted = nbc.predict(vectorizer.transform(X_test)) 
    
    return roc_auc_score(y_test, y_predicted)#Your answer here


# In[10]:


answer_five()


# ### Question 6
# 
# What is the average length of documents (number of characters) for not spam and spam documents?
# 
# *This function should return a tuple (average length not spam, average length spam).*

# In[41]:


def answer_six():
    sum_lengths_spam=0
    sum_lengths_notspam=0
    number_spam=0
    number_notspam=0
    for data in spam_data.iterrows():
        if data[1]['target']==1:
            number_spam+=1
            sum_lengths_spam+=len(data[1]['text'])
        else:
            number_notspam+=1
            sum_lengths_notspam+=len(data[1]['text'])
    return (sum_lengths_notspam/number_notspam, sum_lengths_spam/number_spam)#Your answer here


# In[42]:


answer_six()


# <br>
# <br>
# The following function has been provided to help you combine new features into the training data:

# In[5]:


def add_feature(X, feature_to_add):
    """
    Returns sparse feature matrix with added feature.
    feature_to_add can also be a list of features.
    """
    from scipy.sparse import csr_matrix, hstack
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')


# ### Question 7
# 
# Fit and transform the training data X_train using a Tfidf Vectorizer ignoring terms that have a document frequency strictly lower than **5**.
# 
# Using this document-term matrix and an additional feature, **the length of document (number of characters)**, fit a Support Vector Classification model with regularization `C=10000`. Then compute the area under the curve (AUC) score using the transformed test data.
# 
# *This function should return the AUC score as a float.*

# In[56]:


from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score

def answer_seven():
    vectorizer = TfidfVectorizer(min_df=5)
    x_train_transformed_added_len = add_feature(vectorizer.fit_transform(X_train),X_train.str.len())
    x_test_transformed_added_len = add_feature(vectorizer.transform(X_test),X_test.str.len())
    
    svc_classifier = SVC(C=10000)
    svc_classifier.fit(x_train_transformed_added_len,y_train)
    
    y_predicted = svc_classifier.predict(x_test_transformed_added_len)
    return roc_auc_score(y_test, y_predicted)#Your answer here


# In[57]:


answer_seven()


# ### Question 8
# 
# What is the average number of digits per document for not spam and spam documents?
# 
# *This function should return a tuple (average # digits not spam, average # digits spam).*

# In[64]:


def answer_eight():
    sum_len_digits_spam=0
    sum_len_digits_notspam=0
    number_spam=0
    number_notspam=0
    for data in spam_data.iterrows():
        if data[1]['target']==1:
            sum_len_digits_spam+=len(''.join(filter(str.isdigit, data[1]['text'])))
            number_spam+=1
        else:
            sum_len_digits_notspam+=len(''.join(filter(str.isdigit, data[1]['text'])))
            number_notspam+=1
    return (sum_len_digits_notspam/number_notspam, sum_len_digits_spam/number_spam)#Your answer here


# In[65]:


answer_eight()


# ### Question 9
# 
# Fit and transform the training data `X_train` using a Tfidf Vectorizer ignoring terms that have a document frequency strictly lower than **5** and using **word n-grams from n=1 to n=3** (unigrams, bigrams, and trigrams).
# 
# Using this document-term matrix and the following additional features:
# * the length of document (number of characters)
# * **number of digits per document**
# 
# fit a Logistic Regression model with regularization `C=100`. Then compute the area under the curve (AUC) score using the transformed test data.
# 
# *This function should return the AUC score as a float.*

# In[10]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer

def answer_nine():
    vectorizer = TfidfVectorizer(min_df=5, ngram_range=[1,3])
    x_train_transformed_added_len = add_feature(vectorizer.fit_transform(X_train),X_train.str.len())
    x_train_transformed_added_len_digits = add_feature(vectorizer.fit_transform(X_train),
                                                       X_train.apply(lambda x:len(''.join([d for d in x if d.isdigit() ]))))
    
    x_test_transformed_added_len = add_feature(vectorizer.transform(X_test),X_test.str.len())
    x_test_transformed_added_len_digits = add_feature(vectorizer.transform(X_test),
                                                       X_test.apply(lambda x:len(''.join([d for d in x if d.isdigit() ]))))
    
    lr_classifier = LogisticRegression(C=100)
    lr_classifier.fit(x_train_transformed_added_len_digits,y_train)
    y_predicted=lr_classifier.predict(x_test_transformed_added_len_digits)
    return roc_auc_score(y_test, y_predicted)#Your answer here


# In[11]:


answer_nine()


# ### Question 10
# 
# What is the average number of non-word characters (anything other than a letter, digit or underscore) per document for not spam and spam documents?
# 
# *Hint: Use `\w` and `\W` character classes*
# 
# *This function should return a tuple (average # non-word characters not spam, average # non-word characters spam).*

# In[21]:


import re

def answer_ten():
    sum_len_nonwords_spam=0
    sum_len_nonwords_notspam=0
    number_spam=0
    number_notspam=0
    for data in spam_data.iterrows():
        if data[1]['target']==1:
            sum_len_nonwords_spam+=len(re.compile('\W').findall(data[1]['text']))
            number_spam+=1
        else:
            sum_len_nonwords_notspam+=len(re.compile('\W').findall(data[1]['text']))
            number_notspam+=1
    return (sum_len_nonwords_notspam/number_notspam, sum_len_nonwords_spam/number_spam)#Your answer here


# In[22]:


answer_ten()


# ### Question 11
# 
# Fit and transform the training data X_train using a Count Vectorizer ignoring terms that have a document frequency strictly lower than **5** and using **character n-grams from n=2 to n=5.**
# 
# To tell Count Vectorizer to use character n-grams pass in `analyzer='char_wb'` which creates character n-grams only from text inside word boundaries. This should make the model more robust to spelling mistakes.
# 
# Using this document-term matrix and the following additional features:
# * the length of document (number of characters)
# * number of digits per document
# * **number of non-word characters (anything other than a letter, digit or underscore.)**
# 
# fit a Logistic Regression model with regularization C=100. Then compute the area under the curve (AUC) score using the transformed test data.
# 
# Also **find the 10 smallest and 10 largest coefficients from the model** and return them along with the AUC score in a tuple.
# 
# The list of 10 smallest coefficients should be sorted smallest first, the list of 10 largest coefficients should be sorted largest first.
# 
# The three features that were added to the document term matrix should have the following names should they appear in the list of coefficients:
# ['length_of_doc', 'digit_count', 'non_word_char_count']
# 
# *This function should return a tuple `(AUC score as a float, smallest coefs list, largest coefs list)`.*

# In[53]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import re
from sklearn.metrics import roc_auc_score

def answer_eleven():
    
    vectorizer = CountVectorizer(min_df=5, ngram_range=[2,5],analyzer='char_wb')
    x_train_transformed_added_len_digits_nonwords=add_feature(vectorizer.fit_transform(X_train),
                                                                                       [X_train.str.len(),
                                                                                        X_train.apply(lambda x:len(''.join([d for d in x if d.isdigit() ]))),
                                                                                        X_train.apply(lambda x:len(''.join(re.compile('\W').findall(x))))])
    x_test_transformed_added_len_digits_nonwords=add_feature(vectorizer.transform(X_test),
                                                                                       [X_test.str.len(),
                                                                                        X_test.apply(lambda x:len(''.join([d for d in x if d.isdigit() ]))),
                                                                                        X_test.apply(lambda x:len(''.join(re.compile('\W').findall(x))))])
    
    '''
    x_train_transformed_added_len = add_feature(vectorizer.fit_transform(X_train),X_train.str.len())
    x_train_transformed_added_len_digits = add_feature(vectorizer.fit_transform(X_train),
                                                       X_train.apply(lambda x:len(''.join([d for d in x if d.isdigit() ]))))
    x_train_transformed_added_len_digits_nonwords = add_feature(vectorizer.fit_transform(X_train),
                                                       X_train.apply(lambda x:len(''.join(re.compile('\W').findall(x)))))
    
    x_test_transformed_added_len = add_feature(vectorizer.transform(X_test),X_test.str.len())
    x_test_transformed_added_len_digits = add_feature(vectorizer.transform(X_test),
                                                       X_test.apply(lambda x:len(''.join([d for d in x if d.isdigit() ]))))
    x_test_transformed_added_len_digits_nonwords = add_feature(vectorizer.transform(X_test),
                                                       X_test.apply(lambda x:len(''.join(re.compile('\W').findall(x)))))
    '''
    lr_classifier = LogisticRegression(C=100)
    lr_classifier.fit(x_train_transformed_added_len_digits_nonwords,y_train)
    y_predicted = lr_classifier.predict(x_test_transformed_added_len_digits_nonwords)
    
    feature_names = np.array(vectorizer.get_feature_names() + ['length_of_doc', 'digit_count', 'non_word_char_count'])
    sorted_coef_indices = lr_classifier.coef_[0].argsort()
    return (roc_auc_score(y_test, y_predicted),
           list(feature_names[sorted_coef_indices[:10]]),
           list(feature_names[sorted_coef_indices[:-11:-1]]))#Your answer here


# In[52]:


answer_eleven()


# In[ ]:




