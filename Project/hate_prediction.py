import pandas as pd
import re
import numpy as np

import warnings
warnings.filterwarnings("ignore")

import nltk
import contractions
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
stopwords = nltk.corpus.stopwords.words("english")

from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix, recall_score, precision_score, roc_curve

from sklearn.feature_extraction.text import TfidfVectorizer # initialises an array of frequencies of words.
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier #supervised-learning
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from mlxtend.classifier import StackingClassifier

import pickle


df = pd.read_csv('comments.csv')

def preprocess(tweet):  
    
    # removal of extra spaces
    regex_pat = re.compile(r'\s+')
    tweet_space = tweet.str.replace(regex_pat, ' ')

    # removal of @name (user names)
    regex_pat = re.compile(r'@[\w\-]+')
    tweet_name = tweet_space.str.replace(regex_pat, '')
    
    # removal of capitalization
    tweet_lower = tweet_name.str.lower()
    
    # removal of URLs
    giant_url_regex =  re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
            '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    tweets = tweet_lower.str.replace(giant_url_regex, '')
    
    # removal of punctuations and numbers
    punc_remove = tweets.str.replace("[^a-zA-Z]", " ")
    
    # remove whitespace with a single space
    new_tweet= punc_remove.str.replace(r'\s+', ' ')
    
    # remove leading and trailing whitespace
    new_tweet= new_tweet.str.replace(r'^\s+|\s+?$','')    
    
    # tokenizing
    tokenized_tweet = new_tweet.apply(lambda x: x.split())
        
    # removal of stopwords
    tokenized_tweet=  tokenized_tweet.apply(lambda x: [item for item in x if item not in stopwords])
    
    # stemming
    tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) 
        
    for i in range(len(tokenized_tweet)):
        tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
        tweets_p= tokenized_tweet
    
    return tweets_p


# Remove contractions
df['Comment'] = df['Comment'].apply(lambda x: contractions.fix(x))


# Pre-process data
comments= df.Comment
processed_comments = preprocess(comments)   

df['Comment'] = processed_comments

X = df["Comment"]
y = df["final_isHate"]

# Spliting of dataset into test and train set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

vectorizer = TfidfVectorizer(max_df=0.8, min_df=6, max_features=3000, stop_words=stopwords, ngram_range=(1,1))

X_train_cv1 = vectorizer.fit_transform(X_train) 
X_test_cv1  = vectorizer.transform(X_test)


# Random Forest
model_rf = RandomForestClassifier(random_state=0, n_estimators=140)    
model_rf.fit(X_train_cv1, y_train) 

predictions_rf = model_rf.predict(X_test_cv1) 


# XGBoost
model_xgb = XGBClassifier(base_score=0.35)
model_xgb.fit(X_train_cv1, y_train) 

predictions_xgb = model_xgb.predict(X_test_cv1) 


# Naive Bayes
model_nb = MultinomialNB()
model_nb.fit(X_train_cv1, y_train)

predictions_nb = model_nb.predict(X_test_cv1) 


#Logistic Regression
model_lr = LogisticRegression()
model_lr.fit(X_train_cv1, y_train) 

predictions_lr = model_lr.predict(X_test_cv1) 


# Stacking: Ensemble algorithm
clf_stack = StackingClassifier(classifiers =[model_rf, model_xgb, model_nb], meta_classifier = model_lr, use_probas = True, use_features_in_secondary = True)
clf_stack.fit(X_train_cv1, y_train)  

predictions_stack = clf_stack.predict(X_test_cv1) 
print(accuracy_score(y_test, predictions_stack))
print(f1_score(y_test, predictions_stack))

# Saving model to disk
pickle.dump(clf_stack, open('model.pkl','wb'))

