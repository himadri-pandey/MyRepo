# -*- coding: utf-8 -*-
"""
Created on Thu May  9 14:44:24 2019

@author: himadri
"""
#Sentiment analysis in text
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#importing the dataset in a tab separated file
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)
#cleaning the text:
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer #PorterStemmer class is imported here for stemming

corpus = [] #in NLP is a collection of text of same type, so we name it here like this
for i in range(0,1000): #upper bound=1000 here as it is excluded in range fn
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i] ) #removed , . etc
    review = review.lower()    #substituting all capitals with smalls 
    review = review.split() #splitting into diff words so that the string becomes a list
#review = [word for word in review if not word in set(stopwords.words('english'))]
#stemming done next: keeping only root of the keywords:
#stemming is done in order to avoid too much sparsity (check out details in Udemy Q&A)
    ps = PorterStemmer() #creating an object of PorterStemmer class
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
#join all variables in the list called review to get a string called review, post cleaning operations
    review = ' '.join(review) #adding a space before joining keeps the words separated
#for doing this for all the 1000 reviews, we would of course need a for loop
    corpus.append(review)

#creating bag of words model:
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500) #object creation, 1500 most frequent words are included in the spars matrix
X = cv.fit_transform(corpus).toarray() #creation of sparse matrix
y = dataset.iloc[:,1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

accuracy = (55 + 91)/200
