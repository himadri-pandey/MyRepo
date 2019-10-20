# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 11:40:17 2019

@author: himadri
"""
#Building a spam filter in Python and Apache Spark
#start a pyspark session
import findspark
findspark.init()

from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('NLP').getOrCreate()

#read the data
data = spark.read.csv('SMSSpamCollection', inferSchema = True, sep = '\t')

data.printSchema()
#inspect the data
data.show()
#rename the columns appropriately
data = data.withColumnRenamed('_c0', 'class').withColumnRenamed('_c1', 'text')
data.show()
#clean and prepare / preprocess data
from pyspark.sql.functions import length
data = data.withColumn('length', length(data['text']))
data.show()
#feature engineering to analyze the data
#check if there is any difference between a spam and a ham message length
data.groupBy('class').mean().show()
#takeaway: a spam message in this case is LONGER !
#import relevant NLP tools
from pyspark.ml.feature import Tokenizer, StopWordsRemover, StringIndexer, IDF, CountVectorizer

tokenizer = Tokenizer(inputCol = 'text', outputCol = 'tokenText')
stopremove = StopWordsRemover(inputCol = 'tokenText', outputCol = 'stop_token')
count_vec = CountVectorizer(inputCol = 'stop_token', outputCol = 'cvec')
idf = IDF(inputCol = 'cvec', outputCol = 'tf_idf')
ham_spam_to_numeric = StringIndexer(inputCol = 'class', outputCol = 'label') #converts class strings to numeric equivalents
 
#after this, create a vector assembly of features and elements in the Spark compatible format
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vector

clean_up = VectorAssembler(inputCols = ['tf_idf', 'length'], outputCol = 'features')
#here you are assembling a vector which has one input with all NLP tools, the other with length info of input strings
#then the assembly outputs the feature values
#sometimes, having the input text length info may improve our models

#Now we build the typical Naive Bayes classification model, very popular in this
from pyspark.ml.classification import NaiveBayes
nb = NaiveBayes()
#since we have so many steps and dense feature vectors, we use a pipeline
from pyspark.ml import Pipeline

#define the pipeline stages
data_prep_pipe = Pipeline(stages = [ham_spam_to_numeric, tokenizer, stopremove, count_vec, idf, clean_up])

cleaner = data_prep_pipe.fit(data)
#so that, finally clean data:
clean_data = cleaner.transform(data)

#Training and testing of the Naive Bayes model:
clean_data.columns
clean_data = clean_data.select('label', 'features')
#splitting the dataset into training and test sets
train, test = clean_data.randomSplit([0.7, 0.3])

spam_detector = nb.fit(train)
#testing the model
test_results = spam_detector.transform(test)

test_results.show()

#to compare the performance, let's use multi class classification evaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

acc_eval = MulticlassClassificationEvaluator()

accuracy = acc_eval.evaluate(test_results)

print('accuracy of our Naive Bayes NLP model is:', accuracy*100, '%')
