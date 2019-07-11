# -*- coding: utf-8 -*-

import pandas as pd, numpy as np
import re, string
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib


train = pd.read_csv('./sample_train.csv')
print("train data")
print(train)

test = pd.read_csv('./sample_test.csv')
print("test data")
print(test)

# Prepare text for tokenization
# Remove special characters and substitute them by the same term surrounded by space
# e.g. "car.man" -> " car  man " -> ["car", "man"]
# this is needed to make a clean bag of words matrix

print("Using punctuation characters: {0}".format(string.punctuation))
re_tok = re.compile(r'([!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()

# Create bag of words matrix
# vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,
#                min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
#                smooth_idf=1, sublinear_tf=1)


vec = TfidfVectorizer()

# Learn vocabulary and idf from training set, generate document-term matrix
train_term_doc = vec.fit_transform(train["comment_text"])
print("train_term_doc shape")
print(train_term_doc.shape)

print("feature names sample (matrix columns, bag of words)")
series = pd.Series(vec.get_feature_names())
print(series)
series.to_csv("./Logs/sample_terms.csv")


# Get document-term matrix for test and score sets
test_term_doc = vec.transform(test["comment_text"])
print("test_term_doc shape")
print(test_term_doc.shape)

# Function to obtain the Naive-Bayes probabilities
def pr(x, y_i, y):
    p = x[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)

# Function to obtain a logistic regression model
def get_model(x, y):
    print("x")
    print(x)
    y = y.values
    print("y")
    print(y)
    r = np.log(pr(x,1,y) / pr(x,0,y))
    print("r = {0}".format(r))
    m = LogisticRegression(C=4, dual=True, multi_class="ovr")
    x_nb = x.multiply(r)
    return m.fit(x_nb, y), r

 #### Get model

# r is the log-count ratio for every feature (i.e. every word/column in the matrix)
# we take the log of the ratio because then you can add instead of multiply
# which is better for close to 0 probabilities due to limited floating-point precision
model, r = get_model(train_term_doc, train['toxic'])

# we need to apply the same transformation to the input as the training set
# i.e. multiply by the r ratio
test_model_input = test_term_doc.multiply(r)

#### Get the predictions:
test_predictions = model.predict(test_model_input)
print("test_predictions")
print(test_predictions)

#### Save model
outputFile = './Models/sample_model.pkl'
print("Saving model to {0}".format(outputFile))
joblib.dump(model, outputFile, compress=9)