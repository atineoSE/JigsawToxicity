# -*- coding: utf-8 -*-

import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import re, string
#import joblib
from sklearn.externals import joblib
import sys

# Set display options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Process command line params
ref_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
cols = ref_cols
#cols = ['toxic']

custom_tokenizer = False

num_params = len(sys.argv)
limit_row = None
if num_params < 2:
    print("No arguments are passed. Using default.")
else:
    if num_params > 1:
        model_label = sys.argv[1]
        if model_label in ref_cols:
            print("Proceeding with model label " + model_label)
            cols = [model_label]
        else:
            print("ERROR. Model label {0} not recognized.".format(model_label))
            exit(-1)
    if num_params > 2:
        limit_row = int(sys.argv[2])
        print("Reducing times by choosing first {0} elements for training and scoring.".format(limit_row))

# Read input files
input = pd.read_csv('../data/train.csv')
print("*******************************************")
print("INPUT DATASET: DESCRIBE, ORIGINAL")
print("*******************************************")
print(input.describe())
print("*******************************************")
print("INPUT DATASET: HEAD")
print("*******************************************")
print(input.head())
# Split input intro training and test set
train, test = train_test_split(input, test_size=0.2)

score = pd.read_csv('../data/test.csv')
score_labels = pd.read_csv('../data/test_labels.csv')
score = score.join(score_labels.set_index('id'), on='id')
score = score.where(score[cols[0]] != -1).dropna()
print("*******************************************")
print("SCORE DATASET: DESCRIBE")
print("*******************************************")
print(score.describe())
print("*******************************************")
print("SCORE DATASET: HEAD")
print("*******************************************")
print(score.head())

# Plot histogram of length distribution of comment text
input_lens = input.comment_text.str.len()
print("Input lengths: mean = {0}, std = {1}, max = {2}". format(input_lens.mean(), input_lens.std(), input_lens.max()))
input_lens.hist()
score_lens = score.comment_text.str.len()
print("Score lengths: mean = {0}, std = {1}, max = {2}". format(score_lens.mean(), score_lens.std(), score_lens.max()))
score_lens.hist()
plt.savefig("input_vs_score_text_length_histogram.png")

# Limit, if needed for sample execution
train = train[0:limit_row]
test = test[0:limit_row]
score = score[0:limit_row]
print("Dimensions in training set: {0}".format(train.shape))
print("Dimensions in test set: {0}".format(test.shape))
print("Dimensions in score set: {0}".format(score.shape))

if custom_tokenizer:
    # Prepare text for tokenization
    # Remove special characters and substitute them by the same term surrounded by space
    # e.g. "car.man" -> " car  man " -> ["car", "man"]
    # this is needed to make a clean bag of words matrix

    print("Using punctuation characters: {0}".format(string.punctuation))
    re_tok = re.compile(r'([!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~“”¨«»®´·º½¾¿¡§£₤‘’])')
    def tokenize(s): return re_tok.sub(r' \1 ', s).split()

    #Create bag of words matrix
    vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,
                   min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
                   smooth_idf=1, sublinear_tf=1)
else:
    vec = TfidfVectorizer()

# Learn vocabulary and idf from training set, generate document-term matrix
train_term_doc = vec.fit_transform(train["comment_text"])

print("train_term_doc shape")
print(train_term_doc.shape)

print("feature names sample (matrix columns, bag of words)")
series = pd.Series(vec.get_feature_names())
print(series.sample(50))
series.to_csv("./Logs/train_terms.csv")

# Get document-term matrix for test and score sets
test_term_doc = vec.transform(test["comment_text"])
score_term_doc = vec.transform(score["comment_text"])

# Function to obtain the Naive-Bayes probabilities
def pr(x, y_i, y):
    p = x[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)

# Function to obtain a logistic regression model
def get_model(x, y):
    y = y.values
    r = np.log(pr(x,1,y) / pr(x,0,y))
    m = LogisticRegression(C=4, dual=True, multi_class="ovr")
    x_nb = x.multiply(r)
    return m.fit(x_nb, y), r

for i, label in enumerate(cols):
    print('Creating model for ', label)

    #### Get model

    # r is the log-count ratio for every feature (i.e. every word/column in the matrix)
    # we take the log of the ratio because then you can add instead of multiply
    # which is better for close to 0 probabilities due to limited floating-point precision
    model, r = get_model(train_term_doc, train[label])

    # we need to apply the same transformation to the input as the training set
    # i.e. multiply by the r ratio
    test_model_input = test_term_doc.multiply(r)
    score_model_input = score_term_doc.multiply(r)

    #### Get the predictions:
    test_predictions = model.predict(test_model_input)
    score_predictions = model.predict(score_model_input)

    # If you need the probabilities (as for the Kaggle competition):
    # output of predict_proba is a matrix of 2 columns:
    # column 0 is the probability that the observation does not belong to the class
    # column 1 is the probability that is belongs to the class
    # test_probabilities = model.predict_proba(test_model_input)
    # score_probabilities = model.predict_proba(score_model_input)[:,1]

    #### Evaluate model
    # score takes the input matrix and compares with a vector of the classification (class *not* probability)
    # if comparing with a vector of probabilities, then we get error:
    # Can't Handle mix of binary and continuous target
    mean_accuracy_test = model.score(test_model_input, test[label])
    mean_accuracy_score = model.score(score_model_input, score[label])
    print("Mean accuracy over test set for {0}: {1}".format(label, mean_accuracy_test))
    print("Mean accuracy over score set for {0}: {1}".format(label, mean_accuracy_score))

    test_confusion_matrix = confusion_matrix(test[label], test_predictions)
    score_confusion_matrix = confusion_matrix(score[label], score_predictions)
    print("Confusion matrix over test set for {0}:".format(label))
    print(test_confusion_matrix)
    print("Confusion matrix over score set for {0}:".format(label))
    print(score_confusion_matrix)

    #### Save model
    outputFile = './Models/{0}.pkl'.format(label)
    print("Saving model to {0}".format(outputFile))
    joblib.dump(model, outputFile, compress=9)