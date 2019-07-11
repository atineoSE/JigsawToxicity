import pandas as pd, numpy as np
import re

# Set display options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Read input files
train = pd.read_csv('../data/train.csv')
print(train.head())

train['healthy'] = 1-train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].max(axis=1)
train['only_toxic'] = 1-train[['healthy', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].max(axis=1)
train['only_severe_toxic'] = 1-train[['toxic', 'healthy','identity_hate']].max(axis=1)
train['only_identity_hate'] = 1-train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'healthy']].max(axis=1)

toxic_sample = train[train["only_toxic"] ==  1][10:20]
severe_toxic_sample = train[train["only_severe_toxic"] ==  1][10:20]
identity_hate_sample = train[train["only_identity_hate"] ==  1][10:20]

print("TOXIC SAMPLES:")
for s in toxic_sample["comment_text"].to_numpy():
    print(s)
    print("******* END ***********")

print("SEVERE TOXIC SAMPLES:")
for s in severe_toxic_sample["comment_text"].to_numpy():
    print(s)
    print("******* END ***********")

print("IDENTITY HATE SAMPLES:")
for s in identity_hate_sample["comment_text"].to_numpy():
    print(s)
    print("******* END ***********")