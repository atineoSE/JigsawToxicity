import pandas as pd, numpy as np
import re

# Set display options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Read input files
train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')
print("*******************************************")
print("TRAIN DATASET: DESCRIBE, ORIGINAL")
print("*******************************************")
print(train.describe())
print("*******************************************")
print("TRAIN DATASET: HEAD, ORIGINAL")
print("*******************************************")
print(train.head())

# Clean up texts
train.comment_text = train.comment_text.str.replace('\n', ' ')
train.comment_text = train.comment_text.str.replace('\\', '')
test.comment_text = test.comment_text.str.replace('\n', ' ')
test.comment_text = test.comment_text.str.replace('\\', '')
print("*******************************************")
print("TRAIN DATASET: HEAD, CLEANED UP")
print("*******************************************")
print(train.head())

# Export to CSV file
train.to_csv('../JigsawToxicityCreateML/JigsawToxicityCreateML/Data/train_pre.csv', sep=",", index=False)
test.to_csv('../JigsawToxicityCreateML/JigsawToxicityCreateML/Data/test_pre.csv', sep=",", index=False)
test["comment_text"].to_csv('../JigsawToxicityInference/JigsawToxicityInference/Data/inputs_for_classification.csv', index=False, header=False)

