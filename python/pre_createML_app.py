import pandas as pd, numpy as np

# Read input files
train = pd.read_csv('../data/train.csv')

test = pd.read_csv('../data/test.csv')
test_labels = pd.read_csv('../data/test_labels.csv')
test = test.join(test_labels.set_index('id'), on='id')
test = test.where(test["toxic"] != -1).dropna()
test["toxic"] = test["toxic"].astype(int)

# Clean up texts
train.comment_text = train.comment_text.str.replace('\n', ' ')
train.comment_text = train.comment_text.str.replace('\\', '')
test.comment_text = test.comment_text.str.replace('\n', ' ')
test.comment_text = test.comment_text.str.replace('\\', '')

# Export to folders
classes = ["toxic", "non-toxic"]
train_txts = [train[train["toxic"] == 1]["comment_text"], train[train["toxic"] == 0]["comment_text"]]
test_txts = [test[test["toxic"] == 1]["comment_text"], test[test["toxic"] == 0]["comment_text"]]
datasets = {"train": train_txts, "test":test_txts}
base_path = "../JigsawToxicityCreateMLAppBeta3/Data/" 

for key in datasets:
	for idx1, classification in enumerate(classes):
		for idx2, txt in enumerate(datasets[key][idx1]):
			with open(base_path + "/{0}/".format(key) + classification + "/{0}.txt".format(idx2), 'a') as f:
				f.write(txt) 


	




