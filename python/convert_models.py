from coremltools.converters.sklearn import convert
from sklearn.linear_model import LogisticRegression
#import joblib
from sklearn.externals import joblib

model_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

for label in model_labels:
    sklearn_model = joblib.load("./Models/{0}.pkl".format(label))
    coreml_model = convert(sklearn_model)
    coreml_model.save("./Models/{0}SK.mlmodel".format(label))
