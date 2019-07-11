# JigsawToxicity
Project for ML on the toxic texts wikipedia Kaggle dataset.

## Getting the data

The original datasets can be obtained from the [corresponding Kaggle competition](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge), by accepting terms and conditions, then downloading, and adding to the `data` subdirectory over the root of this repo to be found by the python scripts.

Unlike most Kaggle competitions, the labels for the test set used to evaluate the competitors was released once the competition was finished, which means that we can use the test set as we would normally do in a classical model training and evaluation workflow. That requires that we combine the labels from the `test_labels.csv` into the rows in `test.csv`. This is done when generating the data sets for the experiments below in the python scripts.

## Repo structure

The following projects are included in this repo:

- [JigsawToxicityCreateML](JigsawToxicityCreateML): a Xcode macOS project with a script to create text classification models by using the Create ML API in code (i.e. using `MLTextClassifier`).
- [JigsawToxicityCreateMLAppBeta3](JigsawToxicityCreateMLAppBeta3): a Create ML app project for the same data, available with Xcode 11 beta 3 and macOS Catalina beta.
- [JigsawToxicityInference](JigsawToxicityInference): a Xcode iOS project to test performance of inference on device. This one uses a subset of the available test texts, to avoid very long execution times.
- [python](python): python scripts for 
  - [generating data sets for the Create ML script](python/pre.py) and [for the Create ML app](python/pre_createML_app.py), and 
  - to [generate scikit learn models (logistic regressor)](python/kaggle_kernel_NB-SVM.py) as per the solution by [Jeremy Howard](https://www.kaggle.com/jhoward) in [Kaggle](https://www.kaggle.com/jhoward/nb-svm-strong-linear-baseline).