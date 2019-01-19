# Fasttext-TextClassifier
A Powerful text classifier (NLP) using FastText as engine  with several Cross Validation options and etc.
you can train a model for classifying each line of a document (using facebook FastText)

# Requirement

This script uses [Fast Text](https://fasttext.cc/docs/en/support.html/). 

# Options

* Epoch: '-e' number of epochs (default=30)
* wordNgram: '-wg' length of word n-grams (default=1)
* Learning Rate: '-lr' learning rate (default= 0.7)
* Kfold: '-k' Run K-fold cross validation method (default K=5).
	* Tips:it is strongly suggested using K=10 for large datasets and K=5 otherwise.
* Missed-Matched: '-pm' Print Missed-Match lines, that is, cases where the model predicts wrong label for the line). Very useful for research.
* Reduce Size: '-rs' Quantize (reduce the size of) the binary model.
* Evaluation Model: '-em' with this option, 80 percent of input dataset will be used as training dataset and the rest 0f 20 percent will be used as test dataset for evaluation


