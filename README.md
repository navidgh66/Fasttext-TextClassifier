# Fasttext-TextClassifier
* A Powerful text classifier (NLP) using FastText as engine  with several Cross Validation options and etc.
* you can train a model for classifying each line of a document (using facebook FastText)

# Requirement

* This script uses [Fast Text](https://fasttext.cc/docs/en/support.html/). 

# Options

* __Epoch__: '-e' number of epochs (default=30)
* __wordNgram__: '-wg' length of word n-grams (default=1)
* __Learning Rate__: '-lr' learning rate (default= 0.7)
* __Kfold__: '-k' Run K-fold cross validation method (default K=5).
	* __Tips__:it is strongly suggested using K=10 for large datasets and K=5 otherwise.
* __Missed-Matched__: '-pm' Print Missed-Match lines, that is, cases where the model predicts wrong label for the line). Very useful for research.
* __Reduce Size__: '-rs' Quantize (reduce the size of) the binary model.
* __Evaluation Model__: '-em' with this option, 80 percent of input dataset will be used as training dataset and the rest 0f 20 percent will be used as test dataset for evaluation


