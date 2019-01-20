# Fasttext-TextClassifier
* A Powerful text classifier (NLP) using FastText as engine  with several Cross Validation options and etc.
* you can train a model for classifying each/word line of a document (using Facebook FastText)

# Requirement

* This script uses [Fast Text](https://fasttext.cc/docs/en/support.html/). 

# Options

* __Epoch__: '-e' number of epochs (default=30)
* __wordNgram__: '-wg' length of word n-grams (default=1)
* __Learning Rate__: '-lr' learning rate (default= 0.7)
* __Kfold__: '-k' Run K-fold cross validation method (default K=5).
	* __Tips__:it is strongly suggested using K=10 for large data-sets and K=5 otherwise.
* __Missed-Matched__: '-pm' Print Missed-Match lines, that is, cases where the model predicts wrong label for the line). Very useful for research.
* __Reduce Size__: '-rs' Quantize (reduce the size of) the binary model.
* __Evaluation Model__: '-em' with this option, 80 percent of input data-set will be used as training data-set and the rest 0f 20 percent will be used as test data-set for evaluation
* __Prediction__: '-pr- Predicting new lines/words based on a pre-trained model.
		


# Tip

* The Train data-set that you wish to use muss have the following format:
	
	* For specifying __Target Label__ of line/word please use `__label__`[space][line/word]
* Example: `__positive__` This dog is so cute" ==> "__This dog is so cute__" Class name is __positive__

* For Classifying new lines/words, A path to the binary pre-trained model must be specified by '-m' option 


