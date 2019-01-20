#!/usr/bin/env python3

# # #
# --help
#


import sys
import os
import re
import argparse
import tempfile
import random 

# third party libraries
import fastText as ft
import numpy as np
import pandas as pd

# custom libraries
from utils import Opener

#Tip: if you want to put more than one dataset as train dataset then use => cat input_file1 input_file2 .... | train_classifier_fastext -m [name of model]

argparser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description="""\
Train a model for classifying each line of a document (using facebook FastText)
TIP:
If large training and testing datasets are used then the most economical way to
run this script is by giving testing datasets as a separate file:
    > {} -m path/to/model path/to/training.txt -t path/to/testing.txt
""".format(os.path.basename(__file__)),
    epilog='Have fun')
# positional arguments
argparser.add_argument("input_file", nargs='*', default=[sys.stdin],
                       help="""
                       input file in the format suitable for training, that is, with all
                       necessary features. Each line is then:
                       __label__NAME Orginal text enriched with features.
                       """)

argparser.add_argument("-m", "--model", type=str, required=True,
                       help="path to file under which the model will be saved.")

argparser.add_argument("-t", "--test-dataset", type=str, nargs=1,
                       help="""
                       path to the test (a.k.a. validation) dataset. If absent and option --evaluation is chosen by user , a test dataset
                       will be automatically generated from the training dataset by randomizing and grabbing
                       last 20 percent of dataset as test dataset and the remaining 80 percent of dataset is used as training dataset.
                       """)

argparser.add_argument("-e", "--epochs", type=int,
                       help="number of epochs (default=30)")
argparser.add_argument("-wg", "--word-ngrams", type=int,
                       help="length of word n-grams (default=1)")
argparser.add_argument("-lr", "--learning-rate", type=float,
                       help="learning rate (default= 0.7)")

argparser.add_argument("-k", "--kfold", nargs='?', type=int, const=5,
                       help="""
                       Run K-fold validation method (default K=5).
                       We strongly suggest using K=10 for large datasets and K=5 otherwise.
                       training and testing process repeat K times and binary model for each iteration is saving (overwriting) in the path that is specified with -m option.
                       please pay attention that the model that is saved in the path specified with -m option is the last model trained in iteration number K.
                       """)

argparser.add_argument("-pm", "--pmis", action='store_true', default=False,
                       help="""
                       Print Missed-Match lines, that is, cases where the model predicts
                       a wrong label for the line). Very useful for research.
                       """)

argparser.add_argument("-rs", "--reduce-size", action='store_true', default=False,
                       help="Quantize (reduce the size of) the model binary.")

argparser.add_argument("-em", "--evaluation-model", action='store_true', default=False,
                        help="with this option, 80 percent of input dataset will be used as training dataset and the rest 0f 20 percent will be used as test dataset for evaluation")

argparser.add_argument("-pr", "--prediction", action='store_true', default=False,
                       help="Predicting new inputs based on a pre-trained model.")

args = argparser.parse_args()

################################################################################

def measure_quality(cv_data_name):
    classifier = ft.load_model(args.model)

    cv_test=classifier.test(cv_data_name)

    cv_data_temp=open(cv_data_name).read()
    cv_data_temp=cv_data_temp.split('\n')

    del cv_data_temp[-1]
 
    labels=classifier.get_labels()
    labels.sort()
    numlab=len(labels)
    cmatrix=np.array([[0]*numlab]*numlab, dtype=int)
    for l in range(numlab):
          labels[l]=labels[l].replace("__label__","")
    labels_copy=labels.copy()    
    indc=0
    indr=0
    for i in cv_data_temp :

        temp=str(classifier.predict(i)[0][0])
        i=str(i)
        for c in range(numlab):
            indc=c if labels[c] in temp else indc
            indr=c if labels[c] in i else indr

        cmatrix[indr,indc]=cmatrix[indr,indc]+1
        if args.pmis:
                if indc!=indr:
                    print('Predicted_label='+re.sub('\_\_label\_\_', '', temp)+'\t', 'Gold_label='+re.sub('<', '\t<',re.sub('\_\_label\_\_', '', i)))

    tagsumRe=np.sum(cmatrix, axis=1, dtype=float)
    tagsumPe=np.sum(cmatrix, axis=0, dtype=float)
    tagAcc=np.zeros_like(tagsumPe)
    total_cmatrix=np.sum(cmatrix)


    domclas=max(tagsumRe)

    for i in range (numlab):
         cortag=cmatrix[i,i]
         tagsumRe[i]=round((cortag/tagsumRe[i])*100,2)
         tagsumPe[i]=round((cortag/tagsumPe[i])*100,2)
         clomun_exclude = np.delete(cmatrix, i, axis=1)
         row_exlcude = np.delete(clomun_exclude, i, axis=0)
         tagAcc[i]=round(((cortag+np.sum(row_exlcude))/total_cmatrix)*100,2)

    tagsumPe=np.reshape(tagsumPe,(numlab,1))
    tagsumRe=np.reshape(tagsumRe,(numlab,1))
    tagAcc=np.reshape(tagAcc,(numlab,1))

    cmatrix=np.append(cmatrix,tagsumPe, axis=1)
    cmatrix=np.append(cmatrix,tagsumRe, axis=1)
    cmatrix=np.append(cmatrix,tagAcc, axis=1)
    labels.append('Precision, %')
    labels.append('Recall, %')
    labels.append('Accuracy, %')
    cm=pd.DataFrame.from_items([(labels[0],cmatrix[0,])],orient='index', columns=labels)
    for j in range(1, numlab):
        cm1=pd.DataFrame.from_items([(labels[j],cmatrix[j,])],orient='index', columns=labels)
        cm=cm.append(cm1)

    pd.set_option('display.expand_frame_repr',False)
    cm[labels_copy]=cm[labels_copy].astype(int)
    
    print()
    print('-----------------------------------  Confusion Matrix  -----------------------------------')
    print()
    print(cm)
    print('------------------------------------------------------------------------------------------')
    print('Precision = '+ str(round(cv_test[1]*100,2))+' %')
    print('Recall = '+ str(round(cv_test[2]*100,2))+' %')
    print('F1-Score = '+str(round(((2*cv_test[1]*cv_test[2])/(cv_test[1]+cv_test[2]))*100,2))+' %')
    print('Baseline = '+ str(round((domclas/cv_test[0])*100,2))+' %')
    print('Number of Test-Examples = '+ str(cv_test[0]))
    return (cm)

################################################################################
def k_fold(train_set,test_set):
    k=args.kfold

    datatrain = open(train_set, "rt").readlines()
    datatest = open(test_set, "rt").readlines()
    data=datatrain+datatest
    numfold=round(len(data)*(1/k))

    for j in range(k):
        test_data=data[numfold*j:numfold*(j+1)]
        train_data=data[:]
        del train_data[numfold*j:numfold*(j+1)]

        fd = get_tempfile()
        fd.writelines(train_data)
        fd.close()
        train_fn = fd.name

        fd = get_tempfile()
        fd.writelines(test_data)
        fd.close()
        test_fn = fd.name

        train_model(train_fn)
        cmatx = measure_quality(test_fn)

        perc=cmatx[['Precision, %']]
        recall=cmatx[['Recall, %']]
        accuracy = cmatx[['Accuracy, %']]
        if j ==0 :
            per=perc
            rcal=recall
            acc=accuracy
        else:
            per=pd.concat([per,perc],axis=1,join_axes=[per.index])
            rcal=pd.concat([rcal,recall],axis=1,join_axes=[rcal.index])
            acc = pd.concat([acc, accuracy], axis=1, join_axes=[acc.index])

    permaen=per.mean(axis=1)
    rcalmean=rcal.mean(axis=1)
    accmean = acc.mean(axis=1)
    perst=per.std(axis=1)
    rcalst=rcal.std(axis=1)
    accst = acc.std(axis=1)


    persum=pd.concat([permaen,perst],axis=1, join_axes=[permaen.index])
    recallsum=pd.concat([rcalmean,rcalst],axis=1, join_axes=[rcalmean.index])
    accuracysum = pd.concat([accmean, accst], axis=1,
                          join_axes=[accmean.index])
    kfold_sum=(pd.concat([persum,recallsum,accuracysum], axis=1, join_axes=[persum.index] )).round(2)
    kfold_sum.columns = ['Precision(%), AVG', 'Precision(%), STD',
                         'Recall(%), AVG', 'Recall(%), STD', 'Accuracy(%), AVG', 'Accuracy(%), STD']
    print()
    print('----------------------------------------------------  K-Fold  -----------------------------------------------------')
    print()
    print(kfold_sum)
    print('-------------------------------------------------------------------------------------------------------------------')

################################################################################

def train_model(training_set):
    epoch  = args.epochs         or 30
    wngram = args.word_ngrams    or 3
    lr     = args.learning_rate  or 0.65

    model = ft.train_supervised(input = training_set, epoch = epoch,
                                wordNgrams = wngram, loss = "softmax", lr = lr)
    if args.reduce_size:
        model.quantize()

    model.save_model(args.model)

################################################################################
# Logic:
#
# training | if testset given   | if testset not given
# ---------|--------------------|--------------------------
# stdin    | dump to train file | dump to train/test files
# 1 file   | reuse              | dump to train/test files
# 2+ files | dump to train file | dump to train/test files

my_tmp_files = []

def get_datasets():
    """
    Returns names of the files containing training and testing (aka validation) datasets.
    """
    train_fn = test_fn = None
 
    # use given testset
    if args.test_dataset:
        test_fn = args.test_dataset[0]
        if len(args.input_file) == 1 and type(args.input_file[0]) is str:
            train_fn = args.input_file[0]

    if not train_fn:
        train_fd = get_tempfile()
        for infile in args.input_file:
            append_file_data_to_other_file(infile, train_fd)
        train_fn = train_fd.name
        train_fd.close()

    # separate dataset set into training and testing if no testset was given
    if not test_fn:
        data = open(train_fn, "rt").readlines()
        # randomly shuffle lines
        random.shuffle(data)
        split = round(len(data) * 0.8)
        # print("Dataset split among training/testing: {}/{}".format(len(data[:split]), len(data[split:])))

        # training dataset
        fd = get_tempfile()
        
        fd.writelines(data[:split])
        train_fn = fd.name

        # testing (aka validation) dataset
        fd = get_tempfile()
        fd.writelines(data[split:])
        test_fn = fd.name

    # print("Training / Testing: {} / {}".format(train_fn, test_fn))

    return (train_fn, test_fn)

def append_file_data_to_other_file(src, trg_fd):
    fnr = 0
    with Opener(src) as src_fd:
        for line in src_fd:
            if line.strip(): # if non-empty
                trg_fd.write(line)
                fnr += 1
    return fnr

def get_tempfile():
    #tempfile.NamedTemporaryFile(mode='w+b', buffering=None, encoding=None, newline=None, suffix=None, prefix=None, dir=None, delete=True)

    fd = tempfile.NamedTemporaryFile(mode="w+t", delete=False)
    my_tmp_files.append(fd.name)
    return fd

def cleanup():
    """
    Remove temporary files that we have generated.
    """
    for f in my_tmp_files:
        # print("Removing tmp file: {}".format(f))
        os.remove(f)
################################################################################
def line_prediction(classifier, line):
    klasses = classifier.predict(line)  # => [[klass]]
    klass = (klasses[0][0]).replace("__label__", "")
    return klass

################################################################################
def line_classifier(fname,model):
    classifier = ft.load_model(model)
    with Opener(fname) as f:
        klass_list=[]
        for line in f:
            klass = line_prediction(classifier, line)
            klass_list.append('<{}>'.format(klass))
        return klass_list
################################################################################
def pooling_results(prediction,text):
    for i in range(len(prediction)):
        print('{}\t{}'.format(prediction[i], text[i]))
################################################################################

def main():
    if args.prediction:
        prediction = line_classifier(args.input_file, args.model)
        plain_text = open(args.input_file).readlines()
        pooling_results(prediction,plain_text)
    elif args.evaluation_model:
        (train_set, test_set) = get_datasets()
        train_model(train_set)
        measure_quality(test_set)
    elif args.kfold != None:
        (train_set, test_set) = get_datasets()
        k_fold(train_set, test_set)
    else:
        train_model(args.input_file[0])


    cleanup()

if __name__ == '__main__':
    main()
