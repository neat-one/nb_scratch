#Import the relevant packages.
# import nltk
from distutils.log import Log
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize
import math
from collections import Counter
from nltk.corpus import stopwords
import random

random.seed(42)

# input_path = input('Please enter the file path to your corpus folder: ')
input_path = r'C:/Users/neato/AppData/Roaming/nltk_data/corpora/movie_reviews/'

# input_train = input("Please enter which folds you would like to train. Type 'train fold1 fold2' if you would like to train on the first two folds: ")
input_train = 'train fold1 fold2'

# input_test = input("Please enter the fold you would like to use for validation. Type 'test fold3' if you would like to train on the third fold: ")
input_test = 'test fold3'

def split_folds():
    '''Takes in user input for which folds to include in training and testing.
    Returns the folds to train and the indices of the boundaries for the folds.'''

    fold_dict = {}

    review_len = int(len(movie_reviews.fileids()) / 2)

    fold1_2_boundary = int(review_len/3)
    fold2_3_boundary = (int(review_len/3)*2)

    fold_dict['fold1_pos'] = [movie_reviews.words(f) for f in list(movie_reviews.fileids('pos'))[:fold1_2_boundary]]
    fold_dict['fold2_pos'] = [movie_reviews.words(f) for f in list(movie_reviews.fileids('pos'))[fold1_2_boundary:fold2_3_boundary]]
    fold_dict['fold3_pos'] = [movie_reviews.words(f) for f in list(movie_reviews.fileids('pos'))[fold2_3_boundary:]]

    fold_dict['fold1_neg'] = [movie_reviews.words(f) for f in list(movie_reviews.fileids('neg'))[:fold1_2_boundary]]
    fold_dict['fold2_neg'] = [movie_reviews.words(f) for f in list(movie_reviews.fileids('neg'))[fold1_2_boundary:fold2_3_boundary]]
    fold_dict['fold3_neg'] = [movie_reviews.words(f) for f in list(movie_reviews.fileids('neg'))[fold2_3_boundary:]]

    return fold_dict
    

def get_vocab_train(input_train='train fold1 fold2', input_test='test fold3'):
    '''Put the words from the review files into a dictionary with the vocab 
    name as the key and the vocabulary list as the values.'''

    vocab_dict = {}
    vocab_dict['pos'] = []
    vocab_dict['neg'] = []

    fold_dict = split_folds()

    if 'fold1' in input_train.split()[1:]:
        vocab_dict['pos'].append(fold_dict['fold1_pos'])
        vocab_dict['neg'].append(fold_dict['fold1_neg'])
    if 'fold2' == input_train.split()[1:]:
        vocab_dict['pos'].append(fold_dict['fold2_pos'])
        vocab_dict['neg'].append(fold_dict['fold2_neg'])
    if 'fold3' == input_train.split()[1:]:
        vocab_dict['pos'].append(fold_dict['fold3_pos'])
        vocab_dict['neg'].append(fold_dict['fold3_neg'])

    train_doc_len_pos = len([doc for fold in vocab_dict['pos'] for doc in fold])
    train_doc_len_neg = len([doc for fold in vocab_dict['neg'] for doc in fold])

    vocab_dict['pos'] = [word.lower() for fold in vocab_dict['pos'] for doc in fold for word in doc]
    vocab_dict['neg'] = [word.lower() for fold in vocab_dict['neg'] for doc in fold for word in doc]

    vocab_dict['full_vocab'] = vocab_dict['pos'] + vocab_dict['neg']

    vocab_dict['pos'] = Counter(vocab_dict['pos'])
    vocab_dict['neg'] = Counter(vocab_dict['neg'])
    vocab_dict['full_vocab'] = Counter(vocab_dict['full_vocab'])

    return vocab_dict, train_doc_len_pos, train_doc_len_neg


def train_nb():
    vocab_dict, train_doc_len_pos, train_doc_len_neg = get_vocab_train()
    total_doc_len = train_doc_len_pos + train_doc_len_neg

    log_prior = {}

    log_prior['pos'] = math.log2(train_doc_len_pos / total_doc_len)
    log_prior['neg'] = math.log2(train_doc_len_neg / total_doc_len)

    log_likelihood = {}
    log_likelihood['pos'] = {}
    log_likelihood['neg'] = {}

    vocab = vocab_dict['full_vocab'].keys()

    for current_class in movie_reviews.categories():

        word_count_class = sum([count for count in vocab_dict[current_class].values()])

        for word in vocab:
            log_likelihood[current_class][word] = math.log2(
                #maybe need to add if else for whether it is in vocab_dict[current_class][word] (0 if not)?
                (vocab_dict[current_class][word] + 1) / (word_count_class + len(vocab))
            )


    return log_prior, log_likelihood, vocab_dict


def test_nb(file_sent):
    true_sent = file_sent[1]
    test_file = file_sent[0]
    log_prior, log_likelihood, vocab_dict = train_nb()
    
    vocab = list(vocab_dict['full_vocab'].keys())

    prob_classes = {}

    for current_class in movie_reviews.categories():
        sum_prob = 0
        for word in movie_reviews.words(test_file):
            word = word.lower()
            if word in vocab:
                sum_prob += log_likelihood[current_class][word]
        sum_prob += log_prior[current_class]
        prob_classes[current_class] = sum_prob

    pred_class = max(prob_classes['pos'], prob_classes['neg'])
    pred_sent = list(prob_classes.keys())[list(prob_classes.values()).index(pred_class)]

    print(pred_sent, true_sent)

    return pred_sent, true_sent

def evaluate_model(pred_actual):
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0

    for pred, true in pred_actual:
        if (true == pred) & (true == 'neg'):
            true_neg += 1
        elif (true == pred) & (true == 'pos'):
            true_pos += 1
        elif (true != pred) & (true == 'neg'):
            false_pos += 1
        elif (true != pred) & (true == 'pos'):
            false_neg += 1

    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_neg + false_pos)
    f1 = (2*(precision*recall)) / (precision + recall)

    return precision, recall, accuracy, f1

def test_fold(input_test='test fold3'):
    test_fold = input_test.split()[1]
    review_len = int(len(movie_reviews.fileids()) / 2)

    file_sent = {}
    file_sent['pos'] = [(f, 'pos') for f in movie_reviews.fileids('pos')]
    file_sent['neg'] = [(f, 'neg') for f in movie_reviews.fileids('neg')]

    if test_fold == 'fold1':
        fold1_2_boundary = int(review_len/3)
        file_sent['pos'] = file_sent['pos'][:fold1_2_boundary]
        file_sent['neg'] = file_sent['neg'][:fold1_2_boundary]

    elif test_fold == 'fold2':
        fold1_2_boundary = int(review_len/3)
        fold2_3_boundary = (int(review_len/3)*2)
        file_sent['pos'] = file_sent['pos'][fold1_2_boundary:fold2_3_boundary]
        file_sent['neg'] = file_sent['neg'][fold1_2_boundary:fold2_3_boundary]

    elif test_fold == 'fold3':
        fold2_3_boundary = (int(review_len/3)*2)
        file_sent['pos'] = file_sent['pos'][fold2_3_boundary:]
        file_sent['neg'] = file_sent['neg'][fold2_3_boundary:]

    file_sent['all_files'] = file_sent['pos'] + file_sent['neg']

    return file_sent['all_files']

test_files_sent = test_fold()

pred_actual = []
for f in test_files_sent:
    pred_actual.append(test_nb(f))

print(evaluate_model(pred_actual))
