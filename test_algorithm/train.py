"""
Pre-train classifiers
Author: Yuliang Zou
Date: 04/19/2016
"""
import os
import re
import Porter_stemming as ps
import math
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib


# Helper function
def tokenize_text(input_str):
    temp_list = re.findall(r"\b[a-z'-]+\b", input_str)
    # temp_list = re.split("\s+|\.|,|\"", removeSGML_str)
    tokenizeText_list = list(temp_list)
    for index, item in enumerate(temp_list):
        if "we're" in item:
            tokenizeText_list[index] = ""
            tokenizeText_list.append("we")
            tokenizeText_list.append("are")
        elif "i'm" in item:
            tokenizeText_list[index] = ""
            tokenizeText_list.append("I")
            tokenizeText_list.append("am")
        elif "isn't" in item:
            tokenizeText_list[index] = ""
            tokenizeText_list.append("is")
            tokenizeText_list.append("not")
        elif "doesn't" in item:
            tokenizeText_list[index] = ""
            tokenizeText_list.append("does")
            tokenizeText_list.append("not")
        elif "don't" in item:
            tokenizeText_list[index] = ""
            tokenizeText_list.append("do")
            tokenizeText_list.append("not")
        elif "can't" in item:
            tokenizeText_list[index] = ""
            tokenizeText_list.append("can")
            tokenizeText_list.append("not")
        elif "we've" in item:
            tokenizeText_list[index] = ""
            tokenizeText_list.append("we")
            tokenizeText_list.append("have")
        elif "it's" in item:
            tokenizeText_list[index] = ""
            tokenizeText_list.append("it")
            tokenizeText_list.append("is")
    for i in range(tokenizeText_list.count("")):
        tokenizeText_list.remove("")
    return tokenizeText_list


def remove_stopwords(tokenizeText_list, stopwords_dict):
    removeStopwords_list = list(tokenizeText_list)
    for word in tokenizeText_list:
        if word in stopwords_dict:
            removeStopwords_list.remove(word)
    return removeStopwords_list


def stem_words(removeStopwords_list):
    stemWords_list = list(removeStopwords_list)
    p = ps.PorterStemmer()
    for index, word in enumerate(removeStopwords_list):
        temp1 = p.stem(word, 0, len(word) - 1)
        temp2 = p.stem(temp1, 0, len(temp1) - 1)
        while temp1 != temp2:
            temp1 = temp2
            temp2 = p.stem(temp1, 0, len(temp1) - 1)
        stemWords_list[index] = temp2
    return stemWords_list


def train_naive_bayes(X_train, t_train):
    n_train = len(t_train)
    free_words = []
    not_words = []
    num_free = 0    # number of free documents(events)
    for i in range(n_train):
        if t_train[i] == 0:
            not_words += X_train[i]
        else:
            num_free += 1
            free_words += X_train[i]
    voc_num = len(set(free_words + not_words))
    free_num = len(free_words)    # number of free words
    not_num = len(not_words)
    num = [free_num, not_num, voc_num]
    p_free = 1.0 * num_free / n_train
    p_not = 1 - p_free
    prior = [p_not, p_free]

    not_dict = {}
    free_dict = {}
    for word in not_words:
        if word not in not_dict:
            not_dict[word] = 1
        else:
            not_dict[word] += 1
    for word in not_dict:
        not_dict[word] = 1.0 * (not_dict[word] + 1.0) / (not_num + voc_num)
    for word in free_words:
        if word not in free_dict:
            free_dict[word] = 1
        else:
            free_dict[word] += 1
    for word in free_dict:
        free_dict[word] = 1.0 * (free_dict[word] + 1.0) / (free_num + voc_num)
    likelihood = [not_dict, free_dict]

    return prior, likelihood, num


def train():
    # Load stopwords list
    stopwords_filename = 'stopwords.txt'
    INFILE = open(stopwords_filename)
    stopwords = INFILE.read()
    INFILE.close()
    stopwords = re.split('\s+', stopwords)
    stopwords_dict = {}
    for index, word in enumerate(stopwords):
        stopwords_dict[word] = index

    # Load training data
    file_list = os.listdir('trainingdata/')
    if '.DS_Store' in file_list:
        file_list.remove('.DS_Store')
    voc = []    # vocabulary
    X = []    # documents
    t = []    # true label
    for i in range(len(file_list)):
        file_name = 'trainingdata/' + file_list[i]
        INFILE = open(file_name)
        input_str = INFILE.read().lower()
        INFILE.close()
        text = tokenize_text(input_str)
        # You can change the pre-processing methods here if you want
        text = remove_stopwords(text, stopwords_dict)
        text = stem_words(text)

        voc += text
        X.append(text)
        if 'not' in file_name:
            t.append(0)
        else:
            t.append(1)

    # Calculate TF-IDF to construct vectors
    voc = set(voc)
    word_dict = {}
    doc_num = len(t)
    idx = 0
    for word in voc:
        word_dict[word] = [idx, 0]    # idx in vector, df
        idx += 1
    # Document Frequency
    for x in X:
        x = set(x)
        for word in x:
            word_dict[word][1] += 1
    # Term Frequency
    X_vectors = []
    for x in X:
        vector = [0] * len(voc)
        for word in x:
            vector[word_dict[word][0]] += 1
        for word in x:
            vector[word_dict[word][0]] *= math.log10(1.0 * doc_num / word_dict[word][1])
        X_vectors.append(vector)

    prior, likelihood, num = train_naive_bayes(X, t)
    nb_clf = [prior, likelihood, num]
    joblib.dump(nb_clf, 'nb.pkl')
    clf = joblib.load('nb.pkl')

    svm_clf = SVC()
    svm_clf.fit(X_vectors, t)
    joblib.dump(svm_clf, 'svm.pkl')

    knn_clf = KNeighborsClassifier()
    knn_clf.fit(X_vectors, t)
    joblib.dump(knn_clf, 'knn.pkl')

    dt_clf = DecisionTreeClassifier()
    dt_clf.fit(X_vectors, t)
    joblib.dump(dt_clf, 'dt.pkl')

    joblib.dump(word_dict, 'word_dict.pkl')


train()