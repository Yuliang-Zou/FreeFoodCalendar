"""
Compare difference classification algorithms
Author: Yuliang Zou
Date: 04/15/2016
"""
import os
import re
import Porter_stemming as ps
import numpy as np
import math
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


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


def test_naive_bayes(X_test, t_test, prior, likelihood, num):
    n_test = len(t_test)
    p_not = math.log10(prior[0])
    p_free = math.log10(prior[1])
    not_dict = likelihood[0]
    free_dict = likelihood[1]
    not_num = num[0]
    free_num = num[1]
    voc_num = num[2]
    y_test = []
    for i in range(n_test):
        words = X_test[i]
        for word in words:
            # not free
            if word in not_dict:
                p_not += math.log10(1.0 * not_dict[word])
            else:
                p_not += math.log10(1.0 / (not_num + voc_num))
            # free
            if word in free_dict:
                p_free += math.log10(1.0 * free_dict[word])
            else:
                p_free += math.log10(1.0 / (free_num + voc_num))
        if p_free >= p_not:
            y_test.append(1)
        else:
            y_test.append(0)

    # leave-one-out
    if n_test == 1:
        return y_test == t_test

    # 8/2 split
    t_pos = 0
    f_pos = 0
    t_neg = 0
    f_neg = 0
    for i in range(n_test):
        if t_test[i] == 1 and y_test[i] == 1:
            t_pos += 1
        elif t_test[i] == 0 and y_test[i] == 1:
            f_pos += 1
        elif t_test[i] == 1 and y_test[i] == 0:
            f_neg += 1
        elif t_test[i] == 0 and y_test[i] == 0:
            t_neg += 1

    if t_pos == 0:
        precision = 0
        recall = 0
    else:
        precision = 1.0 * t_pos / (t_pos + f_pos)
        recall = 1.0 * t_pos / (t_pos + f_neg)
    return precision, recall


def naive_bayes(X, t):
    # leave-one-out strategy to get average accuracy
    n = len(t)
    true_num = 0
    for i in range(n):
        X_train = list(X)
        del X_train[i]
        y_train = list(t)
        del y_train[i]
        X_test = X[i]
        y_test = [t[i]]

        prior, likelihood, num = train_naive_bayes(X_train, y_train)

        if test_naive_bayes(X_test, y_test, prior, likelihood, num):
            true_num += 1
    accuracy = 1.0 * true_num / n

    # 8/2 split
    pre = []
    rec = []
    for _ in range(100):
        X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=0.2)
        prior, likelihood, num = train_naive_bayes(X_train, t_train)
        precision, recall = test_naive_bayes(X_test, t_test, prior, likelihood, num)
        pre.append(precision)
        rec.append(recall)
    pre = sum(pre) / len(pre)
    rec = sum(rec) / len(rec)
    F = 2 / (1/pre + 1/rec)

    return accuracy, pre, rec, F


def svm(X_vectors, t):
    # leave-one-out strategy to get average accuracy
    n = len(t)
    true_num = 0
    for i in range(n):
        X_train = list(X_vectors)
        del X_train[i]
        t_train = list(t)
        del t_train[i]
        X_test = X_vectors[i]
        t_test = t[i]

        clf = SVC()
        clf.fit(X_train, t_train)
        y = clf.predict(X_test)
        if y == t_test:
            true_num += 1
    accuracy = 1.0 * true_num / n

    # 8/2 split
    X = np.array(X_vectors)
    tt = list(t)
    pre = []
    rec = []
    for _ in range(100):
        X_train, X_test, t_train, t_test = train_test_split(X, tt, test_size=0.2)
        clf = SVC()
        clf.fit(X_train, t_train)
        y_test = clf.predict(X_test)
        t_pos = 0
        f_pos = 0
        t_neg = 0
        f_neg = 0
        for i in range(len(y_test)):
            if t_test[i] == 1 and y_test[i] == 1:
                t_pos += 1
            elif t_test[i] == 0 and y_test[i] == 1:
                f_pos += 1
            elif t_test[i] == 0 and y_test[i] == 0:
                t_neg += 1
            elif t_test[i] == 1 and y_test[i] == 0:
                f_neg += 1

            if t_pos == 0:
                precision = 0
                recall = 0
            else:
                precision = 1.0 * t_pos / (t_pos + f_pos)
                recall = 1.0 * t_pos / (t_pos + f_neg)
            pre.append(precision)
            rec.append(recall)

    pre = sum(pre) / len(pre)
    rec = sum(rec) / len(rec)
    F = 2 / (1/pre + 1/rec)

    return accuracy, pre, rec, F


def knn(X_vectors, t):
    # leave-one-out strategy to get average accuracy
    n = len(t)
    true_num = 0
    for i in range(n):
        X_train = list(X_vectors)
        del X_train[i]
        t_train = list(t)
        del t_train[i]
        X_test = X_vectors[i]
        t_test = t[i]

        clf = KNeighborsClassifier(n_neighbors=5)
        clf.fit(X_train, t_train)
        y = clf.predict(X_test)
        if y == t_test:
            true_num += 1
    accuracy = 1.0 * true_num / n

    # 8/2 split
    X = np.array(X_vectors)
    tt = list(t)
    pre = []
    rec = []
    for _ in range(100):
        X_train, X_test, t_train, t_test = train_test_split(X, tt, test_size=0.2)
        clf = KNeighborsClassifier(n_neighbors=5)
        clf.fit(X_train, t_train)
        y_test = clf.predict(X_test)
        t_pos = 0
        f_pos = 0
        t_neg = 0
        f_neg = 0
        for i in range(len(y_test)):
            if t_test[i] == 1 and y_test[i] == 1:
                t_pos += 1
            elif t_test[i] == 0 and y_test[i] == 1:
                f_pos += 1
            elif t_test[i] == 0 and y_test[i] == 0:
                t_neg += 1
            elif t_test[i] == 1 and y_test[i] == 0:
                f_neg += 1

            if t_pos == 0:
                precision = 0
                recall = 0
            else:
                precision = 1.0 * t_pos / (t_pos + f_pos)
                recall = 1.0 * t_pos / (t_pos + f_neg)
            pre.append(precision)
            rec.append(recall)

    pre = sum(pre) / len(pre)
    rec = sum(rec) / len(rec)
    F = 2 / (1/pre + 1/rec)

    return accuracy, pre, rec, F


def decision_tree(X_vectors, t):
    # leave-one-out strategy to get average accuracy
    n = len(t)
    true_num = 0
    for i in range(n):
        X_train = list(X_vectors)
        del X_train[i]
        t_train = list(t)
        del t_train[i]
        X_test = X_vectors[i]
        t_test = t[i]

        clf = DecisionTreeClassifier()
        clf.fit(X_train, t_train)
        y = clf.predict(X_test)
        if y == t_test:
            true_num += 1
    accuracy = 1.0 * true_num / n

    # 8/2 split
    X = np.array(X_vectors)
    tt = list(t)
    pre = []
    rec = []
    for _ in range(100):
        X_train, X_test, t_train, t_test = train_test_split(X, tt, test_size=0.2)
        clf = DecisionTreeClassifier()
        clf.fit(X_train, t_train)
        y_test = clf.predict(X_test)
        t_pos = 0
        f_pos = 0
        t_neg = 0
        f_neg = 0
        for i in range(len(y_test)):
            if t_test[i] == 1 and y_test[i] == 1:
                t_pos += 1
            elif t_test[i] == 0 and y_test[i] == 1:
                f_pos += 1
            elif t_test[i] == 0 and y_test[i] == 0:
                t_neg += 1
            elif t_test[i] == 1 and y_test[i] == 0:
                f_neg += 1

            if t_pos == 0:
                precision = 0
                recall = 0
            else:
                precision = 1.0 * t_pos / (t_pos + f_pos)
                recall = 1.0 * t_pos / (t_pos + f_neg)
            pre.append(precision)
            rec.append(recall)

    pre = sum(pre) / len(pre)
    rec = sum(rec) / len(rec)
    F = 2 / (1/pre + 1/rec)

    return accuracy, pre, rec, F


def white_list(X, t):
    n = len(t)
    t_pos = 0
    f_pos = 0
    t_neg = 0
    f_neg = 0
    y = []
    for i in range(n):
        if 'food' in X[i] or 'snack' in X[i] or 'pizza' in X[i]:
            y.append(1)
            if t[i] == 1:
                t_pos += 1
            else:
                f_pos += 1
        else:
            y.append(0)
            if t[i] == 0:
                t_neg += 1
            else:
                f_neg += 1

    accuracy = 1.0 * (t_pos + t_neg) / n
    pre = 1.0 * t_pos / (t_pos + f_pos)
    rec = 1.0 * t_pos / (t_pos + f_neg)
    F = 2 / (1/pre + 1/rec)

    return accuracy, pre, rec, F


def test():
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

    # Naive Bayes
    nb_acc, nb_pre, nb_rec, nb_F = naive_bayes(X, t)

    # SVM
    svm_acc, svm_pre, svm_rec, svm_F = svm(X_vectors, t)

    # kNN
    knn_acc, knn_pre, knn_rec, knn_F = knn(X_vectors, t)

    # Decision Tree
    dt_acc, dt_pre, dt_rec, dt_F = decision_tree(X_vectors, t)

    # White List
    wl_acc, wl_pre, wl_rec, wl_F = white_list(X, t)

    print 'naive bayes'
    print nb_acc, nb_pre, nb_rec, nb_F

    print 'svm'
    print svm_acc, svm_pre, svm_rec, svm_F

    print 'knn'
    print knn_acc, knn_pre, knn_rec, knn_F

    print 'decision tree'
    print dt_acc, dt_pre, dt_rec, dt_F

    print 'while list'
    print wl_acc, wl_pre, wl_rec, wl_F


test()
