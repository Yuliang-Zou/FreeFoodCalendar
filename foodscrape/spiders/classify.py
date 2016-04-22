"""
Use pre-train classifiers to do classification
Author: Yuliang Zou
Date: 04/19/2016
"""
import re
import Porter_stemming as ps
import math
from sklearn.externals import joblib
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


def classify_naive_bayes(X_test, prior, likelihood, num):
    p_not = math.log10(prior[0])
    p_free = math.log10(prior[1])
    not_dict = likelihood[0]
    free_dict = likelihood[1]
    not_num = num[0]
    free_num = num[1]
    voc_num = num[2]
    for word in X_test:
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
        return True
    else:
        return False


def vectorize(text, doc_num=78):
    # Need this to construct the data vectors
    word_dict = joblib.load('foodscrape/spiders/word_dict.pkl')
    vector = [0] * len(word_dict)
    for word in text:
        if word in word_dict:
            vector[word_dict[word][0]] += 1
    for word in text:
        if word in word_dict:
            vector[word_dict[word][0]] *= math.log10(1.0 * doc_num / word_dict[word][1])
    return vector


def classify(input_str, opt=0):
    # Load stopwords list
    stopwords_filename = 'foodscrape/spiders/stopwords.txt'
    INFILE = open(stopwords_filename)
    stopwords = INFILE.read()
    INFILE.close()
    stopwords = re.split('\s+', stopwords)
    stopwords_dict = {}
    for index, word in enumerate(stopwords):
        stopwords_dict[word] = index

    # Pre-processing
    text = tokenize_text(input_str)
    text = remove_stopwords(text, stopwords_dict)
    text = stem_words(text)

    # Classification
    if opt == -1:
        return True

    # white list
    if opt == 0:
        if 'food' in text or 'snack' in text or 'pizza' in text:
            return True
        else:
            return False

    # naive bayes
    elif opt == 1:
        nb_clf = joblib.load('foodscrape/spiders/nb.pkl')
        prior = nb_clf[0]
        likelihood = nb_clf[1]
        num = nb_clf[2]
        return classify_naive_bayes(text, prior, likelihood, num)

    # svm
    elif opt == 2:
        svm_clf = joblib.load('foodscrape/spiders/svm.pkl')
        vector = vectorize(text)
        return svm_clf.predict(vector) == 1

    # knn
    elif opt == 3:
        knn_clf = joblib.load('foodscrape/spiders/knn.pkl')
        vector = vectorize(text)
        return knn_clf.predict(vector) == 1

    # decision tree
    elif opt == 4:
        dt_clf = joblib.load('foodscrape/spiders/dt.pkl')
        vector = vectorize(text)
        return dt_clf.predict(vector) == 1


#if __name__ == '':
classify('something', 2)