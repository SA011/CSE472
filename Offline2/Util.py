import csv
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import numpy as np

def isInvalid(value):
    return value == '' or value == ' ' or value == None or value == ' ?' or value == '?'

# def select_features(train_set, test_set, feature_count):
#     X_train = np.array(train_set)[:, :-1]
#     Y_train = np.array(train_set)[:, -1]

#     X_test = np.array(test_set)[:, :-1]
#     Y_test = np.array(test_set)[:, -1]

#     selector = SelectKBest(f_classif, k=feature_count)
#     selector.fit(X_train, Y_train)
#     X_train = selector.transform(X_train)
#     X_test = selector.transform(X_test)
    
#     train_set = np.concatenate([X_train, Y_train.reshape(-1, 1)], axis=1)
#     test_set = np.concatenate([X_test, Y_test.reshape(-1, 1)], axis=1)

#     return train_set, test_set

# def select_features(X_train, Y_train, X_test, Y_test, feature_count):
#     selector = SelectKBest(f_classif, k=feature_count)
#     selector.fit(X_train, Y_train)
#     X_train = selector.transform(X_train)
#     X_test = selector.transform(X_test)
    
#     return X_train, Y_train, X_test, Y_test

def B(x):
    if x == 0 or x == 1:
        return 0
    return -x * np.log2(x) - (1 - x) * np.log2(1 - x)

def select_features(X_train, Y_train, X_test, Y_test, feature_count):
    info_gain = []
    TP = 0
    for i in range(len(X_train)):
        if Y_train[i] == 1:
            TP += 1
    
    BT = B(TP / len(X_train))

    for i in range(len(X_train[0])):
        positive = [0, 0]
        negative = [0, 0]
        for j in range(len(X_train)):
            if X_train[j][i] == None:
                continue
            if X_train[j][i] >= 0.5:
                if Y_train[j] == 1:
                    positive[0] += 1
                positive[1] += 1
            else:
                if Y_train[j] == 1:
                    negative[0] += 1
                negative[1] += 1
        
        if positive[1] == 0 or negative[1] == 0:
            info_gain.append(BT)
            continue

        B1 = B(positive[0] / positive[1])
        B2 = B(negative[0] / negative[1])
        info_gain.append(BT - positive[1] / len(X_train) * B1 - negative[1] / len(X_train) * B2)

    ord = np.argsort(info_gain)
    ord = reversed(ord)
    ord = list(ord)
    ord = ord[:feature_count]
    X_train = X_train[:, ord]
    X_test = X_test[:, ord]

    return X_train, Y_train, X_test, Y_test
