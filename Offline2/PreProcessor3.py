import csv
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.impute import SimpleImputer
import numpy as np
from Util import *


def preprocess(args):        

    print("Preprocessing...")
    # print("\n--------------------\n")

    with open(args.input_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        data = list(reader)

    #select data#######
    modified_data = []
    rem_data = []
    for i in range(len(data)):
        if data[i][-1] == '0' and args.all_negative:
            modified_data.append(data[i])
        elif data[i][-1] == '1' and args.all_positive:
            modified_data.append(data[i])
        elif not isInvalid(data[i][-1]):
            rem_data.append(data[i])
    # print(len(modified_data))
    limit = min(args.dataset_size - len(modified_data), len(rem_data))
    limit = max(limit, 0)
    np.random.shuffle(rem_data)
    # print(np.array(modified_data).shape)
    # print(np.array(rem_data[:limit]).shape)
    data = np.concatenate([np.array(modified_data).reshape(-1, len(data[0])), np.array(rem_data[:limit]).reshape(-1, len(data[0]))], axis=0)
    #############

    positives = []
    negatives = []
    for i in range(len(data)):
        if data[i][-1] == '0':
            negatives.append(data[i])
        else:
            positives.append(data[i])

    #split data into train set and test set
    # train_set, test_set = model_selection.train_test_split(data, test_size=0.2, random_state=42)
    train_set_positive, test_set_positve = model_selection.train_test_split(positives, test_size=0.2, random_state=42)
    train_set_negative, test_set_negative = model_selection.train_test_split(negatives, test_size=0.2, random_state=42)

    train_set = np.concatenate((np.array(train_set_positive), np.array(train_set_negative)), axis=0)
    test_set = np.concatenate((np.array(test_set_positve), np.array(test_set_negative)), axis=0)
    
    
    # print("Initial split sizes")
    # print("train set size:", len(train_set))
    # print("test set size:", len(test_set))
    # print("\n--------------------\n")

    # split train set features with numerical and categorical features
    train_set_num = [[] for i in range(len(train_set))]
    train_set_cat = [[] for i in range(len(train_set))]
    feature_type = [None for i in range(len(header))]

    for i in range(1, len(header)):
        isNumeric = True
        for j in range(len(train_set)):
            if isInvalid(train_set[j][i]):
                continue
            try:
                float(train_set[j][i])
            except ValueError:
                isNumeric = False
                break
        
        if isNumeric:
            feature_type[i] = 'numeric'

        if not isNumeric:
            isYesNo = True
            for j in range(len(train_set)):
                if isInvalid(train_set[j][i]):
                    continue
                if not train_set[j][i].startswith('Yes') and not train_set[j][i].startswith('No'):
                    isYesNo = False
                    break
            if isYesNo:
                isNumeric = False
                feature_type[i] = 'yesno'
                for j in range(len(train_set)):
                    if isInvalid(train_set[j][i]):
                        continue
                    elif train_set[j][i].startswith('Yes'):
                        train_set[j][i] = 'Yes'
                    else:
                        train_set[j][i] = 'No'
            else:
                feature_type[i] = 'categorical'

        if isNumeric:
            for j in range(len(train_set)):
                if isInvalid(train_set[j][i]):
                    train_set_num[j].append(None)
                else:
                    train_set_num[j].append(float(train_set[j][i]))
        else:
            for j in range(len(train_set)):
                if isInvalid(train_set[j][i]):
                    train_set_cat[j].append(None)
                else:
                    train_set_cat[j].append(train_set[j][i])

    #normalize numeric features
    if len(train_set_num[0]) > 0:
        train_set_num = np.array(train_set_num)
        min_max_scaler = preprocessing.MinMaxScaler().fit(train_set_num)
        train_set_num = min_max_scaler.transform(train_set_num)

        missing_value_handler = SimpleImputer(missing_values=np.nan, strategy=args.missing_value)
        missing_fit = missing_value_handler.fit(train_set_num)
        train_set_num = missing_fit.transform(train_set_num)


    #one hot encoding categorical features
    if len(train_set_cat[0]) > 0:
        train_set_cat = np.array(train_set_cat)
        oneHotEncoder = preprocessing.OneHotEncoder(drop='if_binary').fit(train_set_cat)
        train_set_cat = oneHotEncoder.transform(train_set_cat).toarray()

    #merge
    train_set = np.concatenate([train_set_num, train_set_cat], axis=1)

    #class missing value handling (dropped)
    new_train_set = [train_set[i] for i in range(len(train_set)) if train_set[i][-1] != None]


    #work on test set
    test_set_num = [[] for i in range(len(test_set))]
    test_set_cat = [[] for i in range(len(test_set))]

    for i in range(1, len(header)):
        if feature_type[i] == 'numeric':
            for j in range(len(test_set)):
                if isInvalid(test_set[j][i]):
                    test_set_num[j].append(None)
                else:
                    try:
                        test_set_num[j].append(float(test_set[j][i]))
                    except ValueError:
                        test_set_num[j].append(None)
        elif feature_type[i] == 'yesno':
            for j in range(len(test_set)):
                if isInvalid(test_set[j][i]):
                    test_set_cat[j].append(None)
                elif test_set[j][i].startswith('Yes'):
                    test_set_cat[j].append('Yes')
                elif test_set[j][i].startswith('No'):
                    test_set_cat[j].append('No')
                else:
                    test_set_cat[j].append(None)
        else:
            for j in range(len(test_set)):
                if isInvalid(test_set[j][i]):
                    test_set_cat[j].append(None)
                else:
                    test_set_cat[j].append(test_set[j][i])

    #normalize numeric features
    if len(train_set_num[0]) > 0:
        test_set_num = np.array(test_set_num)
        test_set_num = min_max_scaler.transform(test_set_num)
        test_set_num = missing_fit.transform(test_set_num)

    #one hot encoding categorical features
    if len(train_set_cat[0]) > 0:
        test_set_cat = np.array(test_set_cat)
        test_set_cat = oneHotEncoder.transform(test_set_cat).toarray()

    #merge
    test_set = np.concatenate([test_set_num, test_set_cat], axis=1)

    #class missing value handling (dropped)
    new_test_set = [test_set[i] for i in range(len(test_set)) if test_set[i][-1] != None]

    # print("After missing value handling")
    # print("train set size:", len(new_train_set))
    # print("test set size:", len(new_test_set))
    # print("\n--------------------\n")

    X_train = np.array(new_train_set)[:, :-1]
    Y_train = np.array(new_train_set)[:, -1]

    X_test = np.array(new_test_set)[:, :-1]
    Y_test = np.array(new_test_set)[:, -1]

    #feature selection
    X_train, Y_train, X_test, Y_test = select_features(X_train, Y_train, X_test, Y_test, args.feature_count)

    print("Preprocessing done")
    print("\n--------------------\n")

    return X_train, Y_train, X_test, Y_test

