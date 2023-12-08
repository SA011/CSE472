import argparse
import numpy as np
import csv
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif


#add arguments
parser = argparse.ArgumentParser()
parser.add_argument("--input_file", default='dataset1/WA_Fn-UseC_-Telco-Customer-Churn.csv',  type=str, required=False, help="input file")
parser.add_argument("--missing_value", default='mean',  type=str, required=False, help="how to handle missing value (drop / mean / mediam / mode)")
parser.add_argument("--feature_count", default=19,  type=int, required=False, help="number of features to be selected")
parser.add_argument("--number_of_learners", default=5,  type=int, required=False, help="number of learners")
parser.add_argument("--epochs", default=5,  type=int, required=False, help="number of epochs")
parser.add_argument("--mini_batch_size", default=100,  type=int, required=False, help="mini batch size")
parser.add_argument("--k_fold", default=1,  type=int, required=False, help="k fold")
parser.add_argument("--learning_rate", default=100,  type=float, required=False, help="learning rate")
parser.add_argument("--dataset_size", default=20000,  type=int, required=False, help="dataset set size")
parser.add_argument("--all_positive", default=False,  type=bool, required=False, help="all positive")
parser.add_argument("--all_negative", default=False,  type=bool, required=False, help="all negative")
parser.add_argument("--dataset_no", default=1,  type=int, required=False, help="dataset number")
parser.add_argument("--seed", default=1,  type=int, required=False, help="seed")

args = parser.parse_args()

args.dataset_size = int(round(args.dataset_size / 0.8))
np.random.seed(args.seed)

##############################################################################################################
# UTIL
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


##############################################################################################################

##############################################################################################################
# Preprocessing

def preprocess1(args):        

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
        if data[i][-1] == 'No' and args.all_negative:
            modified_data.append(data[i])
        elif data[i][-1] == 'Yes' and args.all_positive:
            modified_data.append(data[i])
        elif not isInvalid(data[i][-1]):
            rem_data.append(data[i])

    limit = min(args.dataset_size - len(modified_data), len(rem_data))
    limit = max(limit, 0)
    np.random.shuffle(rem_data)
    # print(np.array(modified_data).shape)
    # print(np.array(rem_data[:limit]).shape)
    data = np.concatenate([np.array(modified_data).reshape(-1, len(data[0])), np.array(rem_data[:limit]).reshape(-1, len(data[0]))], axis=0)
    #############

    #split data into train set and test set
    train_set, test_set = model_selection.train_test_split(data, test_size=0.2, random_state=42)
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





##############################################################################################################

def preprocess2(args):        

    print("Preprocessing...")
    # print("\n--------------------\n")

    train_data_filename, test_data_filename = args.input_file.split('$')

    with open(train_data_filename, 'r') as f:
        reader = csv.reader(f)
        train_data = list(reader)
    
    with open(test_data_filename, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        test_data = list(reader)
    

    train_data_count = int(args.dataset_size * 0.8)
    test_data_count = int(args.dataset_size * 0.2)
    # print(len(train_data))
    # print(len(test_data))
    # print(len(train_data[0]))
    #select data#######
    modified_train_data = []
    rem_train_data = []
    for i in range(len(train_data)):
        # print(i)
        # print(train_data[i][-1])
        if len(train_data[i]) != len(train_data[1]):
            continue

        if train_data[i][-1][-1] == '.':
            train_data[i][-1] = train_data[i][-1][:-1]
        if train_data[i][-1] == '<=50K' and args.all_negative:
            modified_train_data.append(train_data[i])
        elif train_data[i][-1] == '>50K' and args.all_positive:
            modified_train_data.append(train_data[i])
        elif not isInvalid(train_data[i][-1]):
            rem_train_data.append(train_data[i])

    limit = min(train_data_count - len(modified_train_data), len(rem_train_data))
    limit = max(limit, 0)
    np.random.shuffle(rem_train_data)
    # print(np.array(modified_train_data).shape)
    # print(np.array(rem_train_data[:limit]).shape)
    train_set = np.concatenate([np.array(modified_train_data).reshape(-1, len(train_data[0])), np.array(rem_train_data[:limit]).reshape(-1, len(train_data[0]))], axis=0)
    #############

    #select data#######
    modified_test_data = []
    rem_test_data = []
    for i in range(len(test_data)):
        if len(test_data[i]) != len(test_data[1]):
            continue
        if test_data[i][-1][-1] == '.':
            test_data[i][-1] = test_data[i][-1][:-1]
        if test_data[i][-1] == '<=50K' and args.all_negative:
            modified_test_data.append(test_data[i])
        elif test_data[i][-1] == '>50K' and args.all_positive:
            modified_test_data.append(test_data[i])
        elif not isInvalid(test_data[i][-1]):
            rem_test_data.append(test_data[i])

    limit = min(test_data_count - len(modified_test_data), len(rem_test_data))
    limit = max(limit, 0)
    np.random.shuffle(rem_test_data)
    # print(np.array(modified_test_data).shape)
    # print(np.array(rem_test_data[:limit]).shape)
    test_set = np.concatenate([np.array(modified_test_data).reshape(-1, len(test_data[0])), np.array(rem_test_data[:limit]).reshape(-1, len(test_data[0]))], axis=0)
    #############

    # print("Initial split sizes")
    # print("train set size:", len(train_set))
    # print("test set size:", len(test_set))
    # print("\n--------------------\n")
    header = train_set[0]
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



##############################################################################################################
def preprocess3(args):        

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

##############################################################################################################

##############################################################################################################
#MODEL
class Model:
    def logistic_function(self, x):
        return 1/(1+np.exp(-x))
    
    
    def train_all(self, X, Y, W):
        # print(len(X), "examples")
        # np.random.seed(np.random.randint(0, 1000))
        self.weight = np.random.rand(len(X[0]) + 1)
        for i in range(len(X[0])):
            self.weight[i] = (self.weight[i] - 0.5)

        minibatch = min(self.MiniBatchSize, len(X))
        ind = [i for i in range(len(X))]

        alpha = self.learning_rate

        for i in range(self.epochs):
            np.random.shuffle(ind)
            # temp = np.concatenate([X[ind[:minibatch]], np.ones((minibatch, 1))], axis=1)
            # temp2 = np.matmul(temp, self.weight.T)
            # for j in range(len(temp)):
            #     self.weight = self.weight + alpha * W[ind[j]] * (Y[ind[j]] - self.logistic_function(temp2[j])) * temp[j]
            dup_weight = np.copy(self.weight)
            for l in range(minibatch):
                j = ind[l]
                temp = np.concatenate([X[j], [1]])
                self.weight = self.weight + alpha * W[j] * (Y[j] - self.logistic_function(np.dot(dup_weight, temp))) * temp
            alpha -= self.alpha_decay
        # print("Training complete")
        
    def k_fold_train(self, X, Y, W, k):
        k = min(k, len(X))
        X_split = np.array_split(np.array(X), k)
        Y_split = np.array_split(np.array(Y), k)
        W_split = np.array_split(np.array(W), k)

        accuracy, f1 = 0, 0
        for i in range(k):
            # print("Training fold", i+1)
            X_train = np.concatenate(np.delete(X_split, i), axis=0)
            Y_train = np.concatenate(np.delete(Y_split, i), axis=0)
            W_train = np.concatenate(np.delete(W_split, i), axis=0)
            # print(len(X_train), "examples")
            self.train_all(X_train, Y_train, W_train)
            acc, _, _, _, f, _ = TestModel(X_split[i], Y_split[i], [self], [1])
            accuracy += acc
            f1 += f
        accuracy /= k
        f1 /= k

        return accuracy, f1





    def __init__(self, X, Y, W, args):
        self.epochs = args.epochs
        self.learning_rate = args.learning_rate
        self.alpha_decay = self.learning_rate / (self.epochs + 1)
        self.MiniBatchSize = args.mini_batch_size
        k_fold = args.k_fold
        
        if k_fold <= 1:
            self.train_all(X, Y, W)
        else:
            optimal_alpha = self.learning_rate
            optimal_accuracy = 0
            optimal_f1 = 0
            d = self.learning_rate / 11
            for i in range(10):
                accuracy, f1 = self.k_fold_train(X, Y, W, k_fold)
                if accuracy > optimal_accuracy:
                    optimal_accuracy = accuracy
                    optimal_f1 = f1
                    optimal_alpha = self.learning_rate
                
                self.learning_rate -= d

            self.learning_rate = optimal_alpha   
            # print("Optimal learning rate:", self.learning_rate)
            self.train_all(X, Y, W)


    def predict(self, x):
        temp = np.concatenate([x, [1]])
        return self.logistic_function(np.dot(self.weight, temp)) > 0.5
    
    # def predict_all(self, X):
    #     ret = np.dot(X, self.weight.T)
    #     for i in range(len(ret)):
    #         ret[i] = self.logistic_function(ret[i]) > 0.5
    #     return ret





##############################################################################################################

##############################################################################################################
#ADABOOST
def adaboost(X, Y, Learner, args):
    """
    X: training set features
    Y: training set labels
    Learner: learner class
    args: arguments
    """
    k = args.number_of_learners
    n = len(X)
    h = [] #list of learners
    z = [] #list of weights
    w = [1/n for i in range(n)] #weights of training examples
    # p = 0
    # for i in range(n):
    #     if Y[i] == 1:
    #         p += 1
    # w = [1/p if Y[i] == 1 else 1/(n-p) for i in range(n)]
    # w = [w[i]/sum(w) for i in range(n)]
    eps = 1e-6
    
    print("Training...")
    print(len(X), "examples")
    print(len(Y), "labels")
    print(k, "learners")
    print("\n--------------------\n")
    
    max_iter = 100
    iter = 0
    i = 0

    def select_top_samples(X, Y, w, limit, allPositive=False, allNegative=False):
        if limit <= 0:
            return [], [], []
        if allPositive:
            if allNegative:
                return X, Y, w
            else:
                retX, retY, retW, remX, remY, remW = [], [], [], [], [], []
                for i in range(len(X)):
                    if Y[i] == 1:
                        retX.append(X[i])
                        retY.append(Y[i])
                        retW.append(w[i])
                    else:
                        remX.append(X[i])
                        remY.append(Y[i])
                        remW.append(w[i])
                remX, remY, remW = select_top_samples(remX, remY, remW, limit - len(retX), False, False)

                return np.concatenate([retX, remX]), np.concatenate([retY, remY]), np.concatenate([retW, remW])
        
        if allNegative:
            retX, retY, retW, remX, remY, remW = [], [], [], [], [], []
            for i in range(len(X)):
                if Y[i] == 0:
                    retX.append(X[i])
                    retY.append(Y[i])
                    retW.append(w[i])
                else:
                    remX.append(X[i])
                    remY.append(Y[i])
                    remW.append(w[i])
            remX, remY, remW = select_top_samples(remX, remY, remW, limit - len(retX), False, False)

            return np.concatenate([retX, remX]), np.concatenate([retY, remY]), np.concatenate([retW, remW])
        limit = min(limit, len(X))
        temp = np.concatenate([X, np.array(Y).reshape(-1, 1), np.array(w).reshape(-1, 1)], axis=1)
        ord = np.argsort(temp[:, -1])
        ord = reversed(ord)
        temp = temp[list(ord)]
        return temp[:limit, :-2], temp[:limit, -2], temp[:limit, -1]
        # w = [w[i]/sum(w) for i in range(len(w))]
        # ord = np.random.choice(len(X), limit, p=w)
        # return X[ord], Y[ord], w[ord]

    while i < k:
        # print("Learner", i + 1)
        #resample
        Xnew, Ynew, w_new = select_top_samples(X, Y, w, args.dataset_size, args.all_positive, args.all_negative)
        h.append(Learner(Xnew, Ynew, w_new, args))
        # h.append(Learner(X, Y, w, args))
        error = 0
        res = []
        for j in range(n):
            res.append(h[i].predict(X[j]))
            if res[j] != Y[j]:
                error += w[j]
        # print("Learner", i + 1, "error:", error)
        if error > 0.5:
            iter += 1
            
            # print(error)
            h.pop()
            if iter == max_iter:
                print("Training failed")
                exit(1)
            continue
        iter = 0
        error = max(error, eps)
        for j in range(n):
            if res[j] == Y[j]:
                w[j] *= error/(1-error)
        
        w = [w[j]/sum(w) for j in range(n)]
        z.append(np.log((1-error)/error))
        i += 1
        print(f"Training complete {i}/{k}", end="\r")
        # print("Learner", i, "error:", error)
    print("Training complete          ")
    print("\n--------------------\n")
    z = [z[i]/sum(z) for i in range(len(z))]
    return h, z


##############################################################################################################

##############################################################################################################
#TESTING
def TestModel(X, Y, h, z, print_result=False):
    """
    X: test set features
    Y: test set labels
    h: list of learners
    z: list of weights
    """
    n = len(X)
    if print_result:
        print("Testing...")
        print(len(X), "examples")
        print(len(Y), "labels")
        print(len(h), "learners")
        print("\n--------------------\n")

    TP, FP, TN, FN = 0, 0, 0, 0

    for i in range(n):
        prediction = 0
        for j in range(len(h)):
            prediction += z[j]*(h[j].predict(X[i]) > 0.5)
        prediction = 1 if prediction > 0.5 else 0
        if prediction == Y[i]:
            if prediction == 1:
                TP += 1
            else:
                TN += 1
        else:
            if prediction == 1:
                FP += 1
            else:
                FN += 1


    accuracy = (TP+TN)/(TP+TN+FP+FN) 
    precision = TP/(TP+FP) if TP+FP != 0 else 0
    false_discovery_rate = FP/(TP+FP) if TP+FP != 0 else 0
    recall = TP/(TP+FN) if TP+FN != 0 else 0
    specificity = TN/(TN+FP) if TN+FP != 0 else 0

    f1 = 2*precision*recall/(precision+recall) if precision+recall != 0 else 0
    mcc = (TP*TN-FP*FN)/((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**0.5 if (TP+FP)*(TP+FN)*(TN+FP)*(TN+FN) != 0 else 0

    if print_result:
        print(f"Accuracy: {accuracy:.6f}")
        print(f"Precision: {precision:.6f}")
        print(f"Recall: {recall:.6f}")
        print(f"Specificity: {specificity:.6f}")
        print(f"False discovery rate: {false_discovery_rate:.6f}")
        print(f"F1: {f1:.6f}")
        print(f"MCC: {mcc:.6f}")
        print("\n--------------------\n")

    return accuracy, precision, recall, specificity, f1, mcc

##############################################################################################################

##############################################################################################################
#Main
if args.dataset_no == 1:
    X_train, Y_train, X_test, Y_test = preprocess1(args)
if args.dataset_no == 2:
    X_train, Y_train, X_test, Y_test = preprocess2(args)
if args.dataset_no == 3:
    X_train, Y_train, X_test, Y_test = preprocess3(args)

h, z = adaboost(X_train, Y_train, Model, args)


print("Testing on train set")
TestModel(X_train, Y_train, h, z, True)
print("Testing on test set")
TestModel(X_test, Y_test, h, z, True)