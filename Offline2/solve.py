import argparse
import PreProcessor1, PreProcessor2, PreProcessor3
from Boost import adaboost
from Model import Model
from TestModel import TestModel
import numpy as np

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
#preprocess
if args.dataset_no == 1:
    X_train, Y_train, X_test, Y_test = PreProcessor1.preprocess(args)
if args.dataset_no == 2:
    X_train, Y_train, X_test, Y_test = PreProcessor2.preprocess(args)
if args.dataset_no == 3:
    X_train, Y_train, X_test, Y_test = PreProcessor3.preprocess(args)

h, z = adaboost(X_train, Y_train, Model, args)


print("Testing on train set")
TestModel(X_train, Y_train, h, z, True)
print("Testing on test set")
TestModel(X_test, Y_test, h, z, True)




