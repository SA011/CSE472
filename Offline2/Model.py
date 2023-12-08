import numpy as np
from TestModel import TestModel

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