# -*- coding: utf-8 -*-
"""
Importing stuff
"""

import numpy as np
import torchvision.datasets as ds
import torchvision.transforms as transforms
from sklearn import model_selection
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pickle

EPS = 1e-5
np.random.seed(100)

"""Layer"""

class Layer:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
    def feedforward(self, input):
        raise NotImplementedError
    def backpropagation(self, grad_output, learning_rate):
        raise NotImplementedError
    def reset(self):
        pass
    def transform(self):
        pass
    def batchNormalization(self):
        pass

"""Other Layers

"""

class BufferLayer(Layer):
    def __init__(self, input_size, output_size):
        assert input_size == output_size
        super().__init__(input_size, output_size)
    def feedforward(self, input):
        return input
    def backpropagation(self, grad_output, learning_rate):
        return grad_output

class DenseLayer(Layer):
    def __init__(self, input_size, output_size, optimizer = 'Adam'):
        super().__init__(input_size, output_size)
        self.standard_deviation = 1
        self.standard_deviation = np.sqrt(2 / (input_size + output_size))
        self.weights = np.random.randn(input_size + 1, output_size) * self.standard_deviation
        self.input = None
        self.output = None

        self.beta1 = 0.9
        self.beta2 = 0.999

        self.m = np.zeros((input_size + 1, output_size))
        self.v = np.zeros((input_size + 1, output_size))
        self.t = 0
        self.eps = 1e-8
        self.opt = None
        if optimizer == 'Adam':
            self.opt = self.Adam
        elif optimizer == 'Nadam':
            self.opt = self.Nadam
        else:
            self.opt = self.NoOpt


    def feedforward(self, input):
        self.input = np.hstack((np.ones((input.shape[0], 1)), input))
        self.output = self.input @ self.weights
        return self.output

    def backpropagation(self, grad_output, learning_rate):
        grad_input = grad_output @ self.weights[1:,:].T
        grad_weights = self.input.T @ grad_output
        grad_weights = self.opt(grad_weights)
        self.weights -= learning_rate * grad_weights
        return grad_input


    def reset(self):
        self.weights = np.random.randn(self.input_size + 1, self.output_size) * self.standard_deviation

    def transform(self):
        try:
            self.weights.get()
            self.weights = self.weights.get()

        except:
            pass
        self.m = None
        self.v = None
        self.input = None
        self.output = None
        self.standard_deviation = None

    def Adam(self, grad_weights):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad_weights
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad_weights * grad_weights
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        return m_hat / (np.sqrt(v_hat) + self.eps)

    def Nadam(self, grad_weights):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad_weights
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad_weights * grad_weights
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        return (self.beta1 * m_hat + (1 - self.beta1) * grad_weights) / (np.sqrt(v_hat) + self.eps)
    def NoOpt(self, grad_weights):
        return grad_weights

class ReluLayer(Layer):
    def __init__(self, input_size, output_size):
        assert input_size == output_size
        super().__init__(input_size, output_size)
        self.input = None
        self.output = None

    def feedforward(self, input):
        self.input = input
        self.output = np.maximum(0, input)
        return self.output

    def relu_derivative(self, _x):
        return 1. * (_x > 0)

    def backpropagation(self, grad_output, learning_rate):
        return grad_output * self.relu_derivative(self.input)

    def transform(self):
        self.input = None
        self.output = None

class SigmoidLayer(Layer):
    def __init__(self, input_size, output_size):
        assert input_size == output_size
        super().__init__(input_size, output_size)
        self.input = None
        self.output = None

    def sigmoid(self, x):
        ret = np.where(x > 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))
        return ret

    def feedforward(self, input):
        self.input = input
        self.output = self.sigmoid(input)
        return self.output

    def backpropagation(self, grad_output, learning_rate):
        return grad_output * (self.output) * (1 - self.output)

    def transform(self):
        self.input = None
        self.output = None

class SoftmaxLayer(Layer):
    def __init__(self, input_size, output_size):
        assert input_size == output_size
        super().__init__(input_size, output_size)
        self.input = None
        self.output = None

    def feedforward(self, input):
        self.input = input
        exp = np.exp(input - np.max(input, axis = 1).reshape(-1, 1))
        self.output = exp / np.sum(exp, axis = 1).reshape(-1, 1)
        return self.output

    def backpropagation(self, grad_output, learning_rate):
        return grad_output * (self.output - self.output * self.output)

    def transform(self):
        self.input = None
        self.output = None


class DropoutLayer(Layer):
    def __init__(self, input_size, output_size, dropout_rate):
        super().__init__(input_size, output_size)
        self.dropout_rate = dropout_rate
        self.mask = None
    def feedforward(self, input):
        self.mask = np.random.binomial(1, 1 - self.dropout_rate, size = input.shape) / (1 - self.dropout_rate)
        # print(self.mask)
        return input * self.mask
    def backpropagation(self, loss, learning_rate):
        return loss * self.mask
    def reset(self):
        pass
    def transform(self):
        self.dropout_rate = 0
        self.mask = None

class BatchNormalizationLayer(Layer):
    def __init__(self, input_size, output_size):
        super().__init__(input_size, output_size)
        self.input = None
        self.output = None
        self.gamma = 1
        self.beta = 0
        self.eps = EPS

    def feedforward(self, input):
        self.input = input
        self.output = (input - np.mean(input, axis = 0)) / np.sqrt(np.var(input, axis = 0) + self.eps)
        return self.gamma * self.output + self.beta

    def backpropagation(self, grad_output, learning_rate):
        grad_input = grad_output * self.gamma / np.sqrt(np.var(self.input, axis = 0) + self.eps)
        grad_gamma = np.sum(grad_output * self.output, axis = 0)
        grad_beta = np.sum(grad_output, axis = 0)
        self.gamma -= learning_rate * grad_gamma
        self.beta -= learning_rate * grad_beta
        return grad_input

    def reset(self):
        self.gamma = 1
        self.beta = 0

    def transform(self):
        self.input = None
        self.output = None
        try:
            self.gamma.get()
            self.gamma = self.gamma.get()
            self.beta = self.beta.get()

        except:
            pass

"""Losses"""

class SquaredLoss:
    def feedforward(self, input, label):
        return np.sum((input - label) ** 2)
    def backpropagation(self, predicted, label):
        return np.sum(2 * (predicted - label), axis = 0) / predicted.shape[0]
class CrossEntropyLoss:
    def feedforward(self, input, label):
        return -np.sum(label * np.log2(input + EPS))

    def backpropagation(self, predicted, label):
        delta = -label / (predicted + EPS)
        return delta

class SoftmaxCrossEntropyLoss:
    def feedforward(self, input, label):
        return -np.sum(label * np.log2(input + EPS))

    def backpropagation(self, predicted, label):
        return predicted - label

"""Model

"""

class FNN:
    def __init__(self, input_size, init_layer = BufferLayer, Loss = SoftmaxCrossEntropyLoss):
        np.random.seed(100)
        self.input_size = input_size
        self.layers = [init_layer(input_size, input_size)]
        self.LossModel = Loss()
    def add_layer(self, Layer, nodes = None, **kwargs):
        if nodes == None:
            nodes = self.layers[-1].output_size
        if Layer == DropoutLayer:
            try:
                dropout_rate = kwargs["dropout_rate"]
            except:
                dropout_rate = 0.5
            self.layers.append(Layer(self.layers[-1].output_size, nodes, dropout_rate))
        elif Layer == DenseLayer:
            try:
                optimizer = kwargs["optimizer"]
            except:
                optimizer = 'Adam'
            self.layers.append(Layer(self.layers[-1].output_size, nodes, optimizer))
        else:
            self.layers.append(Layer(self.layers[-1].output_size, nodes))

    def feedforward(self, input):
        for layer in self.layers:
            input = layer.feedforward(input)
        return input
    def loss(self, output, label):
        return self.LossModel.feedforward(output, label)

    def backpropagation(self, loss, learning_rate):
        n = len(self.layers)
        if self.LossModel.__class__ == SoftmaxCrossEntropyLoss and self.layers[-1].__class__ == SoftmaxLayer:
            n -= 1
        for i in range(n - 1, 0, -1):
            loss = self.layers[i].backpropagation(loss, learning_rate)

    def trainUtil(self, train_data, train_label, epochs, learning_rate, minibatch, verbose = False):
        self.reset()
        decay = learning_rate / (epochs + 1)
        losses = []
        f1_macros = []
        multiplier = 0.95
        for epoch in range(epochs):
            for i in range(0, train_data.shape[0], minibatch):
                output = self.feedforward(train_data[i:i + minibatch])
                self.backpropagation(self.LossModel.backpropagation(output, train_label[i:i + minibatch]), learning_rate)

            loss = self.loss(self.feedforward(train_data), train_label)
            losses.append(loss)
            if verbose:
                print(f"Epoch {epoch + 1}: Loss = {loss}")

            learning_rate -= decay
            # learning_rate *= multiplier
        return losses

    def train(self, train_data, train_label, valid_data, valid_label, minibatch = 64, epochs = 1000, max_learning_rate = 5e-3, verbose = False):
        if self.LossModel.__class__ == SoftmaxCrossEntropyLoss:
            assert self.layers[-1].__class__ == SoftmaxLayer, "The last layer must be SoftmaxLayer"
        lo = 0
        hi = max_learning_rate
        #ternary search
        max_iter = 0
        while hi - lo > 1e-6 and max_iter > 0:
            max_iter -= 1
            mid1 = lo + (hi - lo) / 3
            mid2 = hi - (hi - lo) / 3
            self.trainUtil(train_data, train_label, epochs, mid1, minibatch)
            loss_1, accuracy_1, f1_macro_1 = self.getResult(valid_data, valid_label)
            self.trainUtil(train_data, train_label, epochs, mid2, minibatch)
            loss_2, accuracy_2, f1_macro_2 = self.getResult(valid_data, valid_label)
            if f1_macro_1 > f1_macro_2:
                hi = mid2
            else:
                lo = mid1

        losses = self.trainUtil(train_data, train_label, epochs, hi, minibatch, verbose = verbose)
        losses = np.array(losses)
        try:
          losses.get()
          losses = losses.get()
        except:
          pass
        return losses

    def reset(self):
        for layer in self.layers:
            layer.reset()

    def predict(self, input):
        input = np.array(input)
        input = input.reshape(-1, self.input_size)
        output = self.feedforward(input)
        return output

    def predictClass(self, input):
        output = self.predict(input)
        output = np.argmax(output, axis = 1)
        output = output.reshape(-1, 1)
        output = output + 1
        return output


    def transform(self):
        for layer in self.layers:
            layer.transform()

    def save(self, filename):
        self.transform()
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)


    def getConfusionMatrix(self, output, label):
        confusion_matrix = np.zeros((26, 26))
        for i in range(label.shape[0]):
            confusion_matrix[int(label[i]) - 1][int(output[i]) - 1] += 1
        return confusion_matrix

    def printConfusionMatrix(self, matrix):
        print("Confusion Matrix:")
        print("Letters ↓", end = " ")
        #print TP TN FP FN for each letter in the confusion matrix in a nice format with fixed width
        print("TP".rjust(4), "TN".rjust(4), "FP".rjust(4), "FN".rjust(4))
        for i in range(26):
            print('   ',chr(ord('A') + i), end = "    ")
            TP = matrix[i][i]
            TN = np.sum(matrix) - np.sum(matrix[i]) - np.sum(matrix[:, i]) + matrix[i][i]
            FP = np.sum(matrix[:, i]) - matrix[i][i]
            FN = np.sum(matrix[i]) - matrix[i][i]
            print(str(int(TP)).rjust(4), str(int(TN)).rjust(4), str(int(FP)).rjust(4), str(int(FN)).rjust(4))


    def getResult(self, input, label, printConfusionMatrix = False):
        output = self.feedforward(input)
        prediction = np.argmax(output, axis = 1) + 1
        loss = self.loss(output, label)
        label = np.argmax(label, axis = 1) + 1

        confusion_matrix = self.getConfusionMatrix(prediction, label)
        if printConfusionMatrix:
            self.printConfusionMatrix(confusion_matrix)
        accuracy = np.sum(prediction == label) / label.shape[0] * 100
        f1_score = np.zeros((26))
        for i in range(26):
            f1_score[i] = 2 * confusion_matrix[i][i] / (np.sum(confusion_matrix[i]) + np.sum(confusion_matrix[:, i]))

        f1_macro = np.mean(f1_score)

        return loss, accuracy, f1_macro

    def adjustDropOut(self):
        for layer in self.layers:
            if layer.__class__ == DropoutLayer:
                layer.transform()

    def printResult(self, input, label, printConfusionMatrix = False):
        loss, accuracy, f1_macro = self.getResult(input, label, printConfusionMatrix = printConfusionMatrix)
        print("=========================================")
        print(f"Loss = {loss}")
        print(f"Accuracy = {accuracy}%")
        print(f"F1 Macro = {f1_macro}")
        print("=========================================")

"""Models"""

#model 3 0.005 good. After that i would choose model 2
# def model1(input_size, output_size):
#     model = FNN(input_size, init_layer=BatchNormalizationLayer)
#     model.add_layer(DenseLayer, 512)
#     model.add_layer(SigmoidLayer)
#     model.add_layer(DropoutLayer, dropout_rate = 0.5)
#     model.add_layer(DenseLayer, 256)
#     model.add_layer(ReluLayer)
#     model.add_layer(DropoutLayer, dropout_rate = 0.2)
#     model.add_layer(DenseLayer, 128)
#     model.add_layer(ReluLayer)
#     model.add_layer(DenseLayer, output_size)
#     model.add_layer(SoftmaxLayer)
#     return model

# def model2(input_size, output_size):
#     model = FNN(input_size, init_layer=BatchNormalizationLayer)
#     model.add_layer(DenseLayer, 512, optimizer='Nadam')
#     model.add_layer(ReluLayer)
#     model.add_layer(DropoutLayer, dropout_rate = 0.5)
#     model.add_layer(DenseLayer, 256, optimizer='Nadam')
#     model.add_layer(SigmoidLayer)
#     model.add_layer(DropoutLayer, dropout_rate = 0.2)
#     model.add_layer(BatchNormalizationLayer)
#     model.add_layer(DenseLayer, 128, optimizer='Nadam')
#     model.add_layer(ReluLayer)
#     model.add_layer(DenseLayer, output_size, optimizer='Nadam')
#     model.add_layer(SoftmaxLayer)
#     return model

# def model3(input_size, output_size):
#     model = FNN(input_size, init_layer=BufferLayer)
#     model.add_layer(DenseLayer, 512, optimizer='None')
#     model.add_layer(ReluLayer)
#     model.add_layer(DropoutLayer, dropout_rate = 0.5)
#     model.add_layer(DenseLayer, 128, optimizer='None')
#     model.add_layer(ReluLayer)
#     model.add_layer(DropoutLayer, dropout_rate = 0.2)
#     model.add_layer(DenseLayer, 128, optimizer='None')
#     model.add_layer(SigmoidLayer)
#     model.add_layer(DenseLayer, output_size, optimizer='None')
#     model.add_layer(SoftmaxLayer)
#     return model

# #model 2 0.001 good. Model 3 less overfit
# def model1(input_size, output_size):
#     model = FNN(input_size, init_layer=BatchNormalizationLayer)
#     model.add_layer(DenseLayer, 512)
#     model.add_layer(ReluLayer)
#     model.add_layer(DropoutLayer, dropout_rate = 0.5)
#     model.add_layer(DenseLayer, 256)
#     model.add_layer(ReluLayer)
#     model.add_layer(BatchNormalizationLayer)
#     model.add_layer(DropoutLayer, dropout_rate = 0.3)
#     model.add_layer(DenseLayer, 256)
#     model.add_layer(ReluLayer)
#     model.add_layer(DenseLayer, output_size)
#     model.add_layer(SoftmaxLayer)
#     return model

# def model2(input_size, output_size):
#     model = FNN(input_size, init_layer=BatchNormalizationLayer)
#     model.add_layer(DenseLayer, 512, optimizer='Nadam')
#     model.add_layer(ReluLayer)
#     model.add_layer(DropoutLayer, dropout_rate = 0.5)
#     model.add_layer(DenseLayer, 256, optimizer='Nadam')
#     model.add_layer(ReluLayer)
#     model.add_layer(DropoutLayer, dropout_rate = 0.3)
#     model.add_layer(BatchNormalizationLayer)
#     model.add_layer(DenseLayer, 256, optimizer='Nadam')
#     model.add_layer(ReluLayer)
#     model.add_layer(DenseLayer, output_size, optimizer='Nadam')
#     model.add_layer(SoftmaxLayer)
#     return model

# def model3(input_size, output_size):
#     model = FNN(input_size, init_layer=BatchNormalizationLayer)
#     model.add_layer(DenseLayer, 512, optimizer='None')
#     model.add_layer(ReluLayer)
#     model.add_layer(DropoutLayer, dropout_rate = 0.5)
#     model.add_layer(DenseLayer, 256, optimizer='None')
#     model.add_layer(ReluLayer)
#     model.add_layer(BatchNormalizationLayer)
#     model.add_layer(DropoutLayer, dropout_rate = 0.3)
#     model.add_layer(DenseLayer, 256, optimizer='None')
#     model.add_layer(ReluLayer)
#     model.add_layer(DenseLayer, output_size, optimizer='None')
#     model.add_layer(SoftmaxLayer)
#     return model

#Best model: Model 3 with learning rate = 0.005
# def model1(input_size, output_size):
#     model = FNN(input_size, init_layer=BatchNormalizationLayer)
#     model.add_layer(DenseLayer, 512)
#     model.add_layer(ReluLayer)
#     model.add_layer(DropoutLayer, dropout_rate = 0.5)
#     model.add_layer(DenseLayer, output_size)
#     model.add_layer(SoftmaxLayer)
#     return model

# def model2(input_size, output_size):
#     model = FNN(input_size, init_layer=BatchNormalizationLayer)
#     model.add_layer(DenseLayer, 512, optimizer='Nadam')
#     model.add_layer(ReluLayer)
#     model.add_layer(DropoutLayer, dropout_rate = 0.5)
#     model.add_layer(DenseLayer, output_size, optimizer='Nadam')
#     model.add_layer(SoftmaxLayer)
#     return model

# def model3(input_size, output_size):
#     model = FNN(input_size, init_layer=BatchNormalizationLayer)
#     model.add_layer(DenseLayer, 512, optimizer='None')
#     model.add_layer(ReluLayer)
#     model.add_layer(DropoutLayer, dropout_rate = 0.5)
#     model.add_layer(DenseLayer, output_size, optimizer='None')
#     model.add_layer(SoftmaxLayer)
#     return model

#Final Pick
def model1(input_size, output_size):
    model = FNN(input_size, init_layer=BatchNormalizationLayer)
    model.add_layer(DenseLayer, 512, optimizer='Nadam')
    model.add_layer(ReluLayer)
    model.add_layer(DropoutLayer, dropout_rate = 0.5)
    model.add_layer(DenseLayer, output_size, optimizer='Nadam')
    model.add_layer(SoftmaxLayer)
    return model

def model2(input_size, output_size):
    model = FNN(input_size, init_layer=BatchNormalizationLayer)
    model.add_layer(DenseLayer, 512)
    model.add_layer(ReluLayer)
    model.add_layer(DropoutLayer, dropout_rate = 0.5)
    model.add_layer(DenseLayer, 256)
    model.add_layer(ReluLayer)
    model.add_layer(BatchNormalizationLayer)
    model.add_layer(DropoutLayer, dropout_rate = 0.3)
    model.add_layer(DenseLayer, 128)
    model.add_layer(ReluLayer)
    model.add_layer(DenseLayer, output_size)
    model.add_layer(SoftmaxLayer)
    return model

def model3(input_size, output_size):
    model = FNN(input_size, init_layer=BufferLayer)
    model.add_layer(DenseLayer, 512, optimizer='None')
    model.add_layer(ReluLayer)
    model.add_layer(DropoutLayer, dropout_rate = 0.5)
    model.add_layer(DenseLayer, 256, optimizer='None')
    model.add_layer(ReluLayer)
    model.add_layer(DropoutLayer, dropout_rate = 0.2)
    model.add_layer(DenseLayer, 128, optimizer='None')
    model.add_layer(SigmoidLayer)
    model.add_layer(DenseLayer, output_size, optimizer='None')
    model.add_layer(SoftmaxLayer)
    return model

"""Training

"""

def read_data():
    train_validation_dataset = ds.EMNIST(root='./data', split='letters',
    train=True,
    transform=transforms.ToTensor(),
    download=True)
    independent_test_dataset = ds.EMNIST(root='./data',
    split='letters',
    train=False,
    transform=transforms.ToTensor())
    return train_validation_dataset, independent_test_dataset

def show(index, dataset):
    image, label = dataset[index]
    print(label)
    print(image)
    plt.imshow(image.reshape(28, 28))
    plt.show()

def preprocess(train_validation, test):
    train, validation = model_selection.train_test_split(train_validation, test_size=0.15, random_state=42)
    train_data = np.array([np.array(x[0]).flatten() for x in train])
    train_labels = np.array([x[1] for x in train])
    validation_data = np.array([np.array(x[0]).flatten() for x in validation])
    validation_labels = np.array([x[1] for x in validation])
    test_data = np.array([np.array(x[0]).flatten() for x in test])
    test_labels = np.array([x[1] for x in test])

    # train_data = train_data / 255
    # validation_data = validation_data / 255
    # test_data = test_data / 255

    # train_data = preprocessing.scale(train_data)
    # validation_data = preprocessing.scale(validation_data)
    # test_data = preprocessing.scale(test_data)

    try:
        train_labels.reshape(-1, 1).get()
        oneHotEncoder = preprocessing.OneHotEncoder().fit(train_labels.reshape(-1, 1).get())
        train_labels = oneHotEncoder.transform(train_labels.reshape(-1, 1).get()).toarray()
        validation_labels = oneHotEncoder.transform(validation_labels.reshape(-1, 1).get()).toarray()
        test_labels = oneHotEncoder.transform(test_labels.reshape(-1, 1).get()).toarray()

        train_labels = np.array(train_labels)
        validation_labels = np.array(validation_labels)
        test_labels = np.array(test_labels)
    except:
        oneHotEncoder = preprocessing.OneHotEncoder().fit(train_labels.reshape(-1, 1))
        train_labels = oneHotEncoder.transform(train_labels.reshape(-1, 1)).toarray()
        validation_labels = oneHotEncoder.transform(validation_labels.reshape(-1, 1)).toarray()
        test_labels = oneHotEncoder.transform(test_labels.reshape(-1, 1)).toarray()

    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels




def main():
    train_validation, test = read_data()
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = preprocess(train_validation, test)


    m1 = model1(train_data.shape[1], train_labels.shape[1])
    m2 = model2(train_data.shape[1], train_labels.shape[1])
    m3 = model3(train_data.shape[1], train_labels.shape[1])
    best_model = None
    losses = []
    models = [m1, m2, m3]
    learning_rates = [5e-3, 1e-3, 5e-4, 1e-4]
    model_losses = []
    model_accuracy = []
    model_f1_macro = []
    best_learning_rate = None
    best_model_index = None
    best_f1 = 0
    for model in models:
        model_losses.append([])
        model_accuracy.append([])
        model_f1_macro.append([])
        for learning_rate in learning_rates:
            print(f"Learning rate = {learning_rate}")
            loss = model.train(train_data, train_labels, validation_data, validation_labels, epochs = 10, max_learning_rate = learning_rate, verbose = False)
            model.adjustDropOut()
            losses.append(loss)
            print("Training result:")
            model.printResult(train_data, train_labels)
            print("Validation result:")
            model.printResult(validation_data, validation_labels)
            res = model.getResult(validation_data, validation_labels)
            model_losses[-1].append(res[0])
            model_accuracy[-1].append(res[1])
            model_f1_macro[-1].append(res[2])
            if res[2] > best_f1:
                best_learning_rate = learning_rate
                best_model_index = len(model_losses) - 1
                best_f1 = model_f1_macro[-1][-1]

    best_model = None
    if best_model_index == 0:
      best_model = model1(train_data.shape[1], train_labels.shape[1])
    if best_model_index == 1:
      best_model = model2(train_data.shape[1], train_labels.shape[1])
    if best_model_index == 2:
      best_model = model3(train_data.shape[1], train_labels.shape[1])


    #plot 4 different learning rates in one graph
    for i in range (len(models)):
        plt.figure()
        plt.title(f"Loss vs Epochs (Model {i + 1})")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        for j in range(len(learning_rates)):
            plt.plot(losses[i * len(learning_rates) + j], label = f"Learning rate = {learning_rates[j]}")
        plt.legend()
        plt.savefig(f"model{i + 1}.png")
        plt.show()

    #plot 3 different models in one graph (learning rate vs loss, accuracy, f1_macro)

    plt.figure()
    plt.title("Learning rate vs Loss")
    plt.xlabel("Learning rate")
    plt.ylabel("Loss")
    learning_rates = np.array(learning_rates)
    try:
        learning_rates.get()
        learning_rates = learning_rates.get()
    except:
        pass

    for i in range(len(models)):
        model_losses[i] = np.array(model_losses[i])
        try:
            model_losses[i].get()
            model_losses[i] = model_losses[i].get()
        except:
            pass
        plt.plot(learning_rates, model_losses[i], label = f"Model {i + 1}")

    plt.legend()
    plt.savefig("loss.png")
    plt.show()

    plt.figure()
    plt.title("Learning rate vs Accuracy")
    plt.xlabel("Learning rate")
    plt.ylabel("Accuracy")
    for i in range(len(models)):
        model_accuracy[i] = np.array(model_accuracy[i])
        try:
            model_accuracy[i].get()
            model_accuracy[i] = model_accuracy[i].get()
        except:
            pass
        plt.plot(learning_rates, model_accuracy[i], label = f"Model {i + 1}")


    plt.legend()
    plt.savefig("accuracy.png")
    plt.show()

    plt.figure()
    plt.title("Learning rate vs F1 Macro")
    plt.xlabel("Learning rate")
    plt.ylabel("F1 Macro")
    for i in range(len(models)):
        model_f1_macro[i] = np.array(model_f1_macro[i])
        try:
            model_f1_macro[i].get()
            model_f1_macro[i] = model_f1_macro[i].get()
        except:
            pass
        plt.plot(learning_rates, model_f1_macro[i], label = f"Model {i + 1}")
    plt.legend()
    plt.savefig("f1_macro.png")
    plt.show()


    print(f"Best model: Model {best_model_index + 1} with learning rate = {best_learning_rate}")
    best_model.train(train_data, train_labels, validation_data, validation_labels, epochs = 200, max_learning_rate = best_learning_rate, verbose = False)
    best_model.adjustDropOut()
    best_model.save("model_1805106.pickle")
    print("Training result:")
    best_model.printResult(train_data, train_labels)
    print("Validation result:")
    best_model.printResult(validation_data, validation_labels)

main()
