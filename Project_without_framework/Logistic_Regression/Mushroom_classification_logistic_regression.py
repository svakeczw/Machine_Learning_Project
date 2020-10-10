#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


# import dataset
data = pd.read_csv('/Users/zhangweichen/OneDrive - UTS/Machine_Learning_Project/Project_without_framework/Logistic_Regression/mushrooms.csv')


# data preprocess
def map_value(dataset):
    encoder = LabelEncoder()
    for col in data.columns:
        data[col] = encoder.fit_transform(data[col])
    return dataset


data_new = map_value(data)
simple_x = data_new.iloc[:,1:23].values
simple_y = data_new[['class']].values
simple_x = np.concatenate((np.ones((simple_x.shape[0], 1)), simple_x), axis=1)
theta_init = np.zeros([simple_x.shape[1],1])
simple_x_train, simple_x_valid, simple_y_train, simple_y_valid = train_test_split(simple_x, simple_y)
sample_size = simple_x_train.shape[0]


# build model
class LogisticRegressionModel:
    def __init__(self, theta):
        self.final_theta = np.zeros([theta.shape[0], 1])
        self.final_cost = None

    def sigmoid(self, z):
        """
        sigmoid function
        :param z: linear part input
        :return:
        sigmoid output
        """
        return 1 / (1 + np.exp(-z))

    def activation(self, x, theta):
        """
        sigmoid activation
        :param x: feature matrix
        :param theta: weight matrix
        :return:
        s: activation output
        """
        s1 = np.dot(x, theta)
        s = self.sigmoid(s1)

        assert s.shape == (x.shape[0], 1)  # shape check

        return s

    def cost(self, x, y, theta, m):
        """
        compute cost
        :param x: feature matrix
        :param y: class label matrix
        :param theta: weight matrix
        :param m: size of data
        :return:
        s: cost
        """
        a = np.multiply(y, np.log(self.activation(x, theta)))
        b = np.multiply(1 - y, np.log(1 - self.activation(x, theta)))
        s = np.sum(a + b) / (-m)

        assert s.shape == ()  # shape check

        return s

    def gradient(self, x, y, theta):
        """
        compute gradient
        :param x: feature matrix
        :param y: class label matrix
        :param theta: weight matrix
        :return:
        grad: gradient
        """
        grad = x.T.dot(self.activation(x, theta) - y)

        assert grad.shape == theta.shape  # shape check

        return grad

    def descent(self, x, y, iteration_num, m, initial_alpha, theta, lambd, decay_rate):
        """
        training model by gradient descent
        :param x: feature matrix
        :param y: class label matrix
        :param iteration_num: iteration number
        :param m: size of data
        :param initial_alpha: initial learning rate alpha
        :param theta: weight matrix
        :param lambd: regularization term lambda
        :param decay_rate: decay rate of learning rate
        :return:
        final_theta: trained weight matrix
        final_cost: cost of last training
        """
        i = 1
        costs = [self.cost(x, y, theta, m)]
        alpha = initial_alpha
        while True:
            grad = self.gradient(x, y, theta)
            theta = theta - (alpha / m) * grad + (lambd / m) * theta  # update theta
            current_cost = self.cost(x, y, theta, m)
            costs.append(current_cost)

            # print iteration number and cost every 100 iterations
            if i % 100 == 0:
                print('iteration: ' + str(i) + ' cost: ' + str(current_cost) + ' alpha: ' + str(alpha))

            i += 1
            alpha = (1 / (1 + decay_rate * i)) * initial_alpha  # decrease learning rate

            if i > iteration_num:
                self.final_theta = theta
                self.final_cost = costs
                break

        # plot the cost trend
        plt.plot(range(0, len(costs)), costs)
        plt.xlabel('iteration num')
        plt.ylabel('cost')
        plt.show()
        return self.final_theta, self.final_cost

    def predict(self, x, theta):
        """
        predict new data
        :param x: feature
        :param theta: weight
        :return:
        p: predicted label
        """
        p = self.activation(x, theta)
        for i in range(len(p)):

            if p[i] > 0.5:
                p[i] = 1
            else:
                p[i] = 0
        return p


# training
myModel = LogisticRegressionModel(theta_init)
myModel.descent(simple_x_train, simple_y_train, 1500, sample_size, 0.07, theta_init, lambd=2, decay_rate=0)
theta_final = myModel.final_theta


# predict
predicted = myModel.predict(simple_x_valid, theta_final)
print(predicted)
print(simple_y_valid)
valid_corr_num = (predicted == simple_y_valid).sum()
accuracy = valid_corr_num/len(simple_y_valid)
print(accuracy)

