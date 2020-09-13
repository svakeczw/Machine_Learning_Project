#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# import dataset
data = pd.read_csv('/Users/zhangweichen/OneDrive - UTS/Machine_Learning_Project/Project_without_framework/Perceptron/mushrooms_data.csv')


# data preprocess
def map_value(dataset):
    encoder = LabelEncoder()
    for col in data.columns:
        data[col]=encoder.fit_transform(dataset[col])
    return dataset


data_new = map_value(data)
X=data_new.iloc[:,1:23]
y=data_new[['class']]

simple_x = data_new.iloc[:,1:].values
print(simple_x)
simple_y = data_new.iloc[:,0].values

simple_y[simple_y==0] = -1.0
print(simple_y.shape)

simple_x_train, simple_x_valid, simple_y_train, simple_y_valid = train_test_split(simple_x, simple_y)
print(simple_y)


# build model
class MyPerceptron(object):
    def __init__(self,X):
        # self.h = np.array(np.zeros(X.shape[1]+1))
        self.h = np.array(np.zeros(X.shape[1]+1))

    def linear_function(self, X, h):
        """Compute the linear function"""
        # x =  np.concatenate((X,np.ones((X.shape[0],1))),axis=1)
        # s = x.dot(h.T)
        s = (X * h[:-1]).sum(axis=1) + h[-1]
        return s

    def predict_with_(self, X, h):
        # return 0/1 according to the linear function

        return np.sign(self.linear_function(X, h)).astype(np.int)

    def predict(self, X):
        return self.predict_with_(X, self.h)

    def fit(self, X, y,n):
        """
        :param X: training samples
        :param y: the known answer for each sample

        """
        ii = 0
        while True:

            # predict using the current h
            predicted = self.predict_with_(X,self.h)

            # find errors
            error_indexes = np.nonzero(predicted != y)[0]

            if len(error_indexes) > 0:
                i = error_indexes[np.random.randint(len(error_indexes))]
                self.h[:-1] += X[i] * float(y[i])
                # To update the weights

                self.h[-1] += float(y[i])
                # To update the b-bias

                print(f"{ii} Train errors: {len(error_indexes)}")

                ii += 1

                if ii >= n:  # iteration size
                    break
            else:
                return self.h


# training model
my_model = MyPerceptron(simple_x_train) # initiate an object
my_model.fit(simple_x_train, simple_y_train,5000)
final_h = my_model.h
print(final_h)


# accuracy
pred_valid = my_model.predict(simple_x_valid)
print(pred_valid)
print(simple_y_valid)
valid_corr_num = (pred_valid == simple_y_valid).sum()
print("Valid accu", valid_corr_num / len(simple_y_valid))


