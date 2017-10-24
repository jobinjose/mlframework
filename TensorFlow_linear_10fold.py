import tensorflow as tf
import pandas as pd
import numpy as np
#from matplotlib import pyplot as plt
import os
from sklearn.model_selection import KFold
from dataProcessing import dataProcessing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

rng = np.random
learning_rate = 0.00000000005
epochs = 10
display_step = 1

#Import the data
data = pd.read_csv('housing dataset.csv')
#get the x by droping the dependent variable
x_data=data.drop(['Alley','PoolQC','MiscFeature','Fence','FireplaceQu','HouseStyle','SalePrice'],axis = 1)
x_data=dataProcessing(x_data)    #dataprocessing

#set the dependent variable which is saleprice
y_data=data['SalePrice']
#10 fold cross validation
kf = KFold(n_splits=10)
kf.get_n_splits(x_data)
error = 0
iter=0
sess = tf.Session()
X = tf.placeholder("float")
Y = tf.placeholder("float")

W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")

for train, test in kf.split(x_data):
    iter+=1
    print("Iteration No : ",iter)
    x_train_array = np.asarray(x_data.iloc[train].values.tolist())
    y_train_array = np.asarray(y_data.iloc[train].values.tolist())

    x_test_array = np.asarray(x_data.iloc[test].values.tolist())
    y_test_array = np.asarray(y_data.iloc[test].values.tolist())

    y_train_array = y_train_array[:,None]
    y_test_array = y_train_array[:,None]

    x_train_array=np.float32(x_train_array)
    y_train_array=np.float32(y_train_array)
    x_test_array=np.float32(x_test_array)
    y_test_array=np.float32(y_test_array)

    n_samples = x_train_array.shape[0]

    # tf Graph Input
    X = tf.placeholder("float")
    Y = tf.placeholder("float")

    W = tf.Variable(rng.randn(), name="weight")
    b = tf.Variable(rng.randn(), name="bias")

    # Construct a linear model
    pred = tf.add(tf.multiply(X, W), b)

    # Mean squared error
    cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


    #predictions = tf.add(b, tf.matmul(x_train_array, w))
    error += tf.square(tf.subtract(y_test_array, pred))

    init = tf.global_variables_initializer()
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    sess.run(init)

    for i in list(range(epochs)):
        #sess.run(optimizer)
        #if i % 10 == 0.:
        for (x, y) in zip(x_train_array, y_train_array):
            sess.run(optimizer, feed_dict={X: x_train_array, Y: y_train_array})

rmse_tensor = tf.sqrt(tf.reduce_mean(error))
rmse = sess.run(rmse_tensor, feed_dict={X: x_train_array, Y: y_train_array})
print("Root mean Squre Error : ", rmse)
