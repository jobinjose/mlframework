import tensorflow as tf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
from sklearn.model_selection import KFold
from dataProcessing import dataProcessing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

rng = np.random
learning_rate = 0.00000000005
epochs = 50
display_step = 1

#Import the data
data = pd.read_csv('housing dataset.csv')
#get the x by droping the dependent variable
x_data=data.drop(['Alley','PoolQC','MiscFeature','Fence','FireplaceQu','HouseStyle','SalePrice'],axis = 1)
x_data=dataProcessing(x)    #dataprocessing

#set the dependent variable which is saleprice
y_data=data['SalePrice']
#10 fold cross validation
kf = KFold(n_splits=10)
kf.get_n_splits(x_data)
error = 0

for train, test in kf.split(x_data):
    x_train_array = np.asarray(x_data[train].values.tolist())
    y_train_array = np.asarray(y_data[train].values.tolist())

    x_test_array = np.asarray(x_data[test].values.tolist())
    y_test_array = np.asarray(y_data[test].values.tolist())

    y_train_array = y_train_array[:,None]
    y_test_array = y_train_array[:,None]

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
    #tf.reduce_mean(tf.square(y_train_array - pred))

    #y,cost = calc(x_train_array, y_train_array)


    #points = [[], []]

    init = tf.global_variables_initializer()
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    with tf.Session() as sess:

        sess.run(init)

        for i in list(range(epochs)):
            #sess.run(optimizer)
            #if i % 10 == 0.:
            for (x, y) in zip(x_train_array, y_train_array):
            	sess.run(optimizer, feed_dict={X: x_train_array, Y: y_train_array})

                # Display logs per epoch step
            if (i+1) % display_step == 0:
                c = sess.run(cost, feed_dict={X: x_train_array, Y:y_train_array})
                print("Epoch:", '%04d' % (i+1), "cost=", "{:.9f}".format(c), "W=", sess.run(W), "b=", sess.run(b))

        print("Optimization Finished!")
        training_cost = sess.run(cost, feed_dict={X: x_train_array, Y: y_train_array})
        print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

        #plt.plot(points[0], points[1], 'r--')
        #plt.axis([0, epochs, 50, 600])
        #plt.show()

        # Graphic display
        plt.plot(x_train_array, y_train_array, 'ro', label='Original data')
        plt.plot(x_train_array, sess.run(W) * x_train_array + sess.run(b), label='Fitted line')
        plt.legend()
        plt.show()

        #testing

        print("Testing... (Mean square loss Comparison)")
        testing_cost = sess.run(
            tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * x_test_array.shape[0]),
            feed_dict={X: x_test_array, Y: y_test_array})  # same function as cost above
        print("Testing cost=", testing_cost)
        print("Absolute mean square loss difference:", abs(
            training_cost - testing_cost))

        plt.plot(x_test_array, y_test_array, 'bo', label='Testing data')
        plt.plot(x_train_array, sess.run(W) * x_train_array + sess.run(b), label='Fitted line')
        plt.legend()
        plt.show()

rmse_10cv = np.sqrt(error/len(x_data))
print("Root mean Squre Error : ", rmse_10cv)
