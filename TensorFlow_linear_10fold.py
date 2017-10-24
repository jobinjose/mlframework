import tensorflow as tf
import pandas as pd
from sklearn.model_selection import KFold
import numpy as np
import os
from dataProcessing import dataProcessing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

rng = np.random
learning_rate = 0.00000000005
epochs = 50
display_step = 1
no_of_folds = 10
sess = tf.Session()
#def calc(x, y):

 #   predictions = tf.add(b, tf.matmul(tf.cast(x, tf.float64), w))
  #  error = tf.reduce_mean(tf.square(y - predictions))
   # return [ predictions, error ]


#Import the data
data = pd.read_csv('housing dataset.csv')
#get the x by droping the dependent variable
x_data=data.drop(['Alley','PoolQC','MiscFeature','Fence','FireplaceQu','HouseStyle','SalePrice'],axis = 1)
x_data=dataProcessing(x_data)    #dataprocessing

#set the dependent variable which is saleprice
y_data=data['SalePrice']

kf = KFold(n_splits=no_of_folds)
kf.get_n_splits(x_data)
error = 0
rmse = []
rmse_10fold = 0

for train, test in kf.split(x_data):
    x_train_array = np.asarray(x_data.iloc[train].values.tolist())

    #print(x_train_array)
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


    pred = tf.add(tf.multiply(X, W), b)

    error = tf.subtract(y_test_array, pred)

    cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)


    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)



    init = tf.global_variables_initializer()
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    sess.run(init)

    for i in list(range(epochs)):
    	#sess.run(optimizer)
    	#if i % 10 == 0.:
    	for (x, y) in zip(x_train_array, y_train_array):
    		#print(x_train_array)
    		sess.run(optimizer, feed_dict={X: x, Y: y})

    rmse.append(sess.run(tf.sqrt(tf.reduce_mean(tf.square(error))), feed_dict={X: x_test_array, Y: y_test_array}))

print("\nRMSE : ", sum(rmse)/no_of_folds)
