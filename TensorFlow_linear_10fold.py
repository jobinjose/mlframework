import tensorflow as tf
import pandas as pd
from sklearn.model_selection import KFold
import numpy as np
import os
from dataProcessing_kc_data import dataProcessing_kc_data
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from config import dataset1,learning_rate,epochs, no_of_folds

rng = np.random
#learning_rate = 0.0000000000000005e-5
#epochs = 50
#display_step = 50
#no_of_folds = 10
sess = tf.Session()

#Import the data
data = pd.read_csv(dataset1)
#get the x by droping the dependent variable
#x_data=data.drop(['Alley','PoolQC','MiscFeature','Fence','FireplaceQu','HouseStyle'],axis = 1)
x_data=dataProcessing_kc_data(data)    #dataprocessing

#set the dependent variable which is saleprice
y_data=x_data['price']
x_data=x_data.drop(['price'],axis = 1)

kf = KFold(n_splits=no_of_folds)
kf.get_n_splits(x_data)
error = 0
rmse = []
r2 = []
rmse_10fold = 0
m=0

for train, test in kf.split(x_data):
    m+=1
    print("Iteration : ", m)
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
    #for RMSE
    error = tf.subtract(y_test_array, pred)
    #for R2
    total_error = tf.reduce_sum(tf.square(tf.subtract(y_test_array, tf.reduce_mean(y_test_array))))
    unexplained_error = tf.reduce_sum(tf.square(tf.subtract(y_test_array, pred)))

    cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)


    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)



    init = tf.global_variables_initializer()
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    sess.run(init)

    for i in list(range(epochs)):
    	#sess.run(optimizer)
    	#if i % 10 == 0.:
    	#for (x, y) in zip(x_train_array, y_train_array):
    		#print(x_train_array)
    	sess.run(optimizer, feed_dict={X: x_train_array, Y: y_train_array})

    rmse.append(sess.run(tf.sqrt(tf.reduce_mean(tf.square(error))), feed_dict={X: x_test_array, Y: y_test_array}))
    r2.append(sess.run(tf.subtract(1.0, tf.divide(unexplained_error, total_error)), feed_dict={X: x_test_array, Y: y_test_array}))

print("\nRMSE : ", sum(rmse)/no_of_folds)
print("\nR2 : ",sum(r2)/no_of_folds)
