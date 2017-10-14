import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

rng = np.random
learning_rate = 0.00000000005
epochs = 50
display_step = 1
#def calc(x, y):
	
 #   predictions = tf.add(b, tf.matmul(tf.cast(x, tf.float64), w))
  #  error = tf.reduce_mean(tf.square(y - predictions))
   # return [ predictions, error ]


data = pd.read_csv('housing dataset_numbers.csv')
#get the x by droping the dependent variable
x_data=data.drop('SalePrice',axis = 1)
#x_data=x_data.drop('SaleCondition',axis =1)
#set the dependent variable which is saleprice
y_data=data['SalePrice']
#Splitting the data int train and test as 70/30
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,test_size=.30) 	

x_train_array = np.asarray(x_train.values.tolist())

#print(x_train_array)
y_train_array = np.asarray(y_train.values.tolist())

x_test_array = np.asarray(x_test.values.tolist())
y_test_array = np.asarray(y_test.values.tolist())

y_train_array = y_train_array[:,None]
y_test_array = y_train_array[:,None]

n_samples = x_train_array.shape[0]

# tf Graph Input
X = tf.placeholder("float")
Y = tf.placeholder("float")

#W = tf.Variable(tf.truncated_normal([1022,], mean=0.0, stddev=1.0, dtype=tf.float64))
#b = tf.Variable(tf.zeros(1022, dtype = tf.float64))

W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")

# Construct a linear model
#pred = tf.add(tf.multiply(tf.cast(X, tf.float64), W), b)
pred = tf.add(tf.multiply(X, W), b)

#print(pred)

# Mean squared error
#cost = tf.reduce_mean(tf.square(y_train_array - pred))

cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)

# Gradient descent
#  Note, minimize() knows to modify W and b because Variable objects are trainable=True by default
#= tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


#predictions = tf.add(b, tf.matmul(x_train_array, w))
#error = tf.reduce_mean(tf.square(y_train_array - predictions))

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
        	#print(x_train_array)
        	sess.run(optimizer, feed_dict={X: x_train_array, Y: y_train_array})
            #points[0].append(i+1)
            #points[1].append(sess.run(cost))

            # Display logs per epoch step
        if (i+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: x_train_array, Y:y_train_array})
            print("Epoch:", '%04d' % (i+1), "cost=", "{:.9f}".format(c), "W=", sess.run(W), "b=", sess.run(b))


        #if i % 100 == 0:
        #    print(sess.run(cost))
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

    #valid_cost = calc(valid_features, valid_prices)[1]

    #print('Validation error =', sess.run(valid_cost), '\n')

    #test_cost = calc(x_test_array, y_test_array)[1]

    #print(test_cost)

    # print('Test error =', sess.run(test_cost), '\n')

    

