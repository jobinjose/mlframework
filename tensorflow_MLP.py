import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

rng = np.random
learning_rate = 0.0001
epochs = 50
display_step = 1

# Network Parameters
n_hidden_1 = 10 # 1st layer number of neurons
n_hidden_2 = 10 # 2nd layer number of neurons

n_input = 80 # number of features
n_classes = 1 # one target

data = pd.read_csv('housing dataset.csv')
#get the x by droping the dependent variable
x_data=data.drop('SalePrice',axis = 1)
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

# Create neural network model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(X, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.square(pred-Y))
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer =  tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

#end new


#points = [[], []]

init = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init)

    for i in list(range(epochs)):
        for (x, y) in zip(x_train_array, y_train_array):
        	sess.run(optimizer, feed_dict={X: x_train_array, Y: y_train_array})

            # Display logs per epoch step
        if (i+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: x_train_array, Y:y_train_array})
            print("Epoch:", '%04d' % (i+1), "cost=", "{:.9f}".format(c), "W=", sess.run(weights), "b=", sess.run(biases))

    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: x_train_array, Y: y_train_array})
    print("Training cost=", training_cost, "W=", sess.run(weights), "b=", sess.run(biases), '\n')

    # Graphic display
    plt.plot(x_train_array, y_train_array, 'ro', label='Original data')
    #plt.plot(x_train_array, sess.run(weights) * x_train_array + sess.run(biases), label='Fitted line')
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
    plt.plot(x_train_array, sess.run(weights) * x_train_array + sess.run(biases), label='Fitted line')
    plt.legend()
    plt.show()
