import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt
import os
from dataProcessing import dataProcessing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

rng = np.random
learning_rate = 0.01
epochs = 50
display_step = 1

# Network Parameters
n_hidden_1 = 10 # 1st layer number of neurons
n_hidden_2 = 10 # 2nd layer number of neurons

n_input = 263 # number of features
n_classes = 1 # one target

sess = tf.Session()
#Import the data
data = pd.read_csv('housing dataset.csv')
#get the x by droping the dependent variable
x_data=data.drop(['Alley','PoolQC','MiscFeature','Fence','FireplaceQu','HouseStyle'],axis = 1)
x_data=dataProcessing(x_data)    #dataprocessing

#set the dependent variable which is saleprice
y_data=x_data['SalePrice']
x_data=x_data.drop(['SalePrice'],axis = 1)
#Splitting the data int train and test as 70/30
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,test_size=.30)

x_train_array = np.asmatrix(x_train.values.tolist())

#print(x_train_array)
y_train_array = np.asmatrix(y_train.values.tolist())

x_test_array = np.asmatrix(x_test.values.tolist())
y_test_array = np.asmatrix(y_test.values.tolist())

y_train_array = y_train_array[:,None]
y_test_array = y_train_array[:,None]

x_train_array=np.float32(x_train_array)
y_train_array=np.float32(y_train_array)
x_test_array=np.float32(x_test_array)
y_test_array=np.float32(y_test_array)

n_samples = x_train_array.shape[0]
print("rows: ",n_samples)
print("colums : ", x_train_array.shape[1])
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

error = tf.subtract(y_test_array, pred)

total_error = tf.reduce_sum(tf.square(tf.subtract(y_test_array, tf.reduce_mean(y_test_array))))

unexplained_error = tf.reduce_sum(tf.square(tf.subtract(y_test_array, pred)))

# Define loss and optimizer
cost = tf.reduce_mean(tf.square(Y-pred))
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer =  tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


#end new


#points = [[], []]

init = tf.global_variables_initializer()

sess.run(init)

for i in list(range(epochs)):
    for x, y in zip(x_train_array, y_train_array):
        sess.run(optimizer, feed_dict={X: x, Y: y})

rmse = sess.run(tf.sqrt(tf.reduce_mean(tf.square(error))), feed_dict={X: x_test_array, Y: y_test_array})
print("RMSE: ", rmse)



R2 = sess.run(tf.subtract(1.0, tf.divide(unexplained_error, total_error)), feed_dict={X: x_test_array, Y: y_test_array})
print("\nR2 : ",R2)
