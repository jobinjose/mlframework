import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import os
from dataProcessing_kc_data import dataProcessing_kc_data
from sklearn.model_selection import KFold
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

rng = np.random
learning_rate = 0.001
epochs = 50
display_step = 1

# Network Parameters
n_hidden_1 = 10 # 1st layer number of neurons
n_hidden_2 = 10 # 2nd layer number of neurons

no_of_folds = 10

#n_input = 263 # number of features
#n_classes = 1 # one target

sess = tf.Session()
#Import the data
data = pd.read_csv('kc_house_data.csv')
#get the x by droping the dependent variable
#x_data=data.drop(['Alley','PoolQC','MiscFeature','Fence','FireplaceQu','HouseStyle'],axis = 1)
x_data=dataProcessing_kc_data(data)    #dataprocessing

#set the dependent variable which is saleprice
y_data=x_data['price']
x_data=x_data.drop(['price'],axis = 1)
#Splitting the data int train and test as 70/30
kf = KFold(n_splits=no_of_folds)
kf.get_n_splits(x_data)
error = 0
rmse = []
r2 = []
rmse_10fold = 0
for train, test in kf.split(x_data):
    x_train_array = np.asmatrix(x_data.iloc[train].values.tolist())

    #print(x_train_array)
    y_train_array = np.asmatrix(y_data.iloc[train].values.tolist())

    x_test_array = np.asmatrix(x_data.iloc[test].values.tolist())
    y_test_array = np.asmatrix(y_data.iloc[test].values.tolist())

    y_train_array = y_train_array[:,None]
    y_test_array = y_train_array[:,None]

    x_train_array=np.float32(x_train_array)
    y_train_array=np.float32(y_train_array)
    x_test_array=np.float32(x_test_array)
    y_test_array=np.float32(y_test_array)

    n_samples = x_train_array.shape[0]

    n_input = x_train_array.shape[1] # number of features
    n_classes = 1 # one target

    #print("rows: ",n_samples)
    #print("colums : ", x_train_array.shape[1])
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
        out = tf.sigmoid(out_layer)
        return out

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
    #cost = tf.reduce_mean(tf.square(Y-pred))
    cost = tf.reduce_mean(tf.square(Y-pred))
    #cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
    #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
    #optimizer =  tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
    optimizer =  tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


    #end new


    #points = [[], []]

    init = tf.global_variables_initializer()

    sess.run(init)

    for i in list(range(epochs)):
        for x, y in zip(x_train_array, y_train_array):
            sess.run(optimizer, feed_dict={X: x, Y: y})

    rmse.append(sess.run(tf.sqrt(tf.reduce_mean(tf.square(error))), feed_dict={X: x_test_array, Y: y_test_array}))
    r2.append(sess.run(tf.subtract(1.0, tf.divide(unexplained_error, total_error)), feed_dict={X: x_test_array, Y: y_test_array}))

print("RMSE: ", sum(rmse)/no_of_folds)
print("\nR2 : ",sum(r2)/no_of_folds)
