import pandas as p
import tensorflow as tf
import numpy as n
from tensorflow.contrib.learn.python.learn.estimators import estimator
from tensorflow.contrib.tensor_forest.client import random_forest
from tensorflow.contrib.tensor_forest.python import tensor_forest
from dataProcessing_kc_data import dataProcessing_kc_data
from sklearn.model_selection import train_test_split
from tensorflow.contrib.learn.python.learn import metric_spec
from tensorflow.contrib.tensor_forest.client import eval_metrics
import os
from config import dataset1,no_of_trees,testsize
#from tf.metrics import mean_squared_error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.ERROR)

#no_of_trees = 10

if __name__=="__main__":
    #import dataset
    houseData = p.read_csv(dataset1)

    #all the variables except SalePrice is taken as X variables
    #x=houseData.drop(['Alley','PoolQC','MiscFeature','Fence','FireplaceQu','HouseStyle'],axis=1)
    x=dataProcessing_kc_data(houseData)
    # Saleprice is assined as target variable
    y=x['price']
    x=x.drop(['price'],axis=1)

    # Splitting the dataset into training set(70%) and test set (30%)
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=testsize)
    #print(x_train) #1022 rows
    #print(x_test) #438 rows

    #y_train = y_train.as_matrix;
    #x_train_tensor = tf.constant(x_train)
    #y_train_tensor = tf.constant(y_train)


    #print(x_train.values)
    x_train=n.asarray(x_train.values.tolist())
    y_train=n.asarray(y_train.values.tolist())
    x_test=n.asarray(x_test.values.tolist())
    y_test=n.asarray(y_test.values.tolist())


    x_train=n.float32(x_train)
    y_train=n.float32(y_train)
    x_test=n.float32(x_test)
    y_test=n.float32(y_test)


    sess = tf.Session()
    #building an estimator
    #number_features = x_train.shape[0]
    #print(number_features)
    number_features = 1460
    RForestParams=tensor_forest.ForestHParams(num_classes=1, num_features=number_features, regression=True, num_trees=no_of_trees,max_nodes=100)
    regressor = estimator.SKCompat(random_forest.TensorForestEstimator(RForestParams))
    #regressor = random_forest.TensorForestEstimator(RForestParams)
    regressor.fit(x=x_train,y=y_train)

    result = regressor.predict(x_test)
    #print(result)

    #rmse
    #rmse = mean_squared_error(y_test,result)
    #print(rmse)
    rmse_tensor = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_test, result['scores']))))
    rmse = sess.run(rmse_tensor)

    print("RMSE: ", rmse)

	#R2
    total_error = tf.reduce_sum(tf.square(tf.subtract(y_test, tf.reduce_mean(y_test))))

    unexplained_error = tf.reduce_sum(tf.square(tf.subtract(y_test, result['scores'])))
    R2 = sess.run(tf.subtract(1.0, tf.divide(unexplained_error, total_error)))
    print("\nR2 : ",R2)
