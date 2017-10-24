import pandas as p
import tensorflow as tf
import numpy as n
from tensorflow.contrib.learn.python.learn.estimators import estimator
from tensorflow.contrib.tensor_forest.client import random_forest
from tensorflow.contrib.tensor_forest.python import tensor_forest
from randomForestScikitLearn import dataProcessing
from sklearn.model_selection import KFold
from tensorflow.contrib.learn.python.learn import metric_spec
from tensorflow.contrib.tensor_forest.client import eval_metrics
#from tf.metrics import mean_squared_error


#read data directly with tf

if __name__=="__main__":
    #import dataset
    houseData = p.read_csv("housing dataset.csv")

    #all the variables except SalePrice is taken as X variables
    x=houseData.drop(['Alley','PoolQC','MiscFeature','Fence','FireplaceQu','HouseStyle'],axis=1)
    x=dataProcessing(x)
    # Saleprice is assined as target variable
    y=x['SalePrice']
    x=x.drop(['SalePrice'],axis=1)

    kf = KFold(n_splits=10)
    kf.get_n_splits(x)
    error = 0
    for train, test in kf.split(x):
        x_train=n.asarray(x[train].values.tolist())
        y_train=n.asarray(y[train].values.tolist())
        x_test=n.asarray(x[test].values.tolist())
        y_test=n.asarray(y[test].values.tolist())

        x_train=n.float32(x_train)
        y_train=n.float32(y_train)
        x_test=n.float32(x_test)
        y_test=n.float32(y_test)

        sess = tf.Session()
        #building an estimator
        RForestParams=tensor_forest.ForestHParams(num_classes=1, num_features=1460, regression=True, num_trees=10, max_nodes=100)
        regressor = estimator.SKCompat(random_forest.TensorForestEstimator(RForestParams))
        #regressor = random_forest.TensorForestEstimator(RForestParams)
        regressor.fit(x=x_train,y=y_train)
        result = regressor.predict(x_test)

        error += tf.square(tf.subtract(y_test, pred))

    rmse_10cv = n.sqrt(error/len(x))
    #rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_test, result))))
    print("RMSE : ", rmse_10cv)
