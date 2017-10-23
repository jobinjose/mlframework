import pandas as p
import tensorflow as tf
import numpy as n
from tensorflow.contrib.learn.python.learn.estimators import estimator
from tensorflow.contrib.tensor_forest.client import random_forest
from tensorflow.contrib.tensor_forest.python import tensor_forest
from randomForestScikitLearn import dataProcessing
from sklearn.model_selection import train_test_split
from tensorflow.contrib.learn.python.learn import metric_spec
from tensorflow.contrib.tensor_forest.client import eval_metrics
import os
#from tf.metrics import mean_squared_error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.ERROR)

#read data directly with tf
'''
datafile = tf.train.string_input_producer(["housing dataset.csv"])
reader = tf.TextLineReader(skip_header_lines=1)
key,value = reader.read(datafile)
datacontent = tf.decode_csv(value)
print(datacontent)
'''
'''
def get_input_fn(data_set, num_epochs=None, shuffle=False):
	Features = list(data_set);
	return tf.estimator.inputs.pandas_input_fn(x=p.DataFrame({i: data_set[i].values for i in Features}),y=p.Series(data_set['SalePrice'].values),num_epochs=num_epochs,shuffle=shuffle)
'''

if __name__=="__main__":
    #import dataset
    houseData = p.read_csv("housing dataset.csv")

    #all the variables except SalePrice is taken as X variables
    x=houseData.drop(['Alley','PoolQC','MiscFeature','Fence','FireplaceQu','HouseStyle'],axis=1)
    x=dataProcessing(x)
    # Saleprice is assined as target variable
    y=x['SalePrice']
    x=x.drop(['SalePrice'],axis=1)

    # Splitting the dataset into training set(70%) and test set (30%)
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30)
    #print(x_train) #1022 rows
    #print(x_test) #438 rows



    '''
    # train data processing
    x_train=dataProcessing(x_train)
    x_test=dataProcessing(x_test)

    x_train['label'] = "train"
    x_test['label'] = "test"
    # concatenated two datasets
    x_concat = p.concat([x_train,x_test])
    x_concatdummies = p.get_dummies(x_concat)
    x_train = x_concatdummies[x_concatdummies['label_train'] == 1]
    x_test = x_concatdummies[x_concatdummies['label_test'] == 1]

    # Drop your labels
    x_train = x_train.drop(['label_train','label_test'], axis=1)
    x_test = x_test.drop(['label_train','label_test'], axis=1)
    #print(y_train.head())
    #x_train['SalePrice'] = y_train
    #print(x_train['SalePrice'])
    '''


    #y_train = y_train.as_matrix;


    #x_train=tf.cast(x_train,tf.float32)

    '''
    x_train['LotFrontage']=n.float32(x_train['LotFrontage'])
    x_train['MasVnrArea']=n.float32(x_train['MasVnrArea'])
    x_train['GarageYrBlt']=n.float32(x_train['GarageYrBlt'])

    x_train['LotFrontage']=x_train['LotFrontage'].astype('float32')
    x_train['MasVnrArea']=x_train['MasVnrArea'].astype('float32')
    x_train['GarageYrBlt']=x_train['GarageYrBlt'].astype('float32')
    '''
    #print(x_train.select_dtypes(include=['float64']))

    #x_trin=p.DataFrame(x_train,dtype='float);



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
    #rint(sess.run(x_train_tensor))
    '''
    file = open("outputfile.txt","w")
    sess.run(x_train_tensor).tofile(file,"","%s")
    file.close()
    '''
    #building an estimator
    RForestParams=tensor_forest.ForestHParams(num_classes=1, num_features=1460, regression=True, num_trees=10, max_nodes=100)
    regressor = estimator.SKCompat(random_forest.TensorForestEstimator(RForestParams))
    #regressor = random_forest.TensorForestEstimator(RForestParams)
    regressor.fit(x=x_train,y=y_train)

    result = regressor.predict(x_test)
    #print(result)

    '''
    metric_name = 'accuracy'
    metrics_cal = {metric_name:metric_spec.MetricSpec(eval_metrics.get_metric(metric_name),prediction_key=eval_metrics.get_prediction_key(metric_name))}
    result = regressor.score(x=x_test,y=y_test[1],metrics=metrics_cal)
    '''

    #print(metrics_cal)
    #for key in sorted(result):
    	#print(key)
    	#print('%s : %s' % (key,result[key]))

    #rmse
    #rmse = mean_squared_error(y_test,result)
    #print(rmse)
    rmse_tensor = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_test, result['scores']))))
    rmse = sess.run(rmse_tensor)

    print("RMSE: ", rmse)



    '''
    y_test = tf.constant(y_test)
    result = tf.constant(result)
    correct_prediction = tf.equal(tf.argmax(result,1), tf.argmax(y_test,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy))
    '''
