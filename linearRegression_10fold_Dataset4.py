import sklearn
import csv
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn import datasets, linear_model
from sklearn.model_selection import KFold
from dataProcessing_NYC import dataProcessing_NYC
from sklearn.metrics import r2_score, mean_squared_error
from config_task1 import n_splits
from config_task1 import chunk_start_size
from config_task1 import dataset4

#n_splits = 10
chunk_split_start_loop_size = chunk_start_size

#Import the data
dataset = pd.read_csv(dataset4)
#get the x by droping the dependent variable
data=dataProcessing_NYC(dataset)    #dataprocessing
data=data.drop(['dropoff_datetime','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude'],axis = 1)
#chunk_split_start_loop_size = 100
flag=1
while chunk_split_start_loop_size <= data.shape[0]:
    x=data.head(chunk_split_start_loop_size)
    #set the dependent variable which is saleprice
    y=x['trip_duration']
    x=x.drop(['trip_duration'],axis=1)
    #linear regression object creation
    kf = KFold(n_splits=n_splits)
    kf.get_n_splits(x)
    xval_err = 0
    R2 = []
    RMSE = []
    for train,test in kf.split(x):
        linear_reg = linear_model.LinearRegression()
        linear_reg.fit(x.iloc[train],y.iloc[train])
        y_pred = linear_reg.predict(x.iloc[test])
        #e=y_pred - y.iloc[test]
        #xval_err +=np.dot(e,e)
        RMSE.append(np.sqrt(mean_squared_error(y.iloc[test],y_pred)))
        R2.append(r2_score(y.iloc[test],y_pred))
    #train the model


    # test set predictions

    #rmse_10cv = np.sqrt(xval_err/len(x))


    # Metrics
    #Coefficient
    #print('Coefficients: \n', linear_reg.coef_)
    # The mean squared error
    #print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
    # Explained variance score: 1 is perfect prediction
    #print('Variance score: %.2f' % r2_score(y_test, y_pred))

    print("RMSE on 10 fold for chunk size ",chunk_split_start_loop_size,": ", sum(RMSE)/n_splits)
    print("R2 on 10 fold for chunk size ",chunk_split_start_loop_size,": ", sum(R2)/n_splits)

    if flag == 1:
        chunk_split_start_loop_size=chunk_split_start_loop_size*5
        flag = 0
    else:
        chunk_split_start_loop_size=chunk_split_start_loop_size*2
        flag = 1
