import sklearn
import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from dataProcessing import dataProcessing

testsize = .30
#Import the data
data = pd.read_csv('housing dataset.csv')
#get the x by droping the dependent variable
data=data.drop(['Alley','PoolQC','MiscFeature','Fence','FireplaceQu','HouseStyle'],axis = 1)
data=dataProcessing(data)    #dataprocessing
chunk_split_start_loop_size = 100
flag=1
while chunk_split_start_loop_size <= data.shape[0]:
    x=data.head(chunk_split_start_loop_size)
    #set the dependent variable which is saleprice
    y=x['SalePrice']
    x=x.drop(['SalePrice'],axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=testsize)

    #linear regression object creation
    linear_reg = linear_model.LinearRegression()

    #train the model
    linear_reg.fit(x_train,y_train)

    # test set predictions
    y_pred = linear_reg.predict(x_test)
    #e=y_pred - y_test
    #xval_err =np.dot(e,e)
    #rmse_10cv = np.sqrt(xval_err/len(x))
    RMSE = np.sqrt(mean_squared_error(y_test,y_pred))
    R2 = r2_score(y_test,y_pred)
    print("RMSE for chunk size ",chunk_split_start_loop_size,": ", RMSE)
    print("R2 for chunk size ",chunk_split_start_loop_size,": ", R2)

    # Metrics
    #Coefficient
    #print('Coefficients: \n', linear_reg.coef_)
    # The mean squared error
    #print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
    # Explained variance score: 1 is perfect prediction
    #print('Variance score: %.2f' % r2_score(y_test, y_pred))
    if flag == 1:
        chunk_split_start_loop_size=chunk_split_start_loop_size*5
        flag = 0
    else:
        chunk_split_start_loop_size=chunk_split_start_loop_size*2
        flag = 1
