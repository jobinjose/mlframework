import sklearn
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn import datasets
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from dataProcessing_NYC import dataProcessing_NYC
from sklearn.model_selection import KFold

#input parameters
#number of hidden layers in th perceptron
hiddenlayersizes = 30,30,30

#Import the data
data = pd.read_csv('New York City Taxi Trip Duration.csv')
x=dataProcessing_NYC(data)    #dataprocessing

#set the dependent variable which is saleprice
y=x['trip_duration']
x=x.drop(['dropoff_datetime','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','trip_duration'],axis = 1)

no_of_folds = 10
kf = KFold(n_splits=no_of_folds)
kf.get_n_splits(x)
xval_err = 0
RMSE = []
R2 = []
i=0
for train,test in kf.split(x):
    i+=1
    print("Iteration No : ",i)
    #multi layer perceptron
    mlp = MLPRegressor(hidden_layer_sizes=(hiddenlayersizes))

    #train the model
    mlp.fit(x.iloc[train],y.iloc[train])

    # test set predictions
    y_pred = mlp.predict(x.iloc[test])

    RMSE.append(np.sqrt(mean_squared_error(y.iloc[test],y_pred)))
    R2.append(r2_score(y.iloc[test],y_pred))

    # Metrics
    #Coefficient
    #print('Coefficients: \n', mlp.coefs_)
    # The mean squared error
print("RMSE: " ,sum(RMSE)/no_of_folds)
    # Explained variance score: 1 is perfect prediction
print('R2: ' , sum(R2)/no_of_folds)
