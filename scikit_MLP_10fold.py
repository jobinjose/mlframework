import sklearn
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn import datasets
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from dataProcessing_kc_data import dataProcessing_kc_data
from sklearn.model_selection import KFold
from config import dataset1, hiddenlayersizes, no_of_folds

#input parameters
#dataset file is entered here
inputdataset = dataset1
#number of hidden layers in th perceptron
#hiddenlayersizes = 30,30,30

#Import the data
data = pd.read_csv(dataset1)
#get the x by droping the dependent variable
#x=data.drop(['Alley','PoolQC','MiscFeature','Fence','FireplaceQu','HouseStyle'],axis = 1)
x=dataProcessing_kc_data(data)    #dataprocessing

#set the dependent variable which is saleprice
y=x['price']
x=x.drop(['price'],axis = 1)

#no_of_folds = 10
kf = KFold(n_splits=no_of_folds)
kf.get_n_splits(x)
xval_err = 0
RMSE = []
R2 = []

for train,test in kf.split(x):
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
