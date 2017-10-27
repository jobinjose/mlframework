import sklearn
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn import datasets
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from dataProcessing_kc_data import dataProcessing_kc_data
from config import dataset1,testsize,hiddenlayersizes

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

#split the data into 70:30 (train:test)
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=testsize)

#multi layer perceptron
mlp = MLPRegressor(hidden_layer_sizes=(hiddenlayersizes))

#train the model
mlp.fit(x_train,y_train)

# test set predictions
y_pred = mlp.predict(x_test)

# Metrics
#Coefficient
#print('Coefficients: \n', mlp.coefs_)
# The mean squared error
print("RMSE: " ,np.sqrt(mean_squared_error(y_test, y_pred)))
# Explained variance score: 1 is perfect prediction
print('R2: ' , r2_score(y_test, y_pred))
