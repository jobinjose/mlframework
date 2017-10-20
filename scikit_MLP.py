import sklearn
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn import datasets
from sklearn.metrics import mean_squared_error, r2_score

#input parameters
#dataset file is entered here
inputdataset = 'housing dataset.csv'
#number of hidden layers in th perceptron
hiddenlayersizes = 30,30,30

#Import the data
data = pd.read_csv(inputdataset)
#get the independent variable (X) by dropping the dependent variable
x=data.drop('SalePrice',axis = 1)

#set the dependent variable which is saleprice
y=data['SalePrice']

#split the data into 70:30 (train:test)
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=.30)

#multi layer perceptron
mlp = MLPRegressor(hidden_layer_sizes=(hiddenlayersizes))

#train the model
mlp.fit(x_train,y_train)

# test set predictions
y_pred = mlp.predict(x_test)

# Metrics
#Coefficient
print('Coefficients: \n', mlp.coefs_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))
