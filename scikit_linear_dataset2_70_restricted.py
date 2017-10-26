import sklearn
import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from dataProcessing_NYC import dataProcessing_NYC

#Import the data
data = pd.read_csv('New York City Taxi Trip Duration.csv',nrows = 100000)
x=dataProcessing_NYC(data)    #dataprocessing

#set the dependent variable which is saleprice
y=x['trip_duration']
x=x.drop(['dropoff_datetime','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','trip_duration'],axis = 1)

x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=.30)

#linear regression object creation
linear_reg = linear_model.LinearRegression()

#train the model
linear_reg.fit(x_train,y_train)

# test set predictions
y_pred = linear_reg.predict(x_test)

# Metrics
RMSE = np.sqrt(mean_squared_error(y_test,y_pred))
R2 = r2_score(y_test,y_pred)
print("RMSE : ", RMSE)
print("\nR2 : ",R2)
