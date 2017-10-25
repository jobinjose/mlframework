import sklearn
import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from dataProcessing import dataProcessing

#Import the data
data = pd.read_csv('housing dataset.csv')
#get the x by droping the dependent variable
x=data.drop(['Alley','PoolQC','MiscFeature','Fence','FireplaceQu','HouseStyle'],axis = 1)
x=dataProcessing(x)    #dataprocessing

#set the dependent variable which is saleprice
y=x['SalePrice']
x=x.drop(['SalePrice'],axis = 1)

x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=.30)

#linear regression object creation
linear_reg = linear_model.LinearRegression()

#train the model
linear_reg.fit(x_train,y_train)

# test set predictions
y_pred = linear_reg.predict(x_test)
#e=y_pred - y_test
#xval_err =np.dot(e,e)
#rmse_10cv = np.sqrt(xval_err/len(x))
#print("RMSE: %.2f" %rmse_10cv)

# Metrics
RMSE = np.sqrt(mean_squared_error(y_test,y_pred))
R2 = r2_score(y_test,y_pred)
#accuracy = accuracy_score(y_test,y_result)
print("RMSE : ", RMSE)
print("\nR2 : ",R2)
#Coefficient
#print('Coefficients: \n', linear_reg.coef_)
# The mean squared error
#print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
#print('Variance score: %.2f' % r2_score(y_test, y_pred))
