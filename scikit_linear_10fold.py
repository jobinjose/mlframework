import sklearn
import csv
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn import datasets, linear_model
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from dataProcessing import dataProcessing

#Import the data
data = pd.read_csv('housing dataset.csv')
#get the x by droping the dependent variable
x=data.drop(['Alley','PoolQC','MiscFeature','Fence','FireplaceQu','HouseStyle','SalePrice'],axis = 1)
x=dataProcessing(x)    #dataprocessing

#set the dependent variable which is saleprice
y=data['SalePrice']

#linear regression object creation
no_of_folds = 10
kf = KFold(n_splits=no_of_folds)
kf.get_n_splits(x)
xval_err = 0
RMSE = []
R2 = []
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
print("RMSE : " , sum(RMSE)/no_of_folds)
print("\nR2 : ", sum(R2)/no_of_folds)
#Coefficient
#print('Coefficients: \n', linear_reg.coef_)
# The mean squared error
#print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
#print('Variance score: %.2f' % r2_score(y_test, y_pred))

#print("RMSE on 10 fold: %.2f" %rmse_10cv)
