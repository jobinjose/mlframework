import pandas as p
import sklearn
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from dataProcessing import dataProcessing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
import numpy as n

if __name__=="__main__":
    #import dataset
    houseData = p.read_csv("housing dataset.csv")

    #all the variables except SalePrice is taken as X variables
    x=houseData.drop(['Alley','PoolQC','MiscFeature','Fence','FireplaceQu','HouseStyle'],axis=1)
    x=dataProcessing(x)    #dataprocessing
    # Saleprice is assined as target variable
    y=x['SalePrice']
    x=x.drop(['SalePrice'],axis=1)

    # Splitting the dataset into 10 folds
    kf = KFold(n_splits=10)
    kf.get_n_splits(x)
    xval_err = 0
    # Random Forest Regression
    for train,test in kf.split(x):
        RFRegressor = RandomForestRegressor(n_estimators = 5000)
        RFRegressor.fit(x[train],y[train])
        # testing
        y_result = RFRegressor.predict(x[test])
        e=y_result - y[test]
        xval_err +=n.dot(e,e)

    #Calculating RMSE
    RMSE = n.sqrt(xval_err/len(x))
    #RMSE = mean_squared_error(y_test,y_result)
    #accuracy = accuracy_score(y_test,y_result)
    print("RMSE : " , RMSE)
    #print(accuracy)
