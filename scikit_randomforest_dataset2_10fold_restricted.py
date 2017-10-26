import pandas as p
import sklearn
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from dataProcessing_NYC import dataProcessing_NYC
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
import numpy as n

no_of_trees = 10
if __name__=="__main__":
    #import dataset
    data = p.read_csv("New York City Taxi Trip Duration.csv",nrows = 100000)

    x=dataProcessing_NYC(data)    #dataprocessing
    # Saleprice is assined as target variable
    y=x['trip_duration']
    x=x.drop(['dropoff_datetime','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','trip_duration'],axis=1)
    no_of_folds = 10

    # Splitting the dataset into 10 folds
    kf = KFold(n_splits=no_of_folds)
    kf.get_n_splits(x)
    xval_err = 0

    i=0
    RMSE = []
    R2 = []
    # Random Forest Regression
    for train,test in kf.split(x):
        i+=1
        print("Iteration No : ", i)
        RFRegressor = RandomForestRegressor(n_estimators = no_of_trees)
        RFRegressor.fit(x.iloc[train],y.iloc[train])
        # testing
        y_result = RFRegressor.predict(x.iloc[test])
        #e=y_result - y.iloc[test]
        #xval_err +=n.dot(e,e)
        RMSE.append(n.sqrt(mean_squared_error(y.iloc[test],y_result)))
        R2.append(r2_score(y.iloc[test],y_result))

    #Calculating RMSE
    #RMSE = n.sqrt(xval_err/len(x))
    #RMSE = mean_squared_error(y_test,y_result)
    #accuracy = accuracy_score(y_test,y_result)
    print("RMSE : " , sum(RMSE)/no_of_folds)
    print("\nR2 : ", sum(R2)/no_of_folds)
    #print(accuracy)
