import pandas as p
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from dataProcessing_NYC import dataProcessing_NYC
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
import numpy as np

no_of_trees = 10
if __name__=="__main__":
    #import dataset
    data = p.read_csv("New York City Taxi Trip Duration.csv")
    #print(houseData.head())
    x=dataProcessing_NYC(data)    #dataprocessing
    # Saleprice is assined as target variable
    y=x['trip_duration']
    x=x.drop(['dropoff_datetime','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','trip_duration'],axis=1)
    #print(list(x))
    #print(y)

    # Splitting the dataset into training set(70%) and test set (30%)
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30)
    #print(x_train) #1022 rows
    #print(x_test) #438 rows

    # Random Forest Regression
    RFRegressor = RandomForestRegressor(n_estimators = no_of_trees)
    RFRegressor.fit(x_train,y_train)


    # testing
    y_result = RFRegressor.predict(x_test)

    #RMSE Calsulation
    #e=y_result - y_test
    #xval_err =np.dot(e,e)
    #rmse_10cv = np.sqrt(xval_err/len(x))
    #print("RMSE: %.2f" %rmse_10cv)
    #y_test['result'] = y_result
    #print(y_result)

    #print(y_test)

    #Calculating RMSE
    RMSE = np.sqrt(mean_squared_error(y_test,y_result))
    R2 = r2_score(y_test,y_result)
    #accuracy = accuracy_score(y_test,y_result)
    print("RMSE : ", RMSE)
    print("\nR2 : ",R2)
    #print(accuracy)
