import pandas as p
import sklearn
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from dataProcessing_NYC import dataProcessing_NYC
from sklearn.metrics import mean_squared_error
#from sklearn.metrics import accuracy_score
import numpy as n
from sklearn.metrics import r2_score

n_splits = 10
no_of_trees = 10
if __name__=="__main__":
    #import dataset
    dataset = p.read_csv("New York City Taxi Trip Duration.csv")

    data=dataProcessing_NYC(dataset)    #dataprocessing
    data=data.drop(['dropoff_datetime','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude'],axis = 1)
    chunk_split_start_loop_size = 100
    flag=1
    while chunk_split_start_loop_size <= data.shape[0]:
        x=data.head(chunk_split_start_loop_size)
        # Saleprice is assined as target variable
        y=x['trip_duration']
        x=x.drop(['trip_duration'],axis=1)
        # Splitting the dataset into 10 folds
        kf = KFold(n_splits=n_splits)
        kf.get_n_splits(x)
        xval_err = 0
        R2 = []
        RMSE = []

        i=0
        # Random Forest Regression
        for train,test in kf.split(x):
            i+=1
            #print("Iteration No : ", i)
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
        print("RMSE on 10 fold for chunk size ",chunk_split_start_loop_size," : " , sum(RMSE)/n_splits)
        print("R2 on 10 fold for chunk size ",chunk_split_start_loop_size," : " , sum(R2)/n_splits)
        #print(accuracy)

        if flag == 1:
            chunk_split_start_loop_size=chunk_split_start_loop_size*5
            flag = 0
        else:
            chunk_split_start_loop_size=chunk_split_start_loop_size*2
            flag = 1
