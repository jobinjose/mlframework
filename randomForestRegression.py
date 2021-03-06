import pandas as p
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from dataProcessing import dataProcessing
from sklearn.metrics import mean_squared_error
#from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import r2_score

testsize = 0.30
no_of_trees = 10

if __name__=="__main__":
    #import dataset
    houseData = p.read_csv("housing dataset.csv")
    #print(houseData.head())

    #all the variables except SalePrice is taken as X variables
    data=houseData.drop(['Alley','PoolQC','MiscFeature','Fence','FireplaceQu','HouseStyle'],axis=1)
    data=dataProcessing(data)    #dataprocessing
    chunk_split_start_loop_size = 100
    flag=1
    while chunk_split_start_loop_size <= data.shape[0]:
        x=data.head(chunk_split_start_loop_size)
        # Saleprice is assined as target variable
        y=x['SalePrice']
        x=x.drop(['SalePrice'],axis=1)
        #print(list(x))
        #print(y)

        # Splitting the dataset into training set(70%) and test set (30%)
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=testsize)
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
        RMSE = np.sqrt(mean_squared_error(y_test,y_result))
        R2 = r2_score(y_test,y_result)
        print("RMSE for chunk size ", chunk_split_start_loop_size,": ", RMSE)
        print("R2 for chunk size ", chunk_split_start_loop_size,": ", R2)
        #y_test['result'] = y_result
        #print(y_result)

        #print(y_test)

        #Calculating RMSE
        #RMSE = mean_squared_error(y_test,y_result)
        #accuracy = accuracy_score(y_test,y_result)
        #print(RMSE)
        #print(accuracy)

        if flag == 1:
            chunk_split_start_loop_size=chunk_split_start_loop_size*5
            flag = 0
        else:
            chunk_split_start_loop_size=chunk_split_start_loop_size*2
            flag = 1
