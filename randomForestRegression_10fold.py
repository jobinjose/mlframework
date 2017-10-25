import pandas as p
import sklearn
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from dataProcessing import dataProcessing
from sklearn.metrics import mean_squared_error
#from sklearn.metrics import accuracy_score
import numpy as n
from sklearn.metrics import r2_score

n_splits = 10
no_of_trees = 5000
if __name__=="__main__":
    #import dataset
    houseData = p.read_csv("housing dataset.csv")

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
        # Splitting the dataset into 10 folds
        kf = KFold(n_splits=n_splits)
        kf.get_n_splits(x)
        xval_err = 0
        R2 = []

        i=0
        # Random Forest Regression
        for train,test in kf.split(x):
            i+=1
            #print("Iteration No : ", i)
            RFRegressor = RandomForestRegressor(n_estimators = no_of_trees)
            RFRegressor.fit(x.iloc[train],y.iloc[train])
            # testing
            y_result = RFRegressor.predict(x.iloc[test])
            e=y_result - y.iloc[test]
            xval_err +=n.dot(e,e)
            R2.append(r2_score(y.iloc[test],y_result))

        #Calculating RMSE
        RMSE = n.sqrt(xval_err/len(x))
        #RMSE = mean_squared_error(y_test,y_result)
        #accuracy = accuracy_score(y_test,y_result)
        print("RMSE on 10 fold for chunk size ",chunk_split_start_loop_size," : " , RMSE)
        print("R2 on 10 fold for chunk size ",chunk_split_start_loop_size," : " , sum(R2)/n_splits)
        #print(accuracy)

        if flag == 1:
            chunk_split_start_loop_size=chunk_split_start_loop_size*5
            flag = 0
        else:
            chunk_split_start_loop_size=chunk_split_start_loop_size*2
            flag = 1
