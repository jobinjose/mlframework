import pandas as p
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from dataProcessing_kc_data import dataProcessing_kc_data
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
import numpy as np
from config import dataset1,no_of_trees,testsize

#no_of_trees = 10
if __name__=="__main__":
    #import dataset
    houseData = p.read_csv(dataset1)
    #print(houseData.head())

    #all the variables except SalePrice is taken as X variables
    #x=houseData.drop(['Alley','PoolQC','MiscFeature','Fence','FireplaceQu','HouseStyle'],axis=1)
    x=dataProcessing_kc_data(houseData)    #dataprocessing
    # Saleprice is assined as target variable
    y=x['price']
    x=x.drop(['price'],axis=1)
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

    #Calculating RMSE
    RMSE = np.sqrt(mean_squared_error(y_test,y_result))
    R2 = r2_score(y_test,y_result)
    #accuracy = accuracy_score(y_test,y_result)
    print("RMSE : ", RMSE)
    print("\nR2 : ",R2)
    #print(accuracy)
