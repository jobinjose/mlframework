import pandas as p
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from dataProcessing_NYC import dataProcessing_NYC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
import numpy as np
from sklearn.model_selection import KFold

n_splits = 10

if __name__=="__main__":
    #import dataset
    dataset = p.read_csv("New York City Taxi Trip Duration.csv")
    #print(houseData.head())

    #all the variables except SalePrice is taken as X variables

    data=dataProcessing_NYC(dataset)    #dataprocessing
    data=data.drop(['dropoff_datetime','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude'],axis = 1)
    chunk_split_start_loop_size = 100
    flag=1
    while chunk_split_start_loop_size <= data.shape[0]:
        x=data.head(chunk_split_start_loop_size)
        # Saleprice is assined as target variable
        y=x['vendor_id']
        x=x.drop(['vendor_id'],axis=1)
        #x = p.get_dummies(x)

        # Splitting the dataset into 10 folds
        kf = KFold(n_splits=n_splits)
        kf.get_n_splits(x)
        accuracy=0
        cohenkappascore=0
        i=0
        for train,test in kf.split(x):
            i+=1
            #print("Iteration No : ", i)
            classifier = LogisticRegression()
            classifier.fit(x.iloc[train],y.iloc[train])
            y_pred = classifier.predict(x.iloc[test])
            #print(y_pred)
            #conf_matrix = confusion_matrix(y_test,y_pred)
            #print(classifier.score(x_test,y_test))
            #print(conf_matrix)
            accuracy += accuracy_score(y.iloc[test],y_pred)
            cohenkappascore += cohen_kappa_score(y.iloc[test],y_pred)

        accuracy = accuracy/n_splits
        cohenkappascore = cohenkappascore/n_splits
        print("Accuracy for chunk size ",chunk_split_start_loop_size,":",accuracy*100,"%")
        print("Cohen kappa score for chunk size ",chunk_split_start_loop_size,": ",cohenkappascore)
        #print("Classification Report: \n", classification_report(y_test,y_pred))

        if flag == 1:
            chunk_split_start_loop_size=chunk_split_start_loop_size*5
            flag = 0
        else:
            chunk_split_start_loop_size=chunk_split_start_loop_size*2
            flag = 1
