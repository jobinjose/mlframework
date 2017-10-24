import pandas as p
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from DataProcessingTask1 import dataProcessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
import numpy as np
from sklearn.model_selection import KFold

chunk_size = 100
n_splits = 10

if __name__=="__main__":
    #import dataset
    houseData = p.read_csv("housing dataset.csv")
    #print(houseData.head())

    #all the variables except SalePrice is taken as X variables
    data=houseData.drop(['Alley','PoolQC','MiscFeature','Fence','FireplaceQu'],axis=1)
    data=dataProcessing(data)    #dataprocessing
    flag=1
    while chunk_size <= data.shape[0]:
        x=data.head(chunk_size)
        # Saleprice is assined as target variable
        y=x['SaleCondition']
        x=x.drop(['SaleCondition'],axis=1)
        x = p.get_dummies(x)

        # Splitting the dataset into 10 folds
        kf = KFold(n_splits=n_splits)
        kf.get_n_splits(x)
        accuracy=0
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

        accuracy = accuracy/n_splits
        print("Accuracy for chunk size ",chunk_size,":",accuracy*100,"%")
        #print("Recall score: ",recall_score(y_test,y_pred))
        #print("Classification Report: \n", classification_report(y_test,y_pred))

        if flag == 1:
            chunk_size=chunk_size*5
            flag = 0
        else:
            chunk_size=chunk_size*2
            flag = 1
