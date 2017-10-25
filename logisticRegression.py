import pandas as p
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from DataProcessingTask1 import dataProcessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
<<<<<<< HEAD
from sklearn.metrics import recall_score
import numpy as np

chunk_size = 100
=======
from sklearn.metrics import cohen_kappa_score
import numpy as np


>>>>>>> f9e214768ce2796f3be2d43238fb6fd145c38ea9
testsize = 0.30

if __name__=="__main__":
    #import dataset
    houseData = p.read_csv("housing dataset.csv")
    #print(houseData.head())

    #all the variables except SalePrice is taken as X variables
    data=houseData.drop(['Alley','PoolQC','MiscFeature','Fence','FireplaceQu'],axis=1)
    data=dataProcessing(data)    #dataprocessing
<<<<<<< HEAD
    flag=1
    while chunk_size <= data.shape[0]:
        x=data.head(chunk_size)
=======
    chunk_split_start_loop_size = 100
    flag=1
    while chunk_split_start_loop_size <= data.shape[0]:
        x=data.head(chunk_split_start_loop_size)
>>>>>>> f9e214768ce2796f3be2d43238fb6fd145c38ea9
        # Saleprice is assined as target variable
        y=x['SaleCondition']
        x=x.drop(['SaleCondition'],axis=1)
        x = p.get_dummies(x)
        # Splitting the dataset into training set(70%) and test set (30%)
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=testsize)

        classifier = LogisticRegression()
        classifier.fit(x_train,y_train)
        y_pred = classifier.predict(x_test)
        #print(y_pred)
        #conf_matrix = confusion_matrix(y_test,y_pred)
        #print(classifier.score(x_test,y_test))
        #print(conf_matrix)
        accuracy = accuracy_score(y_test,y_pred)
<<<<<<< HEAD
        print("Accuracy for chunk size ",chunk_size,":",accuracy*100,"%")
        #print("Recall score: ",recall_score(y_test,y_pred))
        print("Classification Report: \n", classification_report(y_test,y_pred))

        if flag == 1:
            chunk_size=chunk_size*5
            flag = 0
        else:
            chunk_size=chunk_size*2
=======
        print("Accuracy for chunk size ",chunk_split_start_loop_size,":",accuracy*100,"%")
        print("Cohen kappa score for chunk size ",chunk_split_start_loop_size,": ",cohen_kappa_score(y_test,y_pred))
        #print("Classification Report: \n", classification_report(y_test,y_pred))

        if flag == 1:
            chunk_split_start_loop_size=chunk_split_start_loop_size*5
            flag = 0
        else:
            chunk_split_start_loop_size=chunk_split_start_loop_size*2
>>>>>>> f9e214768ce2796f3be2d43238fb6fd145c38ea9
            flag = 1
