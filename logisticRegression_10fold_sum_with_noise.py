import pandas as p
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from dataProcessing_sum import dataProcessing_sum_noise
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
import numpy as np
from sklearn.model_selection import KFold
from config_task1 import n_splits
from config_task1 import chunk_start_size
from config_task1 import dataset2

#n_splits = 10
chunk_split_start_loop_size = chunk_start_size

if __name__=="__main__":
    #import dataset
    dataset = p.read_csv(dataset2,delimiter=";")
    #print(houseData.head())
    dataset=dataset.drop(['Noisy Target'],axis = 1)

    #all the variables except SalePrice is taken as X variables
    data=dataProcessing_sum_noise(dataset)    #dataprocessing
    #chunk_split_start_loop_size = 100
    flag=1
    while chunk_split_start_loop_size <= data.shape[0]:
        x=data.head(chunk_split_start_loop_size)
        # Saleprice is assined as target variable
        y=x['Noisy Target Class']
        x=x.drop(['Noisy Target Class'],axis=1)
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
