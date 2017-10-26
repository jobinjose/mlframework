import pandas as p
import sklearn
from sklearn.model_selection import train_test_split
from dataProcessing_sum import dataProcessing_sum_without_noise
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
import numpy as np
from sklearn.model_selection import KFold

n_splits = 10
#number of hidden layers in th perceptron
hiddenlayersizes = 20,20,20
max_iter = 500

if __name__=="__main__":
    #import dataset
    dataset = p.read_csv("The SUM dataset, without noise.csv",delimiter=";")
    dataset=dataset.drop(['Target'],axis = 1)
    #print(houseData.head())

    #all the variables except SalePrice is taken as X variables
    data=dataProcessing_sum_without_noise(dataset)    #dataprocessing
    chunk_split_start_loop_size = 100
    flag=1
    while chunk_split_start_loop_size <= data.shape[0]:
        x=data.head(chunk_split_start_loop_size)
        # Saleprice is assined as target variable
        y=x['Target Class']
        x=x.drop(['Target Class'],axis=1)
        #x = p.get_dummies(x)

        # Splitting the dataset into 10 folds
        kf = KFold(n_splits=n_splits)
        kf.get_n_splits(x)
        accuracy=0
        cohenkappascore=0
        i=0
        for train,test in kf.split(x):
            i+=1
            #print("Iteration No: ",i)
            mlp_classifier = MLPClassifier(hidden_layer_sizes=(hiddenlayersizes),max_iter=max_iter)
            mlp_classifier.fit(x.iloc[train],y.iloc[train])
            #predictions
            y_pred = mlp_classifier.predict(x.iloc[test])
            #print(y_pred)
            #print(confusion_matrix(y_test,y_pred))
            accuracy += accuracy_score(y.iloc[test],y_pred)
            cohenkappascore += cohen_kappa_score(y.iloc[test],y_pred)

        accuracy = accuracy/n_splits
        cohenkappascore = cohenkappascore/n_splits
        print("Accuracy for chunk size ",chunk_split_start_loop_size,":",accuracy*100,"%")
        print("Cohen kappa score for chunk size ",chunk_split_start_loop_size,": ",cohenkappascore)
        #print("Classification Report: \n",classification_report(y_test,y_pred))

        if flag == 1:
            chunk_split_start_loop_size=chunk_split_start_loop_size*5
            flag = 0
        else:
            chunk_split_start_loop_size=chunk_split_start_loop_size*2
            flag = 1
