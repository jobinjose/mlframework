import pandas as p
import sklearn
from sklearn.model_selection import train_test_split
from DataProcessingTask1 import dataProcessing
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import numpy as np

chunk_size = 100
testsize = 0.30
#number of hidden layers in th perceptron
hiddenlayersizes = 30,30,30
max_iter = 500

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
        # Splitting the dataset into training set(70%) and test set (30%)
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=testsize)

        mlp_classifier = MLPClassifier(hidden_layer_sizes=(hiddenlayersizes),max_iter=max_iter)
        mlp_classifier.fit(x_train,y_train)
        #predictions
        y_pred = mlp_classifier.predict(x_test)
        #print(y_pred)
        #print(confusion_matrix(y_test,y_pred))
        accuracy = accuracy_score(y_test,y_pred)
        print("Accuracy for chunk size ",chunk_size,":",accuracy*100,"%")
        #print("F1 score: ",f1_score(y_test,y_pred))
        print("Classification Report: \n",classification_report(y_test,y_pred))

        if flag == 1:
            chunk_size=chunk_size*5
            flag = 0
        else:
            chunk_size=chunk_size*2
            flag = 1