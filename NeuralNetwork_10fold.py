import pandas as p
import sklearn
from sklearn.model_selection import train_test_split
from DataProcessingTask1 import dataProcessing
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
<<<<<<< HEAD
from sklearn.metrics import f1_score
import numpy as np
from sklearn.model_selection import KFold

chunk_size = 100
n_splits = 10
#number of hidden layers in th perceptron
hiddenlayersizes = 30,30,30
=======
from sklearn.metrics import cohen_kappa_score
import numpy as np
from sklearn.model_selection import KFold

n_splits = 10
#number of hidden layers in th perceptron
hiddenlayersizes = 50,50,50
>>>>>>> f9e214768ce2796f3be2d43238fb6fd145c38ea9
max_iter = 500

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

        # Splitting the dataset into 10 folds
        kf = KFold(n_splits=n_splits)
        kf.get_n_splits(x)
        accuracy=0
<<<<<<< HEAD
=======
        cohenkappascore=0
>>>>>>> f9e214768ce2796f3be2d43238fb6fd145c38ea9
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
<<<<<<< HEAD

        accuracy = accuracy/n_splits
        print("Accuracy for chunk size ",chunk_size,":",accuracy*100,"%")
        #print("F1 score: ",f1_score(y_test,y_pred))
        #print("Classification Report: \n",classification_report(y_test,y_pred))

        if flag == 1:
            chunk_size=chunk_size*5
            flag = 0
        else:
            chunk_size=chunk_size*2
=======
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
>>>>>>> f9e214768ce2796f3be2d43238fb6fd145c38ea9
            flag = 1
