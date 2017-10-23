import pandas as p
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from dataProcessing import dataProcessing
#import matplotlib.pyplot as plot
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

if __name__=="__main__":
    #import dataset
    houseData = p.read_csv("housing dataset.csv")
    #print(houseData.head())

    #all the variables except SalePrice is taken as X variables
    x=houseData.drop(['Alley','PoolQC','MiscFeature','Fence','FireplaceQu'],axis=1)
    x=dataProcessing(x)    #dataprocessing
    # Saleprice is assined as target variable
    y=x['SaleCondition']
    x=x.drop(['SaleCondition'],axis=1)
    x = p.get_dummies(x)
    # Splitting the dataset into training set(70%) and test set (30%)
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30)

    classifier = LogisticRegression()
    classifier.fit(x_train,y_train)
    y_pred = classifier.predict(x_test)
    #print(y_pred)
    conf_matrix = confusion_matrix(y_test,y_pred)
    print(classifier.score(x_test,y_test))
    print(conf_matrix)
    print(classification_report(y_test,y_pred))
    
    '''
    result = result.reshape(x_test)
    plot.figure(1,figsize=(4,3))
    plot.pcolormesh(x_test,y_test,result,cmap=plot.cm.Paired)


    plot.xticks(())
    plot.yticks(())
    plot.show()
    '''

