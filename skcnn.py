import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


#Import the data
data = pd.read_csv('housing dataset.csv')
#print(data)
#get the x by droping the dependent variable
x=data.drop('SalePrice',axis = 1)
#print(x)
#print(data.head())
#set the dependent variable which is saleprice
y=data['SalePrice']
#print(y)
x_train, x_test, y_train, y_test = train_test_split(x, y)
model = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500)
model.fit(x_train,y_train)
