import sklearn
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import datasets, linear_model
#with open('U:/Machine learning/Assignment/Task2/House Prices/housing dataset.csv', newline='') as csvfile:
#	spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
#	for row in spamreader:
#		print(', '.join(row))
#Import the data
data = pd.read_csv('test_dataset.csv')
#get the x by droping the dependent variable
x=data.drop('SalePrice',axis = 1)
#x=x.drop('SaleCondition',axis =1)
#print(x)
#print(data.head())
#set the dependent variable which is saleprice
y=data['SalePrice']
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=.30)

#linear regression object creation
linear_reg = linear_model.LinearRegression()

#train the model
linear_reg.fit(x_train,y_train)

# test set predictions
y_pred = linear_reg.predict(x_test)


# Metrics
#Coefficient
print('Coefficients: \n', linear_reg.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))
