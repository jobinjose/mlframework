import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello).decode())

import pandas as pd

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
#test git
#test atom
<<<<<<< HEAD
#test atom once more
=======
>>>>>>> c6406de1209b1ebf108d7a5c34e9e83781ae1214
