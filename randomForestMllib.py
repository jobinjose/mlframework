import pandas as p
import numpy as n
import mllib
#from sklearn.model_selection import train_test_split
from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from dataProcessing import dataProcessing
from pyspark.sql import SQLContext
from pyspark.context import SparkContext
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

if __name__=="__main__":
	#import dataset
    houseData = p.read_csv("housing dataset.csv")

    #all the variables except SalePrice is taken as X variables
    x=houseData.drop(['Alley','PoolQC','MiscFeature','Fence','FireplaceQu','HouseStyle'],axis=1)
    x=dataProcessing(x)
    sc=SparkContext('local','randomForestMllib')
    sqlcontext=SQLContext(sc)
    x_new=sqlcontext.createDataFrame(data=x)
    # Saleprice is assined as target variable
    #y=x['SalePrice'] 
    #x=x.drop(['SalePrice'],axis=1)

    # Splitting the dataset into training set(70%) and test set (30%)
    #x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30)
    (train_data,test_data) = x_new.randomSplit([0.7,0.3])
    label='SalePrice'
    features=list(filter(lambda w: w not in label, train_data.columns))
    #print(features)
    assembler = VectorAssembler(inputCols=features,outputCol="features")
    train_data_transformed = assembler.transform(train_data)

    regressor = RandomForestRegressor(featuresCol="features", labelCol="SalePrice", predictionCol="prediction",numTrees=100,maxDepth=10)
    regression_model = regressor.fit(train_data_transformed)

    test_data_transformed = assembler.transform(test_data)
    prediction = regression_model.transform(test_data_transformed)
    #print(prediction.head().prediction)
    
    evaluator=RegressionEvaluator()

    evaluator = RegressionEvaluator(labelCol="SalePrice", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(prediction)
    print(rmse)
    
    


