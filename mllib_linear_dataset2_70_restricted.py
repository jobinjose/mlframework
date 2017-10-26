from pyspark.ml.regression import LinearRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from pyspark.sql import SQLContext
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from dataProcessing_NYC import dataProcessing_NYC
from pyspark.ml.evaluation import RegressionEvaluator
import mllib

from pyspark import SparkContext, SparkConf

#import dataset
NYCData = pd.read_csv("New York City Taxi Trip Duration.csv",nrows = 100000)

#all the variables except trip_duration is taken as X variables
x=dataProcessing_NYC(NYCData)
x=x.drop(['dropoff_datetime','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude'],axis=1)

sc=SparkContext('local','LinearRegressionMllib')
sqlcontext=SQLContext(sc)
x_new=sqlcontext.createDataFrame(data=x)

(train_data,test_data) = x_new.randomSplit([0.7,0.3])
label='trip_duration'
features=list(filter(lambda w: w not in label, train_data.columns))
#print(features)
assembler = VectorAssembler(inputCols=features,outputCol="features")
train_data_transformed = assembler.transform(train_data)

linearRegressor = LinearRegression(labelCol="trip_duration", featuresCol="features", maxIter=10)
linearModel = linearRegressor.fit(train_data_transformed)

test_data_transformed = assembler.transform(test_data)
prediction = linearModel.transform(test_data_transformed)
evaluator = RegressionEvaluator(predictionCol='prediction', labelCol='trip_duration')
rmse = evaluator.evaluate(prediction, {evaluator.metricName: "rmse"})
r2 = evaluator.evaluate(prediction, {evaluator.metricName: "r2"})
print("RMSE : ", rmse)
print("R2 : ", r2)
