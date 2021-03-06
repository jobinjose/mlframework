import pandas as p
import numpy as n
import mllib
from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from dataProcessing_NYC import dataProcessing_NYC
from pyspark.sql import SQLContext
from pyspark.context import SparkContext
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

no_of_trees = 10
if __name__=="__main__":
	#import dataset
    NYCData = p.read_csv("New York City Taxi Trip Duration.csv")

    #all the variables except trip_duration is taken as X variables
    x=dataProcessing_NYC(NYCData)
    x=x.drop(['dropoff_datetime','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude'],axis=1)
    sc=SparkContext('local','randomForestMllib')
    sqlcontext=SQLContext(sc)
    x_new=sqlcontext.createDataFrame(data=x)

    # Splitting the dataset into training set(70%) and test set (30%)
    #x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30)
    (train_data,test_data) = x_new.randomSplit([0.7,0.3])
    label='trip_duration'
    features=list(filter(lambda w: w not in label, train_data.columns))
    #print(features)
    assembler = VectorAssembler(inputCols=features,outputCol="features")
    train_data_transformed = assembler.transform(train_data)

    regressor = RandomForestRegressor(featuresCol="features", labelCol="trip_duration", predictionCol="prediction",numTrees=no_of_trees)
    regression_model = regressor.fit(train_data_transformed)

    test_data_transformed = assembler.transform(test_data)
    prediction = regression_model.transform(test_data_transformed)

    evaluator = RegressionEvaluator(predictionCol='prediction', labelCol='trip_duration')
    rmse = evaluator.evaluate(prediction, {evaluator.metricName: "rmse"})
    r2 = evaluator.evaluate(prediction, {evaluator.metricName: "r2"})
    print("RMSE : ", rmse)
    print("R2 : ", r2)
