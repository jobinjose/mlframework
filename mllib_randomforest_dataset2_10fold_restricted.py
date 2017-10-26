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
from pyspark.ml.tuning import ParamGridBuilder,CrossValidator

no_of_trees = 10
if __name__=="__main__":
	#import dataset
    NYCData = p.read_csv("New York City Taxi Trip Duration.csv",nrows = 100000)

    #all the variables except trip_duration is taken as X variables
    x=dataProcessing_NYC(NYCData)
    x=x.drop(['dropoff_datetime','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude'],axis=1)
    sc=SparkContext('local','randomForestMllib')
    sqlcontext=SQLContext(sc)
    x_new=sqlcontext.createDataFrame(data=x)

    label='trip_duration'
    features=list(filter(lambda w: w not in label, x_new.columns))
    #print(features)
    assembler = VectorAssembler(inputCols=features,outputCol="features")
    data_transformed = assembler.transform(x_new)

    regressor = RandomForestRegressor(featuresCol="features", labelCol="trip_duration", predictionCol="prediction",numTrees=no_of_trees)
    evaluator_rmse = RegressionEvaluator(labelCol="trip_duration", predictionCol="prediction",metricName="rmse")
    evaluator_r2 = RegressionEvaluator(labelCol="trip_duration", predictionCol="prediction",metricName="r2")

    paramGrid = ParamGridBuilder().build()
    crossval = CrossValidator(estimator=regressor, estimatorParamMaps=paramGrid, evaluator=evaluator_rmse, numFolds=10)
    crossval_r2 = CrossValidator(estimator=regressor, estimatorParamMaps=paramGrid, evaluator=evaluator_r2, numFolds=10)

    crossValModel = crossval.fit(data_transformed)
    #cvSummary = crossValModel.getParam()
    #cvSummary = crossValModel.bestModel.summary
    RMSE = crossValModel.avgMetrics[0]
    #cvSummary1 = crossValModel.avgMetrics[1]
    crossValModel = crossval_r2.fit(data_transformed)
    R2 = crossValModel.avgMetrics[0]
    #print(cvSummary1)


    #rmse = evaluator.evaluate(prediction, {evaluator.metricName: "rmse"})
    print("RMSE : ", RMSE)
    print("\nr2 : ", R2)
