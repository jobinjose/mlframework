from pyspark.ml.regression import LinearRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from pyspark.sql import SQLContext
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from dataProcessing_NYC import dataProcessing_NYC
from pyspark.ml.evaluation import RegressionEvaluator
import mllib
from pyspark.ml.tuning import ParamGridBuilder,CrossValidator
from config import dataset2, no_of_rows,no_of_folds,maxIter

from pyspark import SparkContext, SparkConf

#import dataset
NYCData = pd.read_csv(dataset2,nrows = no_of_rows)

#all the variables except trip_duration is taken as X variables
x=dataProcessing_NYC(NYCData)
x=x.drop(['dropoff_datetime','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude'],axis=1)

sc=SparkContext('local','LinearRegressionMllib')
sqlcontext=SQLContext(sc)
x_new=sqlcontext.createDataFrame(data=x)

label='trip_duration'
features=list(filter(lambda w: w not in label, x_new.columns))
#print(features)
assembler = VectorAssembler(inputCols=features,outputCol="features")
data_transformed = assembler.transform(x_new)

linearRegressor = LinearRegression(labelCol="trip_duration", featuresCol="features", maxIter=maxIter)
evaluator = RegressionEvaluator(predictionCol='prediction', labelCol='trip_duration')

paramGrid = ParamGridBuilder().addGrid(linearRegressor.regParam, [0.1, 0.01]).addGrid(linearRegressor.elasticNetParam, [0, 1]).build()
crossval = CrossValidator(estimator=linearRegressor, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=no_of_folds)

crossValModel = crossval.fit(data_transformed)
cvSummary = crossValModel.bestModel.summary

print("RMSE : ", cvSummary.rootMeanSquaredError)
print("\nr2 : ", cvSummary.r2)
