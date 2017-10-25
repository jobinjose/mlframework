import pandas as p
import numpy as n
import mllib
from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from dataProcessing import dataProcessing
from pyspark.sql import SQLContext
from pyspark.context import SparkContext
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import ParamGridBuilder,CrossValidator

no_of_trees = 10
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
    #(train_data,test_data) = x_new.randomSplit([0.7,0.3])
    label='SalePrice'
    features=list(filter(lambda w: w not in label, x_new.columns))
    #print(features)
    assembler = VectorAssembler(inputCols=features,outputCol="features")
    data_transformed = assembler.transform(x_new)

    regressor = RandomForestRegressor(featuresCol="features", labelCol="SalePrice", predictionCol="prediction",numTrees=no_of_trees)
    evaluator_rmse = RegressionEvaluator(labelCol="SalePrice", predictionCol="prediction",metricName="rmse")
    evaluator_r2 = RegressionEvaluator(labelCol="SalePrice", predictionCol="prediction",metricName="r2")

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


    #test_data_transformed = assembler.transform(test_data)
    #prediction = crossValModel.transform(test_data_transformed)
    #print(prediction.head().prediction)


    #rmse = evaluator.evaluate(prediction, {evaluator.metricName: "rmse"})
    print("RMSE : ", RMSE)
    print("\nr2 : ", R2)
