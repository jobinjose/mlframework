from pyspark.ml.regression import LinearRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from pyspark.sql import SQLContext
#from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD, LinearRegressionModel
#from pyspark.ml.linalg import Vectors
#from pyspark.mllib.util import MLUtils
#from pyspark.mllib.linalg import Vectors
#from pyspark.mllib.feature import StandardScaler
#from pyspark.mllib.evaluation import RegressionMetrics
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from dataProcessing import dataProcessing
from pyspark.ml.evaluation import RegressionEvaluator
import mllib
from pyspark.ml.tuning import ParamGridBuilder

from pyspark import SparkContext, SparkConf

#import dataset
houseData = pd.read_csv("housing dataset.csv")

#all the variables except SalePrice is taken as X variables
x=houseData.drop(['Alley','PoolQC','MiscFeature','Fence','FireplaceQu','HouseStyle'],axis=1)
x=dataProcessing(x)
sc=SparkContext('local','LinearRegressionMllib')
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

linearRegressor = LinearRegression(labelCol="SalePrice", featuresCol="features", maxIter=10)
evaluator = RegressionEvaluator(predictionCol='prediction', labelCol='SalePrice')

paramGrid = ParamGridBuilder().addGrid(linearRegressor.regParam, [0.1, 0.01]).addGrid(linearRegressor.elasticNetParam, [0, 1]).build()
crossval = CrossValidator(estimator=linearRegressor, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=10)

crossValModel = crossval.fit(data_transformed)
#linearModel = linearRegressor.fit(train_data_transformed)
trainingSummary = crossValModel.bestModel.summary
#test_data_transformed = assembler.transform(test_data)
#prediction = crossValModel.transform(test_data_transformed)
#evaluator = RegressionEvaluator(predictionCol='prediction', labelCol='SalePrice')
#rmse = evaluator.evaluate(prediction, {evaluator.metricName: "rmse"})
print("RMSE : ", trainingSummary)
