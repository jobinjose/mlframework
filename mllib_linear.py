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

from pyspark import SparkContext, SparkConf

#import dataset
houseData = p.read_csv("housing dataset.csv")

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
(train_data,test_data) = x_new.randomSplit([0.7,0.3])
label='SalePrice'
features=list(filter(lambda w: w not in label, train_data.columns))
#print(features)
assembler = VectorAssembler(inputCols=features,outputCol="features")
train_data_transformed = assembler.transform(train_data)

linearRegressor = LinearRegression(labelCol="SalePrice", featuresCol="features", maxIter=10)
linearModel = linearRegressor.fit(train_data_transformed)

test_data_transformed = assembler.transform(test_data)
prediction = linearModel.transform(test_data_transformed))
evaluator = RegressionEvaluator(predictionCol='prediction', labelCol='label')
rmse = evaluator.evaluate(prediction, {evaluator.metricName: "rmse"})
print("RMSE : ", rmse)






'''


conf = SparkConf().setAppName("LinearRegression").setMaster("local[8]")
sc = SparkContext(conf = conf)
sqlCtx = SQLContext(sc)

data = sqlCtx.read.format("csv").option("header", "true").option("inferSchema", "true").load("U:/Machine learning/Assignment/Task2/House Prices/housing dataset_numbers.csv")

#print(data)

data.cache() # Cache data for faster reuse
data = data.dropna() # drop rows with missing values
#x_train = data.drop('SalePrice',axis = 1)

# Register table so it is accessible via SQL Context
# For Apache Spark = 2.0
data.createOrReplaceTempView("data_geo")
df = data.select('SalePrice','Id','MSSubClass','LotArea','OverallQual','OverallCond','YearBuilt','YearRemodAdd','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','MoSold','YrSold')
temp = df.rdd.map(lambda line:LabeledPoint(line[0],[line[1:]]))
features = df.rdd.map(lambda row: row[1:])

standardizer = StandardScaler()
model = standardizer.fit(features)
features_transform = model.transform(features)

lab = df.rdd.map(lambda row: row[0])

transformedData = lab.zip(features_transform)

transformedData = transformedData.map(lambda row: LabeledPoint(row[0],[row[1]]))

trainingData, testingData = transformedData.randomSplit([.7,.3],seed=1234)

linearModel = LinearRegressionWithSGD.train(trainingData,100,.000000005)

#data_transform = data.rdd.map(lambda x: [Vectors.dense(x[0:35]), x[36]]).toDF(['features', 'label'])

#features.show(5)
#display(data)
#Metrics

prediObserRDDin = trainingData.map(lambda row: (float(linearModel.predict(row.features[0])),row.label))
metrics = RegressionMetrics(prediObserRDDin)

print(metrics.r2)

print('/n')

prediObserRDDout = testingData.map(lambda row: (float(linearModel.predict(row.features[0])),row.label))
metrics = RegressionMetrics(prediObserRDDout)

print(metrics.rootMeanSquaredError)
'''
