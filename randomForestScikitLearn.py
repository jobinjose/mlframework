import pandas as p
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from dataProcessing import dataProcessing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

'''
def dataProcessing1(df):
	#print(houseData.dtypes)
    #x_objdf = x.select_dtypes(include=['float64']).copy()
    # Check the missing values
    #print(x_objdf.isnull().sum())
    #print(x_objdf["Electrical"].value_counts())
    #print(x_objdf["BsmtQual"].value_counts())
    #print(x_objdf["BsmtCond"].value_counts())
    #print(x_objdf["BsmtExposure"].value_counts())
    #print(x_objdf["BsmtFinType1"].value_counts())
    #print(x_objdf["BsmtFinType2"].value_counts())
    #print(x_objdf["GarageType"].value_counts())
    #print(x_objdf["GarageFinish"].value_counts())
    #print(x_objdf["GarageQual"].value_counts())
    #print(x_objdf["GarageCond"].value_counts())
    #print(x_objdf["MasVnrType"].value_counts())
    # Filling missing 'Electrical' with SBrkr which is the most number present
    df = df.fillna({"Electrical":"SBrkr"})
    df = df.fillna({"BsmtQual":"TA"})
    df = df.fillna({"BsmtCond":"TA"})
    df = df.fillna({"BsmtExposure":"No"})
    df = df.fillna({"BsmtFinType1":"Unf"})
    df = df.fillna({"BsmtFinType2":"Unf"})
    df = df.fillna({"GarageType":"Attchd"})
    df = df.fillna({"GarageFinish":"Unf"})
    df = df.fillna({"GarageQual":"TA"})
    df = df.fillna({"GarageCond":"TA"})
    df = df.fillna({"MasVnrType":"None"})
    df = df.fillna(df.mean())
    #print(x_objdf["Electrical"].value_counts()) 
    #print(x_objdf.isnull().sum())
    #x_objdf = df.select_dtypes(include=['float64']).copy()
    #print(x_objdf.isnull().sum())


    # One hot encoding
    #df = p.get_dummies(df) # if columns = None, then all the categorical columns will be encoded
    #print(list(x_objdf))
    return df
'''

if __name__=="__main__":
    #import dataset
    houseData = p.read_csv("housing dataset.csv")
    #print(houseData.head())

    #all the variables except SalePrice is taken as X variables
    x=houseData.drop(['Alley','PoolQC','MiscFeature','Fence','FireplaceQu','HouseStyle'],axis=1)
    x=dataProcessing(x)    #dataprocessing
    # Saleprice is assined as target variable
    y=x['SalePrice']
    x=x.drop(['SalePrice'],axis=1)
    #print(list(x))
    #print(y)
    
    # Splitting the dataset into training set(70%) and test set (30%)
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30)
    #print(x_train) #1022 rows
    #print(x_test) #438 rows

    '''
    # train data processing
    #x_train=dataProcessing(x_train)
    # One hot encoding
    #x_train_afterdummies = p.get_dummies(x_train) # if columns = None, then all the categorical columns will be encoded
    #print(list(x_objdf))
    #print(list(x_train))
    
    
    # test data processing
    x_test=dataProcessing(x_test)
    x_train['label'] = "train"
    x_test['label'] = "test"
    #print(list(x_train))
    x_concat = p.concat([x_train,x_test])
    #print(x_concat['label'])
    x_concatdummies = p.get_dummies(x_concat)
    #print(x_concatdummies['label_test'])
    x_train = x_concatdummies[x_concatdummies['label_train'] == 1]
    x_test = x_concatdummies[x_concatdummies['label_test'] == 1]

    # Drop your labels
    x_train = x_train.drop(['label_train','label_test'], axis=1)
    x_test = x_test.drop(['label_train','label_test'], axis=1)
    '''


    # Random Forest Regression
    RFRegressor = RandomForestRegressor(n_estimators = 5000)
    RFRegressor.fit(x_train,y_train) 


    # testing 
    y_result = RFRegressor.predict(x_test)
    #y_test['result'] = y_result
    #print(y_result)

    #print(y_test)

    #Calculating RMSE
    RMSE = mean_squared_error(y_test,y_result)
    #accuracy = accuracy_score(y_test,y_result)
    print(RMSE)
    #print(accuracy)



