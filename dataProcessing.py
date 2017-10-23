import pandas as p
import sklearn


def dataProcessing(df):
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


if __name__=="__main__":
    #import dataset
    houseData = p.read_csv("housing dataset.csv")
    #print(houseData.head())

    # dropping the columns with most number of null values
    df=houseData.drop(['Alley','PoolQC','MiscFeature','Fence','FireplaceQu'],axis=1)
    df_new=dataProcessing(df)
    #print(list(df_new))