import pandas as p
import sklearn


def dataProcessing_sum_without_noise(df):
    # Check the missing values
    #df_processed = df.isnull().sum()
    missing_col_list = df.columns[df.isnull().any()].tolist()
    #print(len(missing_col_list))    no missing values in the dataset

    #correlation = data.corr()
    #print(correlation)
    df=df.drop(['Feature 2','Feature 3','Feature 4','Feature 6','Feature 7','Feature 8','Feature 9','Feature 10'],axis=1)

    return df

def dataProcessing_sum_noise(df):
    missing_col_list = df.columns[df.isnull().any()].tolist()
    #print(missing_col_list)
    #correlation = df.corr()
    #print(correlation)
    return df


if __name__=="__main__":
    #import dataset
    #dataset = p.read_csv("The SUM dataset, without noise.csv",delimiter=";")
    #dataset = p.read_csv("The SUM dataset, with noise.csv",delimiter=";")
    dataset = p.read_csv("winequality-white.csv",delimiter=None)
    #print(dataset['Target'].value_counts())
    print(dataset.shape)
    #dataset = p.read_csv("housing dataset.csv",delimiter=None)
    #print(dataset.shape)

    # dropping the columns with most number of null values
    #df=houseData.drop(['Alley','PoolQC','MiscFeature','Fence','FireplaceQu'],axis=1)
    #dataProcessing_sum_without_noise(dataset)
    #dataProcessing_sum_noise(dataset)
    #print(list(df_new))
