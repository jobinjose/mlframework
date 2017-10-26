import pandas as p
import sklearn


def dataProcessing_kc_data(df):
    # Check the missing values
    #df_processed = df.isnull().sum()
    missing_col_list = df.columns[df.isnull().any()].tolist()
    df=df.drop(['date'],axis=1)
    #print(len(missing_col_list))    #no missing values in the dataset

    #correlation = df.corr()
    #print(correlation)
    #df=df.drop(['Feature 2','Feature 3','Feature 4','Feature 6','Feature 7','Feature 8','Feature 9','Feature 10'],axis=1)

    return df


if __name__=="__main__":
    #import dataset
    #dataset = p.read_csv("The SUM dataset, without noise.csv",delimiter=";")
    #dataset = p.read_csv("The SUM dataset, with noise.csv",delimiter=";")
    dataset = p.read_csv("kc_house_data.csv",delimiter=None)
    #print(dataset['Target'].value_counts())
    print(dataset.shape)
    dataProcessing_kc_data(dataset)
    #dataset = p.read_csv("housing dataset.csv",delimiter=None)
    #print(dataset.shape)

    # dropping the columns with most number of null values
    #df=houseData.drop(['Alley','PoolQC','MiscFeature','Fence','FireplaceQu'],axis=1)
    #dataProcessing_sum_without_noise(dataset)
    #dataProcessing_sum_noise(dataset)
    #print(list(df_new))
