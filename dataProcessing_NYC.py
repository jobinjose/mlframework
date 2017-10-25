import pandas as p
import sklearn
import numpy as np

from math import radians, cos, sin, asin, sqrt
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    #print("llalalalalalala : ", [lon1, lat1, lon2, lat2])
    lon1, lat1, lon2, lat2 = map(radians, [lon1.astype(float), lat1.astype(float), lon2.astype(float), lat2.astype(float)])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    # Radius of earth in kilometers is 6371
    km = 6371* c
    return km

def dataProcessing(df):

    df['pickup_datetime'] = p.to_datetime(df['pickup_datetime'])
    df['pickup_datetime'] = (df['pickup_datetime'] - df['pickup_datetime'].min())  / np.timedelta64(1,'D')
    #print(df['pickup_datetime'])

    df['dropoff_datetime'] = p.to_datetime(df['dropoff_datetime'])
    df['dropoff_datetime'] = (df['dropoff_datetime'] - df['dropoff_datetime'].min())  / np.timedelta64(1,'D')
    #print(df['dropoff_datetime'])
    harvesine_Dist = []
    for lon1,lat1,lon2,lat2 in zip(df['pickup_longitude'], df['pickup_latitude'], df["dropoff_longitude"], df["dropoff_latitude"]):
        #print("long iteration : ", lon1, " ",lat1, " ", lon2, " ",lat2)
        harvesine_Dist.append(haversine(lon1, lat1, lon2, lat2))
    df["HarvesineDist"] = harvesine_Dist

    df['id'] = df['id'].str[3:]

    # One hot encoding
    #df = df.drop(['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude'],axis=1)
    df['store_and_fwd_flag'] = p.get_dummies(df['store_and_fwd_flag']) # if columns = None, then all the categorical columns will be encoded
    #print(list(x_objdf))

    return df


if __name__=="__main__":
    #import dataset
    NYCdata = p.read_csv("New York City Taxi Trip Duration.csv")
    df=NYCdata
    df_new=dataProcessing(df)
