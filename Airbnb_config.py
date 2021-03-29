import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.pipeline import make_pipeline 
from sklearn.linear_model import LinearRegression
import xgboost
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import category_encoders as ce
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
###################################################################################################
###################################################################################################

#### Step2-Select the features and discard other.
#Features not required ["id",thumbnail_url"," "]




###################################################################################################

#### Step2-Select the features and discard other.
#Features not required ["id",thumbnail_url"," "]

class FeatureSelection(BaseEstimator,TransformerMixin):
    def __init__(self):
        self.cols = ['property_type', 'room_type', 'accommodates',
       'bathrooms', 'bed_type', 'cancellation_policy', 'cleaning_fee', 'city',
        'host_has_profile_pic',
       'host_identity_verified', 'host_response_rate',
       'instant_bookable', 'latitude', 'longitude',
       'neighbourhood', 'number_of_reviews', 'review_scores_rating'
        , 'bedrooms', 'beds']
      
        
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        Z=X.copy()
        Z=Z[self.cols]
        return Z
    
    
###################################################################################################
###################################################################################################

#datetime-[first_review,host_since,last_review ]
class DateTimeColumns(BaseEstimator,TransformerMixin):
         
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        Z=X.copy()
        for col in ["first_review","host_since","last_review"]:
            Z[col]=pd.to_datetime(Z[col])
            wdays=Z[col].dt.dayofweek
            month=Z[col].dt.month
            day=Z[col].dt.day

            Z[col+'_'+'week'+'_sin']=np.sin(2*np.pi*wdays/7)
            Z[col+'_'+'week'+'_cos']=np.cos(2*np.pi*wdays/7)

            Z[col+'_'+'month'+'_sin']=np.sin(2*np.pi*month/12)
            Z[col+'_'+'month'+'_cos']=np.cos(2*np.pi*month/12)

            Z[col+'_'+'month_day'+'_sin']=np.sin(2*np.pi*day/31)
            Z[col+'_'+'month_day'+'_cos']=np.cos(2*np.pi*day/31)

            del Z[col]
            X=Z.copy()
        return X

#Text=[amenities,name,description]
###################################################################################################
###################################################################################################


class TextData(BaseEstimator,TransformerMixin):
    
    def __init__(self):
        self.tfid1=TfidfVectorizer(lowercase= True, max_features=1000, stop_words=ENGLISH_STOP_WORDS)
        self.tfid2=TfidfVectorizer(lowercase= True, max_features=1000, stop_words=ENGLISH_STOP_WORDS)
        self.tfid3=TfidfVectorizer(lowercase= True, max_features=1000, stop_words=ENGLISH_STOP_WORDS)
    
    def fit(self,X,y=None):
        self.tfid1.fit(X["amenities"].astype('U'))
        self.tfid2.fit(X["name"].astype('U'))
        self.tfid3.fit(X["description"].astype('U'))
        return self
    
    def transform(self,X):
            Z=X.copy()

            
            amen=self.tfid1.transform(Z["amenities"].astype('U'))
            nam=self.tfid1.transform(Z["name"].astype('U'))
            desc=self.tfid1.transform(Z["description"].astype('U'))
            
            amen=pd.DataFrame(amen.toarray())
            nam=pd.DataFrame(nam.toarray())
            desc=pd.DataFrame(desc.toarray())
            
            del Z["amenities"]
            del Z["name"]
            del Z["description"]
            
            Z=pd.concat([Z,amen,nam,desc],axis="columns")
            X=Z.copy()
            return X

###################################################################################################
###################################################################################################

## e)Geo=[latitude,longitude]

class Geo(BaseEstimator,TransformerMixin):
    #lat long represent a 3D space with two values , you can create x,y,z from them like this
    def __init__(self):
        pass
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        Z=X.copy()
        x_latlong=np.cos(Z["latitude"])*np.cos(Z["longitude"])
        y_latlong=np.cos(Z["latitude"])*np.sin(Z["longitude"])
        z_latlong=np.sin(Z["latitude"])
        
        
        Z["x_latlong"]=x_latlong
        #Z["x_latlong"]=x1.fit_transform(Z["x_latlong"])
        
        Z["y_latlong"]=y_latlong
        #Z["y_latlong"]=x1.fit_transform(Z["y_latlong"])
        
        Z["z_latlong"]=z_latlong
        #Z["z_latlong"]=x1.fit_transform(Z["z_latlong"])
        
        X= Z.drop(["latitude","longitude"],axis="columns").copy()
        
        return X
    
###################################################################################################
###################################################################################################    

### [host_response_rate]
class host_response_rate(BaseEstimator,TransformerMixin):
    #lat long represent a 3D space with two values , you can create x,y,z from them like this
    def __init__(self):
        pass
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        Z=X.copy()
        #First convert whole column to string
        Z["host_response_rate"]=Z["host_response_rate"].astype(str)
        #Find values with % a, remove them ,convert to float, divide by 100
        dec=Z.loc[Z["host_response_rate"].str.contains("%"),"host_response_rate"].str.replace("%","").astype(float)/100
        Z.loc[Z["host_response_rate"].str.contains("%"),"host_response_rate"]=dec
        
        #Treat Missing and convert to float
        Z["host_response_rate"]=Z["host_response_rate"].astype(float)
        Z.loc[Z["host_response_rate"].isnull(),"host_response_rate"]=Z["host_response_rate"].mean()
        
        X=Z.copy()
        return X
        
###################################################################################################
###################################################################################################        

#To treat missing values , Just before Executing dummies
class treat_missing_first(BaseEstimator,TransformerMixin):
    def __init__(self,old_df):
        self.old_df=old_df
                    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        data=X.copy()
        data.reset_index(drop=True,inplace=True) 
        
        categorical_columns = []
        numeric_columns = []
        for c in data.columns:
            if self.old_df[c].map(type).eq(str).any(): #check if there are any strings in column
                categorical_columns.append(c)
            else:
                numeric_columns.append(c)

        #create two DataFrames, one for each data type
        data_numeric = data[numeric_columns]
        data_categorical = pd.DataFrame(data[categorical_columns])


        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        data_numeric = pd.DataFrame(imp.fit_transform(data_numeric), columns = data_numeric.columns) #only apply imputer to numeric columns


        #you could do something like one-hot-encoding of data_categorical here

        #join the two masked dataframes back together
        data_joined = pd.concat([data_numeric, data_categorical], axis = 1)

        return data_joined

###################################################################################################
###################################################################################################
    
# dummies Categorical columns
class categorical_encoder(BaseEstimator,TransformerMixin):
    def __init__(self):
        self.cols=["property_type","room_type","bed_type","cancellation_policy",
                             "cleaning_fee","city","neighbourhood",
                             "host_has_profile_pic",
                             "host_identity_verified","instant_bookable"]
        self.LE=LabelEncoder()
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        for col in self.cols:
            nc=self.LE.fit_transform(X[col])
            X[col]=nc
        return X

###################################################################################################
###################################################################################################
    
class treat_missing_second(BaseEstimator,TransformerMixin):
    def __init__(self):

        self.imputer=SimpleImputer(missing_values=np.nan, strategy='mean')
          
                    
    def fit(self,X,y=None):
        self.imputer.fit(X)
        return self
    
    def transform(self,X):
        data=X.copy()
        data=self.imputer.transform(data)
        return data

class Scaler_Min_Max(BaseEstimator,TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()
       
    def fit(self,X,y=None):
        self.scaler.fit(X)
        return self
    
    def transform(self,X):
        scaled=self.scaler.transform(X)
        return scaled
    
    
###################################################################################################
###################################################################################################
class Mydimension_reducer(BaseEstimator,TransformerMixin):
    def __init__(self):
        self.catcols = ["property_type","room_type","bed_type","cancellation_policy",
                             "cleaning_fee","city","neighbourhood",
                             "host_has_profile_pic",
                             "host_identity_verified","instant_bookable"]

    def fit(self,X,y=None):
        return self
    
    
    def transform(self,X): 
        
        Z=X.copy()

        for col in self.catcols:
            no_uniq=Z[col].nunique()
            if(no_uniq>40):
                p= pd.merge( (round(Z.groupby(col)["log_price"].mean(),2)*100),Z[col],on=col)
                Z[col]=p["log_price"].astype(str)+"cat"
        X=Z.copy()
        return X