import pandas as pd
import numpy as np
import io
import os
from io import StringIO
from Airbnb_config import FeatureSelection, DateTimeColumns, Geo ,host_response_rate ,treat_missing_first, categorical_encoder, treat_missing_second ,Scaler_Min_Max, Mydimension_reducer
import joblib
from sklearn.model_selection import train_test_split
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
import modin.pandas as mdpd
from sklearn.preprocessing import LabelEncoder

def compute_predictions(filepath):
    

    # Test_df=pd.read_excel(filepath)
    Test_df = pd.read_excel(
        filepath,
        engine='openpyxl',
    )

    print(Test_df.columns)

    df_Ids=Test_df.id

    prepro_pipe=make_pipeline(FeatureSelection()
                   ,categorical_encoder(),
                   Geo(),host_response_rate())


    nwdf=prepro_pipe.fit_transform(Test_df)

    pipefile3=open('Model_Repos/model_pipe.pkl','rb')
    model_pipe=joblib.load(pipefile3)


    #Final Predictions
    finalpred=model_pipe.predict(nwdf)

    Predictions=pd.DataFrame({"id":df_Ids,"log_price":finalpred})
    print(Predictions)

    del nwdf,Test_df

    return Predictions


