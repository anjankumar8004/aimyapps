import pandas as pd
import numpy as np
import io
import os
from io import StringIO
from Airbnb_config import FeatureSelection, DateTimeColumns, Geo ,host_response_rate ,treat_missing_first, categorical_encoder, treat_missing_second ,Scaler_Min_Max, Mydimension_reducer,TextData
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


def compute_predictions(filepath):
    

    # Test_df=pd.read_excel(filepath)
    Test_df = pd.read_excel(
        filepath,
        engine='openpyxl',
    )

    print(Test_df.columns)

    df_Ids=Test_df.id

    Train_df=pd.read_csv("Model_Repos/train.csv")
    print(Train_df.columns)

    Train_df["Flag"]="Train"
    Test_df["Flag"]="Test"
    Test_df["log_price"]=np.nan

    df_All=pd.concat([Test_df,Train_df],axis=0)

    prepro_pipe=make_pipeline(FeatureSelection(),treat_missing_first(df_All)
                   ,Mydimension_reducer(),categorical_encoder(),DateTimeColumns(),
                   TextData(),Geo(),host_response_rate())

    nwdf=prepro_pipe.fit_transform(df_All)



    NewTrain=nwdf[nwdf["Flag"]=="Train"]
    NewTest=nwdf[nwdf["Flag"]=="Test"]

    del NewTrain["Flag"]
    del NewTest["Flag"]
    del NewTest["log_price"]

    #Take out X and Y from Train , then give X to PCA
    Train_X=NewTrain.drop(["log_price"],axis=1).copy()
    Train_Y=NewTrain["log_price"].copy()


    pca_pipe=make_pipeline(Scaler_Min_Max(),treat_missing_second(),PCA(n_components = round((Train_X.shape[1]*70)/100)))

    model_pipe=make_pipeline(xgboost.XGBRegressor(objective='reg:linear',scale_pos_weight=1.0,max_delta_step=5.0,min_child_weight=10.0,max_depth=8,eta=0.1,random_state=45,n_estimators=100, eval_metric='error', learning_rate=0.1,subsample=1.0,colsample_bytree=0.5,colsample_bylevel=0.5, tree_method='gpu_hist', predictor='gpu_predictor'))

    #fit_transform on train,transform on test
    pca_Train_X=pca_pipe.fit_transform(Train_X)
    pca_Test=pca_pipe.transform(NewTest)


    #fit on pca_Train_X and Train_Y
    model_pipe.fit(pca_Train_X,Train_Y)

    #predict
    trainpred=model_pipe.predict(pca_Train_X)

    #Prediction on Train
    # print("RMSE "+str(mean_squared_error(Train_Y,trainpred,squared=False)))
    # print("MSE "+str(mean_squared_error(Train_Y,trainpred)))

    residuals=(Train_Y-trainpred).mean()
    print(residuals)

    #Final Predictions
    finalpred=model_pipe.predict(pca_Test)

    

    Predictions=pd.DataFrame({"id":df_Ids,"log_price":finalpred})
    print(Predictions)

    del Train_df,Test_df,df_All,NewTrain,NewTest,Train_X,Train_Y,pca_Train_X,pca_Test,trainpred

    return Predictions


