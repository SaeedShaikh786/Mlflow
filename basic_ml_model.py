import mlflow
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import mlflow.sklearn
import os
import sys
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,accuracy_score
from sklearn.linear_model import ElasticNet
from src.exception import CustomException
from src.logger import logging
import argparse
from sklearn.ensemble import RandomForestClassifier


try:
    def get_data():
        ### importing the dataset
        data=pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",sep=";")
        return data
except Exception as e:
    print(e)


def Main(n_estimator,max_depth):
    try:
        df=get_data()
        logging.info("dataset gathered")
        X=df.drop(["quality"],axis=1)
        y=df["quality"]
                    
                    ## Data spliting
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=32)

                    ## Scaling
        scalar=StandardScaler()
        X_train=scalar.fit_transform(X_train)
        X_test=scalar.transform(X_test)
        logging.info("Scaling is done")

                  #model
        ''' model=ElasticNet()
        model.fit(X_train,y_train)
        y_pred=model.predict(X_test)
        logging.info("Prediction has done")
        ## mlflow.sklearn.log_model(model,"ElasticNet")
        with mlflow.start_run():'''
        with mlflow.start_run():
            model=RandomForestClassifier(n_estimators=n_estimator,max_depth=max_depth)
            model.fit(X_train,y_train)
            y_pred=model.predict(X_test)
            mlflow.sklearn.log_model(model,"RandomForestClassifier")

            ## r_score=r2_score(y_test,y_pred)
            ## mlflow.log_metric("R2 score",r_score)
            ## mlflow.sklearn.log_model(model,"ElasticNet")
            ## print("R2 score:",r_score)
            acc=accuracy_score(y_test,y_pred)
            mlflow.log_metric("Accuracy Score",acc)
            print(f"Accuracy score is {acc}")
            logging.info("metrics has captured in mlflow")
    except Exception as e:
        raise CustomException(e,sys)


if __name__=="__main__":
    arg_parse=argparse.ArgumentParser()
    arg_parse.add_argument("--n_estimator","-n",default=30,type=int)
    arg_parse.add_argument("--max_depth","-m",default=4,type=int)
    parse_arg=arg_parse.parse_args()
    try:
        Main(n_estimator=parse_arg.n_estimator,max_depth=parse_arg.max_depth)
    except Exception as e:
        print(e)
