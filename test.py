import mlflow
import os


def cal(a=4,b=90):
    c=a*b
    return c





if __name__=="__main__":
    """ to start the server of mlflow """
    with mlflow.start_run():
        x,y=10,30
        z =cal(x,y)
        """ tracking the methiod by mlflow"""
        mlflow.log_param("x",x)
        mlflow.log_param("y",y)
        mlflow.log_metric("z",z)

