import os 
import sys
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from dataclasses import dataclass
import pandas as pd 
from dotenv import load_dotenv
import pymysql
import pickle 
import numpy as np 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score


load_dotenv()
host = os.getenv('host')
user = os.getenv('user')
passward = os.getenv('password')
database = os.getenv('database')

def read_sql_data():
    logging.info('Reading mysql database started')
    try:
        mydb = pymysql.connect(
        host=host,
        user=user,
        password=passward,
        database=database
)
        logging.info("Conection Established ",mydb)

        #df = pd.read_sql_query("SELECT * FROM retail_fashion_dataset",mydb)
        df = pd.read_sql_query("SELECT * FROM collage.data",mydb)
        print(df.head())
        return df
    
    except Exception as ex:
        raise CustomException(ex,sys)


def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)



def evaluate_model(xtrain,ytrain,xtest,ytest,models,params):
    try:
        report = {}

        for model_name,model in models.items():
            param = params[model_name]

            gs = GridSearchCV(model,param,cv=3)
            gs.fit(xtrain,ytrain)

            best_params = gs.best_params_
            model.set_params(**best_params)
            model.fit(xtrain,ytrain)

            ypred_train = model.predict(xtrain)
            ypred_test = model.predict(xtest)

            train_score = r2_score(ytrain,ypred_train)
            test_score = r2_score(ytest,ypred_test)

            report[model_name] = test_score
            print(report)
            #print(best_params)

        return report

    except Exception as ex:
        raise CustomException(ex,sys)
    

def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as ex:
        raise CustomException(ex,sys)
    



        
