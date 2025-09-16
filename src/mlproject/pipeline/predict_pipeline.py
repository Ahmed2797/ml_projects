import os 
import sys 
from src.mlproject.exception import CustomException 
from src.mlproject.logger import logging
import pandas as pd
from src.mlproject.utils import load_object

class Pipeline_Predict:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path = 'art\model.pkl'
            preprocessor_path = 'art\preprocessor.pkl'
            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)
            data_scale = preprocessor.transform(features)
            pred = model.predict(data_scale)
            return pred
        except Exception as e:
            raise CustomException(e,sys)

class Customdata:
    def __init__(self,age,gender,math_score,english_score,total_score):
        self.age = age
        self.gender = gender
        self.math_score = math_score
        self.english_score = english_score
        self.total_score = total_score
        
    def getdata_as_dataframe(self):
        try:
            custom_data_point = {
                'gender': [self.gender],
                'age':[self.age],
                'math_score':[self.math_score],
                'english_score':[self.english_score],
                'total_score':[self.total_score],

            }

            return pd.DataFrame(custom_data_point)
        except Exception as ex:
            raise CustomException(ex,sys)