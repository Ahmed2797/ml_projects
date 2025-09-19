import os 
import sys 
from src.mlproject.exception import CustomException 
from src.mlproject.logger import logging
import pandas as pd
from src.mlproject.utils import load_object


class Custom:
    def __init__(self,
                 gender:str,
                 race_ethnicity:str,
                 parental_level_of_education:str,
                 lunch:str,
                 test_preparation_course:str,
                 reading_score:int,
                 writing_score:int):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score 

    def getdata_as_dataframe(self):
        try:
            custom_datapoint = {
                'gender':[self.gender],
                'race_ethnicity':[self.race_ethnicity],
                'parental_level_of_education':[self.parental_level_of_education],
                'lunch':[self.lunch],
                'test_preparation_course':[self.test_preparation_course],
                'reading_score':[self.reading_score],
                'writing_score':[self.writing_score]
            }

            return pd.DataFrame(custom_datapoint)
        
        except Exception as ex:
            raise CustomException(ex,sys)
        

class Predictpipeline:
    def __init__(self):
        preprocessor_path = os.path.join(r'artifacts/preprocessor.pkl')
        model_path = os.path.join(r'artifacts/best_model.pkl')

        
        print(os.path.exists(r'artifacts/best_model.pkl'))
        print(os.path.exists(r'artifacts/preprocessor.pkl'))


        self.model = load_object(model_path)
        self.preprocessor = load_object(preprocessor_path)

    def predict(self,features):
        df_scale = self.preprocessor.transform(features)
        pred = self.model.predict(df_scale)
        return pred