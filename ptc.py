#

'''
from setuptools import setup,find_packages
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path:str)->List[str]:

    with open(file_path) as file_obj:
        requirements = file_obj.readlines()

        #requirements = [reg.replace('/n','') for reg in requirements] 
        requirements = [reg.strip() for reg in requirements if reg.strip()]


        #requirements = [req for req in requirements if HYPEN_E_DOT in req] #-- only '-e .'

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
        #requirements = [req for req in requirements if HYPEN_E_DOT not in req]
            
        return requirements

setup(
    name='ml_projects',
    version='0.0.1',
    author='Ahmed',
    author_email='tanvirahmed754575@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
'''

# template.py

'''
from pathlib import Path
import os
import logging

logging.basicConfig(level=logging.INFO)

project_name = 'mlproject'
list_of_file = [
    f'src/{project_name}/__init__.py',
    f'src/{project_name}/components/__init__.py',
    f'src/{project_name}/components/data_ingestion.py',
    f'src/{project_name}/components/data_transformation.py',
    f'src/{project_name}/components/model_trainer.py',
    f'src/{project_name}/components/model_monitering.py',
    f'src/{project_name}/pipeline/train_pipeline.py',
    f'src/{project_name}/pipeline/predict_pipeline.py',
    f'src/{project_name}/exception.py',
    f'src/{project_name}/logger.py',
    f'src/{project_name}/utils.py',
    'app.py',
    'main.py',
    'requirements.txt'
]

for filepath in list_of_file:
    filepath = Path(filepath)
    file_dir, file_name = os.path.split(filepath)

    
    if file_dir != "":
        os.makedirs(file_dir, exist_ok=True)
        logging.info(f'Creating directory: {file_dir} for the file {file_name}')

    
    if (not os.path.exists(filepath) or os.path.getsize(filepath) == 0):
        with open(filepath, 'w') as f:
            pass
        logging.info(f'Creating empty file: {file_name}')
    else:
        logging.info(f'{file_name} is already exists')


sre/
└── mlproject/
    ├── __init__.py
    ├── components/
    │   ├── __init__.py
    │   ├── data_ingestion.py
    │   ├── data_transformation.py
    │   ├── model_trainer.py
    │   └── model_monitering.py
    ├── pipeline/
    │   ├── train_pipeline.py
    │   └── predict_pipeline.py
    ├── exception.py
    ├── logger.py
    └── utils.py
app.py
main.py
requirements.txt

'''

# data_ingestion 

'''
import os 
import sys 
from dataclasses import dataclass
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class DataIngestionconfig:
    train_data_path:str = os.path.join("artifacts","train.csv")
    test_data_path:str = os.path.join("artifacts","test.csv")
    raw_data_path:str = os.path.join("artifacts","raw.csv")

class DataIngestion:
    def __init__(self):
        self.data_ingestiuon_config = DataIngestionconfig()
    
    def init_data_ingestion(self):
        try:
            logging.info('Read data connection')
            df = pd.read_csv('Load the data')
            os.makedirs(os.path.join(self.data_ingestiuon_config.raw_data_path),exist_ok=True)
            df.to_csv(self.data_ingestiuon_config.raw_data_path,index=False,header=True)
            
            train_path,test_path = train_test_split(df,test_size=0.2,random_state=42)
            train_path.to_csv(self.data_ingestiuon_config.train_data_path,index=False,header=True)
            test_path.to_csv(self.data_ingestiuon_config.test_data_path,index=False,header=True)

            logging.info('Data import Succes')
            return (
                self.data_ingestiuon_config.train_data_path,
                self.data_ingestiuon_config.test_data_path
            )

        except Exception as ex:
            raise CustomException(ex,sys)

'''

# data_transformation.py

'''
import os
import sys 
from dataclasses import dataclass 
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
import numpy as np 
import pandas as pd
from src.mlproject.utils import save_object

@dataclass 
class DataTransformatinConfig:
    preprocessor_pkl = os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.preprocessor_pkl_config = DataTransformatinConfig()

    def get_transformation(self):
        try:
            logging.info('Starting the Transformation')
            numeric_columns = ['pass the numeric_columns & logic']
            catgorical_columns = ['pass the categorical_colums & logic']

            num_pipeline = Pipeline(steps=[
                ('impute',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ('impute',SimpleImputer(strategy='most_frequent')),
                ('ohe',OneHotEncoder(handle_unknown='ignore')),
                ('scaler',StandardScaler(with_mean=False))
            ])

            preprocessor = ColumnTransformer([
                ('num_col',num_pipeline,numeric_columns),
                ('cat_col',cat_pipeline,catgorical_columns)
            ])

            logging.info('Ready preprocessing for Data_transformatin')

            return preprocessor

        except Exception as ex:
            raise CustomException(ex,sys)
    
    def init_data_transform(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('Train&Test')

            target_column = 'Pass the Target_Column'

            input_train_df = train_df.drop([target_column],axis=1)
            target_train_df = train_df[target_column]

            input_test_df = test_df.drop([target_column],axis=1)
            target_test_df = test_df[target_column]

            preprocessor_obj = self.get_transformation()

            train_df_arr = preprocessor_obj.fit_transform(input_train_df)
            test_df_arr = preprocessor_obj.transform(input_test_df)

            train_arr = np.c_[train_df_arr,np.array(target_train_df)]
            test_arr = np.c_[test_df_arr,np.array(target_test_df)]
            logging.info('Data_Transformation Completed')

            save_object(
                filename = self.preprocessor_pkl_config.preprocessor_pkl,
                obj = preprocessor_obj
            )

            return (
                train_arr,test_arr,
                self.preprocessor_pkl_config.preprocessor_pkl
            )
        
        except Exception as e:
            raise CustomException(e,sys)

'''

# model_training.py

'''  
import os 
import sys 
from src.mlproject.exception import CustomException 
from src.mlproject.logger import logging
from dataclasses import dataclass 


@dataclass
class ModelTrainer_config:
    model_train_path = os.path.join('art','model.pkl')
class Model_train:
    def __init__(self):
        self.model_train_config = ModelTrainer_config()

    def init_model_train(self,train_arr,test_arr):
        try:
            logging.info('started the model training')
            xtrain = train_arr[:,:-1]
            xtest = test_arr[:,:-1]
            ytrain = train_arr[:,-1]
            ytest = test_arr[:,-1]
            
            models = {
                # "Random Forest": RandomForestRegressor(),
                # "Decision Tree": DecisionTreeRegressor(),
                # "Gradient Boosting": GradientBoostingRegressor(),
                # "Linear Regression": LinearRegression(),
                # "XGBRegressor": XGBRegressor(),
                # "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                # "AdaBoost Regressor": AdaBoostRegressor(),
            }

            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter': ['best', 'random'],
                    'max_depth': [3, 5, 10, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 5],
                    'max_features': ['sqrt', 'log2', None]
                },
                "Random Forest": {
                    'n_estimators': [50, 100, 200, 300],
                    'criterion': ['squared_error', 'friedman_mse'],
                    'max_depth': [5, 10, None],
                    'max_features': ['sqrt', 'log2', None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                "Gradient Boosting": {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.1, 0.05, 0.01],
                    'subsample': [0.6, 0.8, 1.0],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 5]
                },
                "Linear Regression": {
                    'fit_intercept': [True, False]
                },
                "XGBRegressor": {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.1, 0.05, 0.01],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.6, 0.8, 1.0],
                    'colsample_bytree': [0.6, 0.8, 1.0],
                    'gamma': [0, 1, 5]
                },
                "CatBoosting Regressor": {
                    'iterations': [100, 200, 300],
                    'depth': [6, 8, 10],
                    'learning_rate': [0.1, 0.05, 0.01],
                    'l2_leaf_reg': [1, 3, 5, 7]
                },
                "AdaBoost Regressor": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.1, 0.5, 1.0],
                    'loss': ['linear', 'square', 'exponential']
                }
            }

            model_report = evaluate_model(xtrain,ytrain,xtest,ytest,models,params)
            best_model_name = max(model_report,key = model_report.get) 
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise Exception("Best_model doesnot found")
            
            save_object(
                self.model_train_config.model_train_path,
                best_model
            )
            pred = best_model.predict(xtest)
            R2_score = r2_score(ytest,pred)
            return R2_score
        except Exception as ex:
            raise CustomException(ex,sys)



def evaluate_model(xtrain,ytrain,xtest,ytest,models,params):
    try:
        model_report = {}

        for model_name,model in models.items():
            param = params[model_name]

            gs = GridSearchCV(model,param,cv=3)
            gs.fit(xtrain,ytrain)

            best_params = gs.best_params_
            model.set_params(**best_params)
            model.fit(xtrain, ytrain)


            pred_train = model.predict(xtrain)
            pred_test = model.predict(xtest)

            train_score = r2_score(ytrain,pred_train)
            test_score = r2_score(ytest,pred_test)
            model_report[model_name] = test_score
        return model_report
    except Exception as ex:
        raise CustomException(ex,sys)

'''
    
# utils.py

'''
import os
import sys
import pickle
from src.mlproject.exception import CustomException

def save(file_path,obj):
    try:
        file_dir = os.path.dirname(file_path)
        os.makedirs(file_dir,exist_ok=True)

        with open(file_dir,'wb') as f:
            pickle.dump(obj,f)
    except Exception as ex:
        raise CustomException(ex,sys)

'''

# app.py

'''
import os 
import sys 
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from src.mlproject.components.data_ingestion import Dataingestion
from src.mlproject.components.data_transformation import DataTransformation

if __name__=='__main__':
    logging.info("Execution Start")

    try:
        dt_ingestion = Dataingestion()
        train_data_path,test_data_path = dt_ingestion.initiate_data_ingestion()

        dt_transformation = DataTransformation()
        train_arr,test_arr,_ = dt_transformation.initiate_data_transformation(train_data_path,test_data_path)#('art/train.csv','art/test.csv')
        
        model_train = Model_train()
        print(model_train.init_model_train(train_arr,test_arr))

    except Exception as ex:
        raise CustomException(ex,sys)

'''

# fk.py

'''
from flask import Flask,request,render_template
from src.mlproject.pipeline.predict_pipeline import Customdata,Pipeline_Predict

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=Customdata(
            gender=request.form.get(''),
            race_ethnicity=request.form.get(''),
            parental_level_of_education=request.form.get(''),
            lunch=request.form.get(''),
            test_preparation_course=request.form.get(''),
            reading_score=float(request.form.get('')),
            writing_score=float(request.form.get(''))

        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=Pipeline_Predict()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0")        

'''


# predict_pipeline.py

'''
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
        preprocessor_path = os.path.join('artifacts','preprocessor.pkl')
        model_path = os.path.join('art','model.pkl')

        self.model = load_object(model_path)
        self.preprocessor = load_object(preprocessor_path)

    def predict(self,features):
        df_scale = self.preprocessor.transform(features)
        pred = self.model.predict(df_scale)
        return pred
    
'''

# --------------------------------practice mlflow
'''
best_model_name = max(model_report,key=model_report.get)
best_model_score = model_report[best_model_name]
best_model = models[beat_model_name]

model_names = list(params.keys())
actural = ''

for model in model_names:
    if best_model_name == model :
        actural_model = actural + model

best_params = params[actural_model]

mlflow.ser_register_ural('ejckfle2jopfepojfpefe2[je[j32h3ddsdklsakjdklasl]]')
track_url = url_parces(mlflow.get_tracking_url()).schema

with mlflow.start_run():
    predict = best_model.predict(xtest)

    (rmse,mse,r2) = evulate_matries(true,oredict)
    mlflow.log_params(best_params)
    mlflow.log_metrics(rmse)

    if track_url != 'file':
        mlflow.sklearn.log_model(best_model,'model',)

'''









