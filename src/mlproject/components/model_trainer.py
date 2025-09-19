import os 
import sys 
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging 
from dataclasses import dataclass 
from src.mlproject.utils import save_object,evaluate_model
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from catboost import CatBoostRegressor



@dataclass
class ModelTrain_config:
    model_train_path = os.path.join('artifacts','best_model.pkl')

class Model_Train:
    def __init__(self):
        self.model_train = ModelTrain_config()

    def init_model_traniner(self,train_arr,test_arr):
        try:
            logging.info("Spliting the data")
            xtrain = train_arr[:,:-1]
            ytrain = train_arr[:,-1]
            xtest = test_arr[:,:-1]
            ytest = test_arr[:,-1]

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            params = {
                "Decision Tree": {
                    # 'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter': ['best', 'random'],
                    # 'max_depth': [3, 5, 10, None],
                    # 'min_samples_split': [2, 5, 10],
                    # 'min_samples_leaf': [1, 2, 5],
                    'max_features': ['sqrt', 'log2', None]
                },
                "Random Forest": {
                    'n_estimators': [50, 100, 200, 300],
                    # 'criterion': ['squared_error', 'friedman_mse'],
                    # 'max_depth': [5, 10, None],
                    # 'max_features': ['sqrt', 'log2', None],
                    # 'min_samples_split': [2, 5, 10],
                    # 'min_samples_leaf': [1, 2, 4]
                },
                "Gradient Boosting": {
                    'n_estimators': [100, 200, 300],
                    #'learning_rate': [0.1, 0.05, 0.01],
                    # 'subsample': [0.6, 0.8, 1.0],
                    # 'max_depth': [3, 5, 7],
                    # 'min_samples_split': [2, 5, 10],
                    # 'min_samples_leaf': [1, 2, 5]
                },
                "Linear Regression": {
                    'fit_intercept': [True, False]
                },

                "XGBRegressor": {
                    'n_estimators': [100, 200, 300],
                    #'learning_rate': [0.1, 0.05, 0.01],
                    # 'max_depth': [3, 5, 7],
                    # 'subsample': [0.6, 0.8, 1.0],
                    # 'colsample_bytree': [0.6, 0.8, 1.0],
                    # 'gamma': [0, 1, 5]
                },
                "CatBoosting Regressor": {
                    'iterations': [100, 200, 300],
                    # 'depth': [6, 8, 10],
                    # 'learning_rate': [0.1, 0.05, 0.01],
                    # 'l2_leaf_reg': [1, 3, 5, 7]
                },
                "AdaBoost Regressor": {
                    'n_estimators': [50, 100, 200],
                    # 'learning_rate': [0.1, 0.5, 1.0],
                    # 'loss': ['linear', 'square', 'exponential']
                }
            }

            model_report = evaluate_model(xtrain,ytrain,xtest,ytest,models,params)
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            print(f"Best Model Name: {best_model_name}")
            print(f"Best Model Score: {best_model_score}")

            if best_model_score < 0.6:
                raise Exception('Best model doesnot found')
            
            save_object(
                self.model_train.model_train_path,
                best_model
            )

            pred = best_model.predict(xtest)

            R2_scre = r2_score(ytest,pred)
            # print('best_model',best_model)
            # print('R2_scre',R2_scre)
            # print('best_model_score',best_model_score)

            return R2_scre

        except Exception as ex:
            raise CustomException(ex,sys)



