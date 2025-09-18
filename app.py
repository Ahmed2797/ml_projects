from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
import sys
import logging
#from src.mlproject.components import data_ingestion
from src.mlproject.components.data_ingestion import Dataingestion
from src.mlproject.components.data_ingestion import DataIngestionConfig
from src.mlproject.components.data_transformation import DataTransformationConfig
from src.mlproject.components.data_transformation import DataTransformation
from src.mlproject.components.model_trainer import Model_Train


if __name__=='__main__':
    logging.info('Execution Start')
    try:
        #data_ingestion_config = DataIngestionConfig()
        dt_ingestion = Dataingestion()
        train_data_path,test_data_path = dt_ingestion.initiate_data_ingestion()

        #data_transformation_config = DataTransformationConfig()
        dt_transformation = DataTransformation()
        train_arr,test_arr,_ = dt_transformation.initiate_data_transformation(train_data_path,test_data_path)#('artifacts/train.csv','artifacts/test.csv')
        print('Train_array_Shape:',train_arr.shape)
        print('Test_array_Shape:',test_arr.shape)

        # # model_train_config = ModelTrain_config()
        model_train = Model_Train()
        print(model_train.init_model_traniner(train_arr,test_arr))
        
    except Exception as e:
        raise CustomException(e, sys)



