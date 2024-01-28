from src.MLProject.logger import logging
from src.MLProject.exception import CustomException
import sys
from src.MLProject.components.data_ingestion import DataIngestion,DataIngestionConfig
from src.MLProject.components.data_transformation import DataTransformation,DataTransformationConfig

if __name__ == "__main__":
  logging.info("The execution has started.")

  try:
    # Data Ingestion
    #data_ingestion_config = DataIngestionConfig()
    data_ingestion = DataIngestion()
    train_data_path,test_data_path = data_ingestion.initiate_data_ingestion()

    # Data Transformation
    data_transformation_config = DataTransformationConfig()
    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data_path,test_data_path)


  except Exception as e:
    logging.info("Custom Exception")
    raise CustomException(e,sys)