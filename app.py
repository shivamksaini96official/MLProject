from src.MLProject.logger import logging
from src.MLProject.exception import CustomException
import sys
from src.MLProject.components.data_ingestion import DataIngestion,DataIngestionConfig
if __name__ == "__main__":
  logging.info("The execution has started.")

  try:
    data_ingestion = DataIngestion()
    data_ingestion.initiate_data_ingestion()
  except Exception as e:
    logging.info("Custom Exception")
    raise CustomException(e,sys)