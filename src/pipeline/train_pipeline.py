from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

from src.logger import logging
from src.exception import CustomException

import sys


class TrainPipeline:

    def run_pipeline(self):

        try:

            logging.info("Pipeline started")

            ingestion = DataIngestion()

            train_path, test_path = ingestion.initiate_data_ingestion()

            transformation = DataTransformation()

            train_array, test_array = transformation.initiate_data_transformation(
                train_path,
                test_path
            )

            trainer = ModelTrainer()

            trainer.initiate_model_training(
                train_array,
                test_array
            )

            logging.info("Model training completed")

        except Exception as e:

            raise CustomException(e, sys)


if __name__ == "__main__":

    pipeline = TrainPipeline()

    pipeline.run_pipeline()