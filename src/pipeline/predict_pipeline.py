import os
import sys
import pandas as pd
import pickle

from src.exception import CustomException
from src.logger import logging


class PredictPipeline:

    def __init__(self):

        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")


    def load_object(self, file_path):

        try:

            logging.info(f"Loading object from {file_path}")

            with open(file_path, "rb") as file_obj:

                return pickle.load(file_obj)

        except Exception as e:

            raise CustomException(e, sys)


    def predict(self, input_df):

        try:

            logging.info("Loading preprocessor and model")

            model = self.load_object(self.model_path)

            preprocessor = self.load_object(self.preprocessor_path)

            logging.info("Applying preprocessing")

            data_scaled = preprocessor.transform(input_df)

            logging.info("Making prediction")

            prediction_numeric = model.predict(data_scaled)

            # Convert numeric prediction → business output
            prediction = [
                "yes" if pred == 1 else "no"
                for pred in prediction_numeric
            ]

            logging.info("Prediction completed")

            return prediction

        except Exception as e:

            raise CustomException(e, sys)