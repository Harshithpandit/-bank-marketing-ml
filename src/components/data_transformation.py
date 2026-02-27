import os
import sys
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


class DataTransformation:

    def __init__(self):

        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")


    # Your notebook cleaning logic
    def clean_data(self, df):

        try:

            logging.info("Starting data cleaning")

            # Fix category spelling errors
            df['marital'] = df['marital'].replace('maried', 'married')

            job_mapping = {
                "self-employed": "entrepreneur",
                "blue-collar": "technician",
                "admin": "services"
            }

            df['job'] = df['job'].replace(job_mapping)

            # Replace unknown values
            cols = ['job', 'education', 'poutcome', 'contact']

            for col in cols:

                mode_val = df[col].mode()[0]

                df[col] = df[col].replace("unknown", mode_val)

            # Handle missing values categorical
            cat_cols = ['job','education','housing','loan','contact','poutcome']

            for col in cat_cols:

                df[col].fillna(df[col].mode()[0], inplace=True)

            # Handle missing values numeric
            df['balance'].fillna(df['balance'].median(), inplace=True)

            # Drop leakage column
            if 'duration' in df.columns:

                df.drop('duration', axis=1, inplace=True)

            logging.info("Data cleaning completed")

            return df

        except Exception as e:

            raise CustomException(e, sys)


    # Outlier capping from your notebook
    def cap_outliers(self, df):

        try:

            logging.info("Starting outlier treatment")

            numeric_cols = ['age', 'balance', 'campaign', 'pdays', 'previous']

            for col in numeric_cols:

                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)

                IQR = Q3 - Q1

                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR

                df[col] = df[col].clip(lower, upper)

            logging.info("Outlier treatment completed")

            return df

        except Exception as e:

            raise CustomException(e, sys)


    def get_preprocessor(self, df):

        try:

            target_column = "y"

            numerical_columns = df.select_dtypes(
                exclude="object"
            ).columns.tolist()

            categorical_columns = df.select_dtypes(
                include="object"
            ).columns.tolist()

            categorical_columns.remove(target_column)

            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")

            num_pipeline = Pipeline(

                steps=[
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(

                steps=[
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            preprocessor = ColumnTransformer(

                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]

            )

            return preprocessor

        except Exception as e:

            raise CustomException(e, sys)


    def initiate_data_transformation(self, train_path, test_path):

        try:

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and test data loaded")

            # Apply cleaning
            train_df = self.clean_data(train_df)
            test_df = self.clean_data(test_df)

            # Apply outlier capping
            train_df = self.cap_outliers(train_df)
            test_df = self.cap_outliers(test_df)

            target_column = "y"

            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]

            preprocessor = self.get_preprocessor(train_df)

            logging.info("Applying preprocessing")

            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            train_array = np.c_[X_train_transformed, y_train.to_numpy()]
            test_array = np.c_[X_test_transformed, y_test.to_numpy()]

            save_object(

                self.preprocessor_path,
                preprocessor

            )

            logging.info("Preprocessor saved")

            return train_array, test_array

        except Exception as e:

            raise CustomException(e, sys)