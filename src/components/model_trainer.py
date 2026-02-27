import os
import sys
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from imblearn.over_sampling import SMOTE

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


class ModelTrainer:

    def __init__(self):

        self.model_path = os.path.join("artifacts", "model.pkl")


    def evaluate_model(self, y_true, y_pred, y_proba):

        try:

            # Convert to pandas Series
            y_true = pd.Series(y_true)
            y_pred = pd.Series(y_pred)

            accuracy = accuracy_score(y_true, y_pred)

            f1 = f1_score(
                y_true,
                y_pred
            )

            roc_auc = roc_auc_score(
                y_true,
                y_proba
            )

            return accuracy, f1, roc_auc

        except Exception as e:

            raise CustomException(e, sys)


    def initiate_model_training(self, train_array, test_array):

        try:

            logging.info("Splitting train and test arrays")

            X_train, y_train = (
                train_array[:, :-1],
                train_array[:, -1]
            )

            X_test, y_test = (
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Convert to pandas Series
            y_train = pd.Series(y_train)
            y_test = pd.Series(y_test)

            # Convert target from yes/no → 1/0
            y_train = y_train.map({'no': 0, 'yes': 1})
            y_test = y_test.map({'no': 0, 'yes': 1})

            logging.info("Target converted to numeric")

            # Apply SMOTE
            logging.info("Applying SMOTE")

            smote = SMOTE(random_state=42)

            X_train_res, y_train_res = smote.fit_resample(
                X_train,
                y_train
            )

            logging.info("SMOTE applied successfully")

            # Define models
            models = {

                "Logistic Regression":
                LogisticRegression(
                    max_iter=500,
                    class_weight='balanced',
                    random_state=42
                ),

                "Random Forest":
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=12,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    class_weight='balanced',
                    random_state=42
                ),

                "XGBoost":
                XGBClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=6,
                    random_state=42,
                    eval_metric='logloss'
                )
            }

            best_model = None
            best_score = -1
            best_model_name = ""

            # Train and evaluate models
            for name, model in models.items():

                logging.info(f"Training {name}")

                model.fit(X_train_res, y_train_res)

                y_pred = model.predict(X_test)

                y_proba = model.predict_proba(X_test)[:, 1]

                accuracy, f1, roc_auc = self.evaluate_model(
                    y_test,
                    y_pred,
                    y_proba
                )

                logging.info(
                    f"{name} → Accuracy: {accuracy:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}"
                )

                # Select best model based on ROC-AUC
                if roc_auc > best_score:

                    best_score = roc_auc
                    best_model = model
                    best_model_name = name

            logging.info(f"Best model selected: {best_model_name}")
            logging.info(f"Best ROC-AUC Score: {best_score:.4f}")

            # Save best model
            save_object(
                self.model_path,
                best_model
            )

            logging.info("Model saved successfully")

            return best_model

        except Exception as e:

            raise CustomException(e, sys)