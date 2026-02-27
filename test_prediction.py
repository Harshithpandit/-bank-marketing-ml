import pandas as pd
from src.pipeline.predict_pipeline import PredictPipeline

data = pd.read_csv("artifacts/test.csv")

data = data.drop("y", axis=1)

pipeline = PredictPipeline()

result = pipeline.predict(data)

print(result[:10])