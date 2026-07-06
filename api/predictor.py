import joblib
import pandas as pd
from src.logger import logger

from src.config import load_config


config = load_config()

# Load model and preprocessor only once when API starts
model = joblib.load(config["artifacts"]["model_path"])
preprocessor = joblib.load(config["artifacts"]["preprocessor_path"])


def predict_customer(customer_data):

    logger.info("Prediction request received")
    # Convert dictionary to DataFrame
    df = pd.DataFrame([customer_data])

    # Apply preprocessing
    transformed_data = preprocessor.transform(df)

    prediction = int(model.predict(transformed_data)[0])

    probability = float(model.predict_proba(transformed_data)[0].max())


    logger.info(
    f"Prediction: {prediction}, Probability: {probability}"
)

    return {
        "prediction": "Yes" if prediction == 1 else "No",
         "probability": round(probability, 4)
    }

