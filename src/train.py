import joblib
from src.logger import logger
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

import mlflow
import mlflow.sklearn

from src.preprocessing import preprocess_data
from src.config import load_config


def train_models():

    config = load_config()

    X_train, X_test, y_train, y_test = preprocess_data()

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

    best_model = None
    best_score = 0

    for name, model in models.items():

        with mlflow.start_run(run_name=name):

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(
                y_test,
                predictions,
                pos_label="Yes",
            )
            recall = recall_score(
                y_test,
                predictions,
                pos_label="Yes",
            )
            f1 = f1_score(
                y_test,
                predictions,
                pos_label="Yes",
            )

            mlflow.log_param("model", name)

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)

            mlflow.sklearn.log_model(
                sk_model=model,
                name="model"
            )

            print(f"\n{name}")
            print(f"Accuracy : {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall   : {recall:.4f}")
            print(f"F1 Score : {f1:.4f}")

            if f1 > best_score:
                best_score = f1
                best_model = model

    joblib.dump(
        best_model,
        config["artifacts"]["model_path"],
    )

    logger.info("Best model saved successfully.")

if __name__ == "__main__":
    train_models()