import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from xgboost import XGBClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from src.logger import logger
from src.preprocessing import preprocess_data
from src.config import load_config


def train_models():

    config = load_config()

    X_train, X_test, y_train, y_test = preprocess_data()

    models = {

        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            random_state=42
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            random_state=42
        ),

        "XGBoost": XGBClassifier(
            random_state=42,
            eval_metric="logloss"
        )

    }

    param_grids = {

        "Random Forest": {

            "n_estimators": [100, 200],
            "max_depth": [5, 10, None],
            "min_samples_split": [2, 5]

        },

        "XGBoost": {

            "n_estimators": [100, 200],
            "max_depth": [3, 5],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0]

        }

    }

    best_model = None
    best_f1 = 0

    mlflow.set_experiment("Customer Churn Prediction")

    for model_name, model in models.items():

        with mlflow.start_run(run_name=model_name):

            if model_name in param_grids:

                print(f"\nRunning GridSearchCV for {model_name}...")

                grid = GridSearchCV(
                    estimator=model,
                    param_grid=param_grids[model_name],
                    cv=3,
                    scoring="f1",
                    n_jobs=-1
                )

                grid.fit(X_train, y_train)

                model = grid.best_estimator_

                mlflow.log_params(grid.best_params_)

            else:

                model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)

            precision = precision_score(y_test, y_pred)

            recall = recall_score(y_test, y_pred)

            f1 = f1_score(y_test, y_pred)

            y_prob = model.predict_proba(X_test)[:, 1]

            roc_auc = roc_auc_score(y_test, y_prob)

            mlflow.log_param("model", model_name)

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("roc_auc", roc_auc)

            if model_name == "XGBoost":
                mlflow.xgboost.log_model(
                    xgb_model=model,
                    name="model"
                )
            else:
                mlflow.sklearn.log_model(
                    sk_model=model,
                    name="model"
    )

            print("=" * 50)
            print(model_name)
            print("=" * 50)
            print(f"Accuracy : {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall   : {recall:.4f}")
            print(f"F1 Score : {f1:.4f}")
            print(f"ROC-AUC  : {roc_auc:.4f}")
            
            best_roc_auc = 0
            if roc_auc > best_roc_auc:
                best_roc_auc = roc_auc
                best_model = model

    joblib.dump(
        best_model,
        config["artifacts"]["model_path"],
    )

    logger.info("Best model saved successfully.")


if __name__ == "__main__":
    train_models()