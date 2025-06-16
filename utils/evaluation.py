import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)

    return {
        "R2 Score": round(r2, 4),
        "MAE": round(mae, 4),
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
    }

def run_all_models(models: dict, X_train, X_test, y_train, y_test):
    results = []
    for name, model in models.items():
        try:
            scores = evaluate_model(model, X_train, X_test, y_train, y_test)
            scores["Model"] = name
            results.append(scores)
        except Exception as e:
            print(f"Error with model {name}: {e}")
    return pd.DataFrame(results).set_index("Model")
