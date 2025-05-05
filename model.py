from sklearn.neural_network import MLPRegressor
import joblib
import pandas as pd
import os
from database import load_data

def train_model():
    conn = load_data()
    df = pd.read_sql_query("SELECT * FROM weather", conn)
    conn.close()

    features = [col for col in df.columns if col.startswith("day") and "date" not in col and "forecast" not in col]
    X = df[features]

    if "day10_morning" in df.columns:
        y_morning = df["day10_morning"]
        y_day = df["day10_day"]
        y_evening = df["day10_evening"]
    else:
        y_morning = df["day7_morning"]
        y_day = df["day7_day"]
        y_evening = df["day7_evening"]

    models = {}
    for target, filename in zip([y_morning, y_day, y_evening], ["morning_model.pkl", "day_model.pkl", "evening_model.pkl"]):
        model = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=2000, random_state=42)
        model.fit(X, target)
        joblib.dump(model, filename)
        models[filename] = model

    return models

def load_model():
    models = {}
    for name in ["morning_model.pkl", "day_model.pkl", "evening_model.pkl"]:
        if os.path.exists(name):
            models[name] = joblib.load(name)
    return models

def predict_days(models, input_data, n_days=10):
    expected_columns = [f"day{i}_{p}" for i in range(1, 11) for p in ["morning", "day", "evening"]]
    if len(input_data) != len(expected_columns):
        raise ValueError(f"Ожидалось {len(expected_columns)} значений, получено {len(input_data)}")

    df = pd.DataFrame([input_data], columns=expected_columns)
    forecast = []

    for i in range(n_days):
        df_copy = df.copy()
        cols = df.columns.tolist()
        shift = 3 * (i % 10)
        shifted = cols[shift:] + cols[:shift]
        df_copy = df_copy[shifted]
        df_copy.columns = cols

        morning = models["morning_model.pkl"].predict(df_copy)[0]
        day = models["day_model.pkl"].predict(df_copy)[0]
        evening = models["evening_model.pkl"].predict(df_copy)[0]

        forecast.extend([round(morning, 2), round(day, 2), round(evening, 2)])

    return forecast
