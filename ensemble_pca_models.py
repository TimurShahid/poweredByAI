from sklearn.ensemble import GradientBoostingRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import catboost as cb
import pandas as pd
import joblib
import sqlite3
import os


def load_training_data():
    import sqlite3
    conn = sqlite3.connect("weather_data.db")
    df = pd.read_sql_query("SELECT * FROM weather", conn)
    conn.close()

    features = [col for col in df.columns if col.startswith("day") and "date" not in col and "forecast" not in col]
    X = df[features]

    # Выбираем целевую переменную динамически
    if "day10_morning" in df.columns:
        y = df["day10_morning"]
    else:
        y = df["day7_morning"]

    return X, y


def train_gradient_boosting():
    X, y = load_training_data()
    model = GradientBoostingRegressor()
    model.fit(X, y)
    joblib.dump(model, "gb_model.pkl")
    return model


def train_xgboost():
    X, y = load_training_data()
    model = xgb.XGBRegressor(objective="reg:squarederror")
    model.fit(X, y)
    joblib.dump(model, "xgb_model.pkl")
    return model


def train_catboost():
    X, y = load_training_data()
    model = cb.CatBoostRegressor(verbose=0)
    model.fit(X, y)
    joblib.dump(model, "cat_model.pkl")
    return model


def train_with_pca(n_components=10):
    X, y = load_training_data()
    max_components = min(len(X), X.shape[1])  
    n_components = min(n_components, max_components)

    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)

    model = GradientBoostingRegressor()
    model.fit(X_reduced, y)

    joblib.dump(model, "pca_model.pkl")
    joblib.dump(pca, "pca_transform.pkl")
    return model, pca



def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    return mean_absolute_error(y_test, preds)


if __name__ == "__main__":
    X, y = load_training_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\nGradient Boosting:")
    gb = train_gradient_boosting()
    print("MAE:", evaluate_model(gb, X_test, y_test))

    print("\nXGBoost:")
    xgbm = train_xgboost()
    print("MAE:", evaluate_model(xgbm, X_test, y_test))

    print("\nCatBoost:")
    cat = train_catboost()
    print("MAE:", evaluate_model(cat, X_test, y_test))

    print("\nGradient Boosting with PCA:")
    model_pca, pca = train_with_pca()
    X_test_pca = pca.transform(X_test)
    print("MAE:", evaluate_model(model_pca, X_test_pca, y_test))
