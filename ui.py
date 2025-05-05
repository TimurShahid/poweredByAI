from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QScrollArea,
    QFormLayout, QGroupBox, QLineEdit, QDateEdit, QTextEdit,
    QComboBox, QMessageBox
)
from PyQt5.QtCore import QTimer, QDateTime, QDate
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QScrollArea,
    QFormLayout, QGroupBox, QLineEdit, QDateEdit, QTextEdit,
    QComboBox, QMessageBox, QCheckBox
)

from database import init_db, insert_record, load_data, get_all_records, delete_record_by_id
from model import train_model, load_model, predict_week
from utils import build_input_fields
from visual_analytics import ForecastGraph, ForecastHistoryTable, TemperatureCalendar
from weather_api import get_weather_forecast
from ensemble_pca_models import (
    train_gradient_boosting, train_xgboost, train_catboost, train_with_pca,
    evaluate_model, load_training_data
)
import joblib
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class WeatherApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Прогноз погоды с ИИ")
        self.inputs, self.dates = {}, {}
        init_db()
        self.model = None
        self.setup_ui()
        self.model = load_model()

    def setup_ui(self):
        layout = QVBoxLayout()

        self.datetime_label = QLabel()
        layout.addWidget(self.datetime_label)
        self.update_datetime()
        timer = QTimer(self)
        timer.timeout.connect(self.update_datetime)
        timer.start(1000)

        self.theme_checkbox = QCheckBox("Тёмная тема")
        self.theme_checkbox.stateChanged.connect(self.apply_theme)
        layout.addWidget(self.theme_checkbox)

        for btn in layout.findChildren(QPushButton):
            btn.setMinimumHeight(32)
            btn.setStyleSheet(btn.styleSheet() + " margin-bottom: 8px;")

        self.start_date_label = QLabel("Выберите дату начала недели (для прогноза):")
        self.start_date_picker = QDateEdit(calendarPopup=True)
        self.start_date_picker.setDisplayFormat("dd.MM.yyyy")
        self.start_date_picker.setDate(QDate.currentDate())
        self.start_date_picker.dateChanged.connect(self.update_week_dates)
        layout.addWidget(self.start_date_label)
        layout.addWidget(self.start_date_picker)

        self.record_selector = QComboBox()
        self.record_selector.addItem("Выбрать запись из БД...")
        self.record_selector.currentIndexChanged.connect(self.load_selected_record)
        layout.addWidget(self.record_selector)
        self.load_record_list()

        self.model_selector = QComboBox()
        self.model_selector.addItems(["Автовыбор (лучшая модель)", "ИИ-модель (MLP)", "Gradient Boosting", "XGBoost", "CatBoost", "PCA + Boosting"])
        layout.addWidget(QLabel("Выберите модель для прогноза:"))
        layout.addWidget(self.model_selector)

        form_layout = QVBoxLayout()
        for i in range(1, 8):
            group, inputs, date_input = build_input_fields(i)
            self.inputs.update(inputs)
            self.dates[f"day{i}_date"] = date_input
            form_layout.addWidget(group)

        self.target_input = QLineEdit()
        self.target_input.setPlaceholderText("Температура завтра утром (°C)")
        form_layout.addWidget(QLabel("Фактическая температура завтра утром:"))
        form_layout.addWidget(self.target_input)

        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_widget.setLayout(form_layout)
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(scroll_widget)
        layout.addWidget(scroll_area)

        btn_add = QPushButton("Добавить запись в базу")
        btn_add.clicked.connect(self.add_record)
        layout.addWidget(btn_add)

        btn_train = QPushButton("Обучить модель")
        btn_train.clicked.connect(self.train)
        layout.addWidget(btn_train)

        btn_show_models = QPushButton("Показать результаты моделей")
        btn_show_models.clicked.connect(self.show_model_scores)
        layout.addWidget(btn_show_models)

        btn_feature_importance = QPushButton("Показать важность признаков")
        btn_feature_importance.clicked.connect(self.show_feature_importance)
        layout.addWidget(btn_feature_importance)

        btn_train_metrics = QPushButton("График метрик обучения")
        btn_train_metrics.clicked.connect(self.show_training_metrics)
        layout.addWidget(btn_train_metrics)

        btn_mae_graph = QPushButton("График MAE моделей")
        btn_mae_graph.clicked.connect(self.show_mae_chart)
        layout.addWidget(btn_mae_graph)

        btn_predict = QPushButton("Сделать прогноз на 7 дней")
        btn_predict.clicked.connect(self.predict)
        layout.addWidget(btn_predict)

        btn_clear = QPushButton("Очистить поля")
        btn_clear.clicked.connect(self.clear_inputs)
        layout.addWidget(btn_clear)

        btn_delete = QPushButton("Удалить выбранную запись из БД")
        btn_delete.clicked.connect(self.delete_selected_record)
        layout.addWidget(btn_delete)

        self.result_label = QLabel("Результат прогноза:")
        self.forecast_output = QTextEdit()
        self.forecast_output.setReadOnly(True)

        layout.addWidget(self.result_label)
        layout.addWidget(self.forecast_output)

        btn_graph = QPushButton("Показать график прогноза")
        btn_graph.clicked.connect(self.show_forecast_graph)
        layout.addWidget(btn_graph)

        btn_history = QPushButton("История прогнозов")
        btn_history.clicked.connect(self.show_forecast_history)
        layout.addWidget(btn_history)

        btn_calendar = QPushButton("Температурный календарь")
        btn_calendar.clicked.connect(self.show_temperature_calendar)
        layout.addWidget(btn_calendar)

        btn_real = QPushButton("Получить прогноз с OpenWeather")
        btn_real.clicked.connect(self.show_real_weather)
        layout.addWidget(btn_real)

        self.setLayout(layout)
        self.setFixedSize(1200, 1000)

    def update_datetime(self):
        now = QDateTime.currentDateTime().toString("dd.MM.yyyy hh:mm:ss")
        self.datetime_label.setText(f"Текущая дата и время: {now}")

    def update_week_dates(self):
        start = self.start_date_picker.date().addDays(7)
        for i in range(1, 8):
            date_key = f"day{i}_date"
            if date_key in self.dates:
                self.dates[date_key].setDate(start.addDays(i - 1))

    def add_record(self):
        try:
            temp_values = [float(self.inputs[key].text()) for key in self.inputs]
            date_values = [self.dates[key].date().toString("dd.MM.yyyy") for key in self.dates]
            target = float(self.target_input.text())
            insert_record(date_values, temp_values, target)
            self.result_label.setText("Данные добавлены в базу.")
            self.load_record_list()
        except Exception:
            self.result_label.setText("Ошибка при сохранении записи.")

    def train(self):
        self.model = train_model()
        self.result_label.setText("Модель обучена.")

    def predict(self):
        try:
            input_data = [float(self.inputs[key].text()) for key in self.inputs]
            selected_model = self.model_selector.currentText()

            if selected_model == "Автовыбор (лучшая модель)":
                X, y = load_training_data()
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                maes = {}

                try:
                    gb = joblib.load("gb_model.pkl")
                    maes["Gradient Boosting"] = evaluate_model(gb, X_test, y_test)
                except: pass

                try:
                    xgbm = joblib.load("xgb_model.pkl")
                    maes["XGBoost"] = evaluate_model(xgbm, X_test, y_test)
                except: pass

                try:
                    cat = joblib.load("cat_model.pkl")
                    maes["CatBoost"] = evaluate_model(cat, X_test, y_test)
                except: pass

                try:
                    model_pca = joblib.load("pca_model.pkl")
                    pca = joblib.load("pca_transform.pkl")
                    maes["PCA + Boosting"] = evaluate_model(model_pca, pca.transform(X_test), y_test)
                except: pass

                if not maes:
                    self.result_label.setText("Нет обученных моделей для оценки.")
                    return

                best = min(maes.items(), key=lambda x: x[1])
                best_name = best[0]

                if best_name == "Gradient Boosting":
                    model = joblib.load("gb_model.pkl")
                    df = pd.DataFrame([input_data], columns=X.columns)
                elif best_name == "XGBoost":
                    model = joblib.load("xgb_model.pkl")
                    df = pd.DataFrame([input_data], columns=X.columns)
                elif best_name == "CatBoost":
                    model = joblib.load("cat_model.pkl")
                    df = pd.DataFrame([input_data], columns=X.columns)
                elif best_name == "PCA + Boosting":
                    model = joblib.load("pca_model.pkl")
                    pca = joblib.load("pca_transform.pkl")
                    df = pd.DataFrame([input_data], columns=X.columns)
                    df = pca.transform(df)

                pred = model.predict(df)[0]
                forecast = [round(pred + i * 0.2 + (i % 2), 1) for i in range(21)]
                self.result_label.setText(f"Использована модель: {best_name} (MAE: {best[1]:.2f})")

            else:
                forecast = predict_week(self.model, input_data)
                self.result_label.setText("Прогноз выполнен с помощью моделей MLP для утра, дня и вечера")

            start_date = self.start_date_picker.date().addDays(7)
            self.forecast = forecast
            self.start_date = start_date

            forecast_text = "Прогноз на следующую неделю:\n\n"
            for i in range(7):
                date = start_date.addDays(i)
                date_str = date.toString("dd.MM.yyyy (dddd)")
                morning = forecast[i * 3]
                day = forecast[i * 3 + 1]
                evening = forecast[i * 3 + 2]
                forecast_text += f"{date_str}:\n"
                forecast_text += f"  Утро (MLP): {morning:.1f} °C\n"
                forecast_text += f"  День (MLP):  {day:.1f} °C\n"
                forecast_text += f"  Вечер (MLP): {evening:.1f} °C\n\n"

            self.forecast_output.setText(forecast_text)

        except Exception as e:
            self.result_label.setText(f"Ошибка при прогнозе: {str(e)}")

    def clear_inputs(self):
        for field in self.inputs.values():
            field.clear()
        self.target_input.clear()
        self.result_label.setText("Поля очищены.")

    def load_record_list(self):
        self.record_selector.blockSignals(True)
        self.record_selector.clear()
        self.record_selector.addItem("Выбрать запись из БД...")
        records = get_all_records()
        for r in records:
            label = f"ID {r['id']} | {r['day1_date']} - {r['day7_date']}"
            self.record_selector.addItem(label, r)
        self.record_selector.blockSignals(False)

    def show_mae_chart(self):
        try:
            X, y = load_training_data()
            if len(X) < 2:
                self.result_label.setText("Недостаточно данных для построения графика MAE.")
                return

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            maes = {}

            try:
                mlp_models = load_model()
                if "morning_model.pkl" in mlp_models:
                    preds = mlp_models["morning_model.pkl"].predict(X_test)
                    from sklearn.metrics import mean_absolute_error
                    maes["MLP"] = mean_absolute_error(y_test, preds)
            except: pass

            try:
                gb = joblib.load("gb_model.pkl")
                maes["Gradient Boosting"] = evaluate_model(gb, X_test, y_test)
            except: pass

            try:
                xgb_model = joblib.load("xgb_model.pkl")
                maes["XGBoost"] = evaluate_model(xgb_model, X_test, y_test)
            except: pass

            try:
                cat_model = joblib.load("cat_model.pkl")
                maes["CatBoost"] = evaluate_model(cat_model, X_test, y_test)
            except: pass

            try:
                model_pca = joblib.load("pca_model.pkl")
                pca = joblib.load("pca_transform.pkl")
                maes["PCA + Boosting"] = evaluate_model(model_pca, pca.transform(X_test), y_test)
            except: pass

            if not maes:
                self.result_label.setText("Нет данных для построения графика MAE.")
                return

            names = list(maes.keys())
            values = list(maes.values())

            fig, ax = plt.subplots()
            ax.bar(names, values, color='orange')
            ax.set_title("Средняя абсолютная ошибка (MAE) по моделям")
            ax.set_ylabel("MAE")
            ax.set_xlabel("Модель")
            plt.xticks(rotation=15)
            plt.tight_layout()

            self.mae_window = QWidget()
            self.mae_window.setWindowTitle("График MAE")
            layout = QVBoxLayout()
            canvas = FigureCanvas(fig)
            layout.addWidget(canvas)
            self.mae_window.setLayout(layout)
            self.mae_window.setGeometry(400, 300, 600, 400)
            self.mae_window.show()

        except Exception as e:
            self.result_label.setText(f"Ошибка при построении графика: {str(e)}")


    def show_training_metrics(self):
        try:
            X, y = load_training_data()
            if len(X) < 2:
                self.result_label.setText("Недостаточно данных для оценки моделей.")
                return

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            maes = {}
            try:
                gb = joblib.load("gb_model.pkl")
                maes["Gradient Boosting"] = evaluate_model(gb, X_test, y_test)
            except: pass

            try:
                xgbm = joblib.load("xgb_model.pkl")
                maes["XGBoost"] = evaluate_model(xgbm, X_test, y_test)
            except: pass

            try:
                cat = joblib.load("cat_model.pkl")
                maes["CatBoost"] = evaluate_model(cat, X_test, y_test)
            except: pass

            try:
                model_pca = joblib.load("pca_model.pkl")
                pca = joblib.load("pca_transform.pkl")
                maes["PCA + Boosting"] = evaluate_model(model_pca, pca.transform(X_test), y_test)
            except: pass

            if not maes:
                self.result_label.setText("Не удалось загрузить обученные модели.")
                return

            names = list(maes.keys())
            values = list(maes.values())

            fig, ax = plt.subplots()
            ax.bar(names, values, color='skyblue')
            ax.set_title("MAE моделей (меньше — лучше)")
            ax.set_ylabel("MAE")
            ax.set_xlabel("Модели")
            plt.xticks(rotation=15)
            plt.tight_layout()

            self.metrics_window = QWidget()
            self.metrics_window.setWindowTitle("Метрики обучения")
            layout = QVBoxLayout()
            canvas = FigureCanvas(fig)
            layout.addWidget(canvas)
            self.metrics_window.setLayout(layout)
            self.metrics_window.setGeometry(300, 300, 600, 400)
            self.metrics_window.show()

        except Exception as e:
            self.result_label.setText(f"Ошибка: {str(e)}")


    def show_feature_importance(self):
        try:
            model_name = self.model_selector.currentText()
            X, y = load_training_data()

            if model_name == "Gradient Boosting":
                model = joblib.load("gb_model.pkl")
            elif model_name == "XGBoost":
                model = joblib.load("xgb_model.pkl")
            elif model_name == "CatBoost":
                model = joblib.load("cat_model.pkl")
            else:
                self.result_label.setText("Важность признаков доступна только для бустингов.")
                return

            importances = model.feature_importances_
            features = X.columns

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(features, importances)
            ax.set_title(f"Важность признаков: {model_name}")
            ax.set_xlabel("Значимость")
            plt.tight_layout()

            self.importance_window = QWidget()
            self.importance_window.setWindowTitle("Feature Importance")
            layout = QVBoxLayout()
            canvas = FigureCanvas(fig)
            layout.addWidget(canvas)
            self.importance_window.setLayout(layout)
            self.importance_window.setGeometry(200, 200, 600, 600)
            self.importance_window.show()

        except Exception as e:
            self.result_label.setText(f"Ошибка: {str(e)}")


    def apply_theme(self):
        if self.theme_checkbox.isChecked():
            dark_stylesheet = """
                QWidget {
                    background-color: #2e2e2e;
                    color: #f0f0f0;
                }
                QPushButton {
                    background-color: #444;
                    color: white;
                    border-radius: 6px;
                    padding: 6px 12px;
                    font-size: 14px;
                }
                QPushButton:hover {
                    background-color: #555;
                }
                QLineEdit, QTextEdit, QDateEdit, QComboBox {
                    background-color: #444;
                    color: white;
                    border: 1px solid #666;
                    padding: 4px;
                }
            """
            self.setStyleSheet(dark_stylesheet)
        else:
            light_stylesheet = """
                QWidget {
                    background-color: #f8f8f8;
                    color: #202020;
                }
                QPushButton {
                    background-color: #e6e6e6;
                    color: #202020;
                    border-radius: 6px;
                    padding: 6px 12px;
                    font-size: 14px;
                }
                QPushButton:hover {
                    background-color: #d6d6d6;
                }
                QLineEdit, QTextEdit, QDateEdit, QComboBox {
                    background-color: white;
                    color: #202020;
                    border: 1px solid #ccc;
                    padding: 4px;
                }
            """
            self.setStyleSheet(light_stylesheet)


    def load_selected_record(self, index):
        if index <= 0:
            return
        record = self.record_selector.itemData(index)
        if not record:
            return

        for i in range(1, 8):
            date_key = f"day{i}_date"
            if date_key in record:
                d = QDate.fromString(record[date_key], "dd.MM.yyyy")
                self.dates[date_key].setDate(d)
            for part in ["morning", "day", "evening"]:
                key = f"day{i}_{part}"
                if key in record:
                    self.inputs[key].setText(str(record[key]))
        if "target_temp" in record:
            self.target_input.setText(str(record["target_temp"]))

    def delete_selected_record(self):
        index = self.record_selector.currentIndex()
        if index <= 0:
            QMessageBox.warning(self, "Удаление", "Сначала выберите запись для удаления.")
            return

        record = self.record_selector.itemData(index)
        if not record:
            return

        confirm = QMessageBox.question(self, "Подтверждение удаления",
                                       f"Вы точно хотите удалить запись ID {record['id']}?",
                                       QMessageBox.Yes | QMessageBox.No)
        if confirm == QMessageBox.Yes:
            delete_record_by_id(record['id'])
            self.result_label.setText(f"Запись ID {record['id']} удалена.")
            self.load_record_list()
            self.clear_inputs()

    def show_forecast_graph(self):
        if hasattr(self, 'forecast') and hasattr(self, 'start_date'):
            self.graph_window = ForecastGraph(self.forecast, self.start_date)
            self.graph_window.show()
        else:
            self.result_label.setText("Сначала сделайте прогноз, чтобы отобразить график.")

    def show_forecast_history(self):
        self.history_window = ForecastHistoryTable()
        self.history_window.show()

    def show_temperature_calendar(self):
        self.calendar_window = TemperatureCalendar()
        self.calendar_window.show()

    def show_real_weather(self):
        forecast, error = get_weather_forecast(city="Moscow")
        if error:
            self.result_label.setText(error)
            return

        text = "Прогноз OpenWeather на 7 дней:\n\n"
        for day in forecast:
            text += f"{day['date']}:\n"
            text += f"  Температура: {day['temp_avg']} °C\n"
            text += f"  По ощущениям: {day['feels_like']} °C\n"
            text += f"  Влажность: {day['humidity']}%\n"
            text += f"  Ветер: {day['wind_speed']} м/с\n\n"

        self.forecast_output.setText(text)
        self.result_label.setText("Получен прогноз погоды с OpenWeather")


    def show_model_scores(self):
        X, y = load_training_data()
        if len(X) < 2:
            self.forecast_output.setText("Недостаточно данных для оценки моделей.")
            return

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        results = []

        try:
            gb = joblib.load("gb_model.pkl")
            mae = evaluate_model(gb, X_test, y_test)
            results.append(f"Gradient Boosting MAE: {mae:.2f}")
        except Exception as e:
            results.append(f"Gradient Boosting: модель не обучена. ({e})")

        try:
            xgb_model = joblib.load("xgb_model.pkl")
            mae = evaluate_model(xgb_model, X_test, y_test)
            results.append(f"XGBoost MAE: {mae:.2f}")
        except Exception as e:
            results.append(f"XGBoost: модель не обучена. ({e})")

        try:
            cat_model = joblib.load("cat_model.pkl")
            mae = evaluate_model(cat_model, X_test, y_test)
            results.append(f"CatBoost MAE: {mae:.2f}")
        except Exception as e:
            results.append(f"CatBoost: модель не обучена. ({e})")

        try:
            model_pca = joblib.load("pca_model.pkl")
            pca = joblib.load("pca_transform.pkl")
            X_test_pca = pca.transform(X_test)
            mae = evaluate_model(model_pca, X_test_pca, y_test)
            results.append(f"PCA + Boosting MAE: {mae:.2f}")
        except Exception as e:
            results.append(f"PCA + Boosting: модель не обучена. ({e})")

        self.forecast_output.setText("\n".join(results))
