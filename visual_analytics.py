import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget, QLabel
from PyQt5.QtCore import QDate
import sqlite3
import pandas as pd
import numpy as np

class ForecastGraph(QWidget):
    def __init__(self, forecast, start_date):
        super().__init__()
        self.setWindowTitle("График прогноза на неделю")
        self.setGeometry(200, 200, 800, 400)
        layout = QVBoxLayout()

        canvas = self.plot_forecast(forecast, start_date)
        layout.addWidget(canvas)
        self.setLayout(layout)

    def plot_forecast(self, forecast, start_date):
        days = [start_date.addDays(i).toString("dd.MM") for i in range(7)]
        morning = [forecast[i*3] for i in range(7)]
        day = [forecast[i*3+1] for i in range(7)]
        evening = [forecast[i*3+2] for i in range(7)]

        fig, ax = plt.subplots()
        ax.plot(days, morning, label="Утро")
        ax.plot(days, day, label="День")
        ax.plot(days, evening, label="Вечер")
        ax.set_title("Прогноз температуры на неделю")
        ax.set_ylabel("Температура °C")
        ax.legend()
        return FigureCanvas(fig)


class ForecastHistoryTable(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("История прогнозов")
        self.setGeometry(300, 300, 1000, 400)
        layout = QVBoxLayout()

        self.table = QTableWidget()
        layout.addWidget(self.table)

        self.load_data()
        self.setLayout(layout)

    def load_data(self):
        conn = sqlite3.connect("weather_data.db")
        df = pd.read_sql_query("SELECT * FROM weather ORDER BY id DESC", conn)
        conn.close()

        self.table.setColumnCount(len(df.columns))
        self.table.setRowCount(len(df))
        self.table.setHorizontalHeaderLabels(df.columns)

        for i in range(len(df)):
            for j in range(len(df.columns)):
                self.table.setItem(i, j, QTableWidgetItem(str(df.iloc[i, j])))


class TemperatureCalendar(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Температурный календарь")
        self.setGeometry(300, 300, 600, 400)
        layout = QVBoxLayout()

        label = QLabel("Температурная карта по датам (утро)")
        layout.addWidget(label)

        self.load_heatmap()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def load_heatmap(self):
        conn = sqlite3.connect("weather_data.db")
        df = pd.read_sql_query("SELECT day1_date, day1_morning FROM weather", conn)
        conn.close()

        if df.empty:
            self.canvas = QLabel("Нет данных для отображения")
            return

        df["day1_date"] = pd.to_datetime(df["day1_date"], dayfirst=True)
        df.sort_values("day1_date", inplace=True)
        df.set_index("day1_date", inplace=True)

        dates = df.index.strftime("%d.%m")
        temps = df["day1_morning"].values

        fig, ax = plt.subplots()
        ax.bar(dates, temps, color=plt.cm.coolwarm((temps - np.min(temps)) / (np.max(temps) - np.min(temps))))
        ax.set_title("Температура утром по датам")
        ax.set_ylabel("°C")
        plt.xticks(rotation=45)
        self.canvas = FigureCanvas(fig)