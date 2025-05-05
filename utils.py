from PyQt5.QtWidgets import QGroupBox, QFormLayout, QLineEdit, QDateEdit
from PyQt5.QtCore import QDate

def build_input_fields(day_index):
    group = QGroupBox(f"День {day_index}")
    layout = QFormLayout()
    inputs = {}

    date_input = QDateEdit()
    date_input.setDisplayFormat("dd.MM.yyyy")
    date_input.setDate(QDate.currentDate())
    layout.addRow("Дата:", date_input)

    for period in ["morning", "day", "evening"]:
        key = f"day{day_index}_{period}"
        line = QLineEdit()
        line.setPlaceholderText(f"Температура {period} (°C)")
        label = {"morning": "Утро", "day": "День", "evening": "Вечер"}[period]
        layout.addRow(label, line)
        inputs[key] = line

    group.setLayout(layout)
    return group, inputs, date_input
