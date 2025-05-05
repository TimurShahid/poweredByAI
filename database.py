import sqlite3

def init_db():
    conn = sqlite3.connect("weather_data.db")
    cursor = conn.cursor()
    date_fields = ", ".join([f"day{i}_date TEXT" for i in range(1, 8)])
    temp_fields = ", ".join([f"day{i}_{p} REAL" for i in range(1, 8) for p in ["morning", "day", "evening"]])
    forecast_fields = ", ".join([f"forecast_day{i}_{p} REAL" for i in range(1, 8) for p in ["morning", "day", "evening"]])
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS weather (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            {date_fields},
            {temp_fields},
            target_temp REAL,
            {forecast_fields}
        )
    """)
    conn.commit()
    conn.close()

def insert_record(dates, temps, target):
    conn = sqlite3.connect("weather_data.db")
    cursor = conn.cursor()
    fields = [f"day{i}_date" for i in range(1, 8)] + \
             [f"day{i}_{p}" for i in range(1, 8) for p in ["morning", "day", "evening"]] + \
             ["target_temp"] + \
             [f"forecast_day{i}_{p}" for i in range(1, 8) for p in ["morning", "day", "evening"]]
    values = dates + temps + [target] + [None]*21
    cursor.execute(f"INSERT INTO weather ({', '.join(fields)}) VALUES ({', '.join(['?']*len(values))})", values)
    conn.commit()
    conn.close()

def load_data():
    return sqlite3.connect("weather_data.db")

def get_all_records():
    conn = sqlite3.connect("weather_data.db")
    conn.row_factory = sqlite3.Row  
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM weather ORDER BY id DESC")
    records = cursor.fetchall()
    conn.close()
    return [dict(row) for row in records]

def delete_record_by_id(record_id):
    conn = sqlite3.connect("weather_data.db")
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM weather WHERE id = ?", (record_id,))
    conn.commit()

    cursor.execute("SELECT COUNT(*) FROM weather")
    count = cursor.fetchone()[0]

    if count == 0:
        cursor.execute("DELETE FROM sqlite_sequence WHERE name='weather'")  # сброс ID
        conn.commit()

    conn.close()
