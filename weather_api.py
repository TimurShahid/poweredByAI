import requests
from datetime import datetime

API_KEY = "e4018c8dbfb3e5b513f63dfab814dc37"
BASE_URL = "https://api.openweathermap.org/data/2.5/forecast"


def get_weather_forecast(city="Moscow", days=7):
    params = {
        "q": city,
        "appid": API_KEY,
        "units": "metric",
        "cnt": days * 8
    }
    response = requests.get(BASE_URL, params=params)
    if response.status_code != 200:
        return None, f"Ошибка API: {response.status_code}"

    data = response.json()
    forecast = {}

    for item in data["list"]:
        dt_txt = item["dt_txt"]
        date_str = dt_txt.split(" ")[0]
        time_str = dt_txt.split(" ")[1][:5]

        temp = item["main"]["temp"]
        feels_like = item["main"]["feels_like"]
        humidity = item["main"]["humidity"]
        wind = item["wind"]["speed"]

        if date_str not in forecast:
            forecast[date_str] = {"temps": [], "feels": [], "humidity": [], "wind": []}

        forecast[date_str]["temps"].append(temp)
        forecast[date_str]["feels"].append(feels_like)
        forecast[date_str]["humidity"].append(humidity)
        forecast[date_str]["wind"].append(wind)

    processed = []
    for date in sorted(forecast.keys())[:days]:
        temps = forecast[date]["temps"]
        feels = forecast[date]["feels"]
        hums = forecast[date]["humidity"]
        winds = forecast[date]["wind"]

        avg = lambda lst: round(sum(lst) / len(lst), 1) if lst else 0.0

        processed.append({
            "date": datetime.strptime(date, "%Y-%m-%d").strftime("%d.%m.%Y"),
            "temp_avg": avg(temps),
            "feels_like": avg(feels),
            "humidity": avg(hums),
            "wind_speed": avg(winds),
        })

    return processed, None
