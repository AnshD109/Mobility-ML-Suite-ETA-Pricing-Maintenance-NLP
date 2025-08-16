import random, time
import requests

URL = "http://127.0.0.1:8000/predict"
N = 100

for i in range(N):
    hour = random.choice([8, 9, 12, 17, 18, 21])
    dow = random.randint(0,6)
    dist = max(0.8, random.gauss(5, 2))
    demand = max(0.5, random.gauss(1.2 if hour in [8,9,17,18] else 0.9, 0.2))
    supply = max(0.4, random.gauss(0.8 if hour in [8,9,17,18] else 1.1, 0.2))
    city = random.choice(["Berlin","Munich"])
    body = {
        "distance_km": round(dist, 2),
        "hour": hour,
        "day_of_week": dow,
        "demand_index": round(demand, 2),
        "supply_index": round(supply, 2),
        "city": city
    }
    r = requests.post(URL, json=body, timeout=5)
    print(i, r.status_code)
    time.sleep(0.03)
