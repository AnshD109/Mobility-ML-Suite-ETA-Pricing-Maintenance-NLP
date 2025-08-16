
import random
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

LABELS = ["ETA_delay","Pricing","Vehicle_issue","Driver_behavior","App_bug","Support"]

ETA_TEMPLATES = [
    "My ride arrived {min} minutes late in {city}. ETA was wrong.",
    "Driver took a long route; massive delay to the destination.",
    "Pickup was late and the app kept changing ETA.",
    "The estimated time to arrival was off by more than {min} minutes."
]
PRICING_TEMPLATES = [
    "I was overcharged for a {km} km trip; fare seems wrong.",
    "Why is the surge so high? Price unexpectedly expensive.",
    "Promo code did not apply; charged full price.",
    "Receipt shows extra waiting time I didn't have."
]
VEHICLE_TEMPLATES = [
    "The car AC was not working and engine temperature seemed high.",
    "Vehicle made strange vibration and noise during ride.",
    "Battery warning light turned on while driving.",
    "Check engine light appeared; car needs maintenance."
]
DRIVER_TEMPLATES = [
    "Driver was speeding and driving aggressively.",
    "Unprofessional behavior; driver talked on the phone while driving.",
    "Driver missed the pickup point and didn't follow the route.",
    "Rude attitude from the driver; please take action."
]
APP_TEMPLATES = [
    "App kept crashing at checkout on Android.",
    "Could not add payment method; app bug after update.",
    "Map did not load and booking failed.",
    "Notification stuck; app shows spinner forever."
]
SUPPORT_TEMPLATES = [
    "Need help changing my booking time.",
    "How do I contact support about a lost item?",
    "Please update my email on the account.",
    "I want a refund but the help center link is broken."
]

def generate_csv(path: str, n_rows: int = 300, seed: int = 42):
    rng = random.Random(seed)
    cities = ["Berlin","Munich","Hamburg","Cologne"]
    channels = ["app","email","phone"]
    rows = []
    now = datetime.utcnow()
    for i in range(n_rows):
        label = rng.choice(LABELS)
        if label == "ETA_delay":
            text = rng.choice(ETA_TEMPLATES).format(min=rng.randint(10,40), city=rng.choice(cities))
        elif label == "Pricing":
            text = rng.choice(PRICING_TEMPLATES).format(km=rng.randint(3,15))
        elif label == "Vehicle_issue":
            text = rng.choice(VEHICLE_TEMPLATES)
        elif label == "Driver_behavior":
            text = rng.choice(DRIVER_TEMPLATES)
        elif label == "App_bug":
            text = rng.choice(APP_TEMPLATES)
        else:
            text = rng.choice(SUPPORT_TEMPLATES)
        ts = now - timedelta(minutes=rng.randint(0, 60*24*14))
        rows.append({
            "ticket_id": f"T{i:04d}",
            "ts": ts.isoformat(),
            "city": rng.choice(cities),
            "channel": rng.choice(channels),
            "text": text,
            "label": label
        })
    df = pd.DataFrame(rows)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Wrote {len(df)} rows to {path}")

if __name__ == "__main__":
    generate_csv("data/incidents.csv", n_rows=300)
