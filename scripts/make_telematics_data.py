import argparse, math, random, datetime as dt
from pathlib import Path
import pandas as pd
import numpy as np

def simulate_telematics(n_vehicles=5, days=7, freq_min=15, seed=42, out_csv='data/telematics/sim_telematics.csv'):
    rng = np.random.default_rng(seed)
    start = dt.datetime(2024,1,1,0,0,0)
    periods = int((days*24*60)//freq_min)
    ts = [start + dt.timedelta(minutes=freq_min*i) for i in range(periods)]

    rows = []
    for vid in range(1, n_vehicles+1):
        health = rng.uniform(0.85, 1.0)
        battery_health = rng.uniform(0.9, 1.0)
        use_intensity = rng.uniform(0.6, 1.4)
        base_temp = rng.normal(85, 3)

        for i, t in enumerate(ts):
            hour = t.hour
            dow = t.weekday()
            driving = 1 if (hour in range(7,10) or hour in range(16,19) or rng.random()<0.15) else 0
            speed = max(0, rng.normal(30*driving*use_intensity, 8))
            ambient = 10 + 10*math.sin(2*math.pi*(i/periods)*3)
            engine_temp = base_temp + 0.4*speed + 0.2*ambient + rng.normal(0, 1.5) + (1-health)*25
            voltage = 13.8 - (1-battery_health)*1.8 + rng.normal(0, 0.05)
            vibration = min(1.0, max(0.0, rng.normal(0.15 + 0.01*speed + (1-health)*0.5, 0.05)))
            hazard = (max(0, engine_temp-95)/25) + (max(0, 13.2-voltage)/0.6) + (vibration*1.5)
            hazard *= (1 + (1-health)*1.5)
            prob_fail = min(0.006, 0.0002 + 0.0012*hazard)
            failure = 1 if rng.random() < prob_fail else 0
            if failure:
                health = min(1.0, health + rng.uniform(0.2,0.35))
                battery_health = min(1.0, battery_health + rng.uniform(0.1,0.25))
                base_temp = base_temp - rng.uniform(1.0, 2.0)
            health = max(0.5, health - rng.uniform(0.0003, 0.0007))
            battery_health = max(0.6, battery_health - rng.uniform(0.0001, 0.0003))
            rows.append({
                "vehicle_id": f"V{vid:03d}",
                "ts": t.isoformat(),
                "hour": hour,
                "day_of_week": dow,
                "speed_kmh": round(speed,2),
                "engine_temp_c": round(engine_temp,2),
                "battery_v": round(voltage,3),
                "vibration": round(vibration,3),
                "failure_event": failure
            })
    df = pd.DataFrame(rows)
    out_path = Path(out_csv); out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df):,} rows to {out_path}")

if __name__ == "__main__":
    simulate_telematics()
