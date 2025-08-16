from dataclasses import dataclass

@dataclass
class PricingParams:
    flagfall: float = 2.0
    per_km: float = 1.2
    per_min: float = 0.15
    surge_cap: float = 1.8
    surge_floor: float = 0.8
    min_fare: float = 5.0
    uncertainty_fee_per_min: float = 0.02

def surge_from_market(demand_index: float, supply_index: float, params: PricingParams) -> float:
    ratio = max(0.01, demand_index / max(0.01, supply_index))
    raw = ratio ** 0.5
    return max(params.surge_floor, min(params.surge_cap, raw))

def compute_price(distance_km: float, eta_minutes: float, demand_index: float, supply_index: float,
                  interval_width: float, params: PricingParams = PricingParams()) -> dict:
    base = params.flagfall + params.per_km * max(0.0, distance_km) + params.per_min * max(0.0, eta_minutes)
    surge = surge_from_market(demand_index, supply_index, params)
    uncertainty_fee = params.uncertainty_fee_per_min * max(0.0, interval_width)
    total = base * surge + uncertainty_fee
    total = max(total, params.min_fare)
    return {
        "components": {
            "flagfall": round(params.flagfall,2),
            "per_km": round(params.per_km * max(0.0, distance_km),2),
            "per_min": round(params.per_min * max(0.0, eta_minutes),2),
            "surge_multiplier": round(surge,2),
            "uncertainty_fee": round(uncertainty_fee,2),
        },
        "price": round(total,2)
    }
