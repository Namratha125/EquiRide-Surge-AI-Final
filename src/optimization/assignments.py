import logging
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import sys
import os

# Ensure project paths are resolvable for modular imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- CORRECTED IMPORTS: Matching src/data/utils.py definitions ---
from src.data.utils import (
    get_driver_history_final,
    get_zone_eco_metrics_final,     # Corrected from get_zone_eco_metrics
    get_current_available_drivers,
    get_target_zones,
    get_zone_anomaly_flags
)

from src.utils import load_csv_data

PREDICTION_PATH = 'datasets/forecast_15min_predictions.csv'

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ------------------------
# Configurable Weights (The Core Policy Decision)
# ------------------------
DEFAULT_WEIGHTS = {
    "w_profit": 1.5,
    "w_fair": 1.0,
    "w_eco": 0.8
}


# ------------------------
# Utility Functions (Deterministic Logic)
# ------------------------
def calculate_emission(distance_km, vehicle_type="auto"):
    """Calculate estimated CO2 emissions (kg) for a given distance and vehicle type."""
    emission_factors = {"auto": 0.13, "car": 0.16, "suv": 0.19} 
    return distance_km * emission_factors.get(vehicle_type, 0.13)


def calculate_trip_fare(distance_km: float, vehicle_type="auto") -> float:
    """Calculate realistic fare per trip based on vehicle type and distance."""
    base_fare = 30.0
    per_km_rate = 11.0
    fare = base_fare + max(0, distance_km - 2.0) * per_km_rate

    if vehicle_type == "car":
        return fare * 1.5 
    elif vehicle_type == "suv":
        return fare * 2.0
    else:
        return fare


def fairness_score(driver_id, zone_id, driver_history, zone_demand):
    """
    Compute fairness score: favors drivers with lower recent total earnings or fewer surges.
    
    Relies on get_driver_history_final format: 
    {"total_earnings": X, "recent_surge_assignments": Y}
    """
    driver_data = driver_history.get(driver_id, {"total_earnings": 0, "recent_surge_assignments": 0})
    
    # Invert relationship: lower earnings/surges result in higher fairness score (lower cost penalty)
    # Using .get() to handle simulation dictionary variations
    earnings = driver_data.get("earnings", driver_data.get("total_earnings", 0))
    surges = len(driver_data.get("recent_zone_surges", {}))
    
    earnings_penalty = 1.0 / (1.0 + earnings / 1000)
    surge_penalty = 1.0 / (1.0 + surges)
    
    # Combined score, weighted by potential demand (zone_demand + 1)
    score = (earnings_penalty * surge_penalty) * (zone_demand + 1)
    return score


# ------------------------
# Core Cost Function
# ------------------------
def build_cost_matrix(drivers, zones, forecast_map, anomaly_flags, driver_history, eco_data, weights=None):
    """
    Builds the multi-objective cost matrix for the Hungarian algorithm.
    Cost is low if profit is high, emissions are low, and fairness score is high.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    n, m = len(drivers), len(zones)
    if n == 0 or m == 0:
        return np.zeros((1, 1))
        
    # Ensure eco_data is 2D array
    eco_data = np.array(eco_data)
    
    # Resize logic: If simulation mismatched dimensions, resize eco_data safely
    if eco_data.shape != (n, m):
        # logger.warning(f"Eco data shape mismatch: {eco_data.shape} != ({n}, {m}). Resizing...")
        new_eco = np.zeros((n, m))
        # Fill with available data or default 5.0km
        rows = min(n, eco_data.shape[0])
        cols = min(m, eco_data.shape[1])
        new_eco[:rows, :cols] = eco_data[:rows, :cols]
        new_eco[new_eco == 0] = 5.0 # Default distance
        eco_data = new_eco

    # --- 1. Prepare Priority/Demand Data ---
    priority = np.array([forecast_map.get(z, 0) for z in zones], dtype=float)
    
    # Apply Anomaly Boost 
    event_boost = np.array([anomaly_flags.get(z, 0) * priority[i] * 0.3 for i, z in enumerate(zones)], dtype=float)
    priority = priority + event_boost
    
    # Profit Cost Term: Low Cost = High Demand/Priority (Normalization)
    max_priority = np.max(priority) if len(priority) > 0 else 1.0
    ptp_priority = np.ptp(priority) + 1e-5 if len(priority) > 0 else 1.0
    profit_cost_term = (max_priority - priority) / ptp_priority
    
    # --- 2. Build Multi-Objective Cost Matrix ---
    cost_matrix = np.zeros((n, m))
    
    # Pre-calculate Normalization factors
    # We estimate max possible emission based on data
    max_emission = np.max(eco_data) * 0.19 if eco_data.size > 0 else 1.0
    
    for i, driver in enumerate(drivers):
        for j, zone_id in enumerate(zones):
            dist_km = float(eco_data[i, j])
            
            # Eco Cost Term (Minimize Emission)
            emission_kg = calculate_emission(dist_km, driver.get("vehicle_type", "auto"))
            eco_cost_term = emission_kg / (max_emission + 1e-5) 
            
            # Fairness Cost Term (Minimize Cost = Maximize Fairness Score)
            fair = fairness_score(driver["id"], zone_id, driver_history, priority[j])
            # We just invert fairness raw score because strictly defined max_fair is hard to determine dynamically
            fair_cost_term = 1.0 / (fair + 1e-5) 
            
            # Weighted Sum of Costs (Minimize total cost)
            cost_matrix[i, j] = (
                weights["w_profit"] * profit_cost_term[j] + 
                weights["w_eco"] * eco_cost_term + 
                weights["w_fair"] * fair_cost_term
            )
            
    return cost_matrix


# ------------------------
# Main Assignment Function
# ------------------------
def assign_drivers(drivers, zones, forecast_df, anomaly_flags, driver_history, eco_data, weights=None):
    """
    Performs multi-objective assignment using the Hungarian algorithm.
    """
    # Convert forecast DataFrame to Map for quick lookup
    forecast_map = forecast_df.set_index('h3_index')['pred_bookings_15min'].to_dict()
    
    cost_matrix = build_cost_matrix(drivers, zones, forecast_map, anomaly_flags, driver_history, eco_data, weights)
    
    # Hungarian assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    results = []
    for i, j in zip(row_ind, col_ind):
        driver = drivers[i]
        zone = zones[j]
        
        # Calculate final metrics for output logging
        dist_km = float(eco_data[i, j]) if i < eco_data.shape[0] and j < eco_data.shape[1] else 5.0
        vehicle_type = driver.get("vehicle_type", "auto")
        emission_kg = calculate_emission(dist_km, vehicle_type=vehicle_type)
        trips_est = forecast_map.get(zone, 0)
        fare_per_trip = calculate_trip_fare(dist_km, vehicle_type=vehicle_type)
        profit_est = trips_est * fare_per_trip 
        current_fairness = fairness_score(driver["id"], zone, driver_history, trips_est)

        results.append({
            "driver_id": driver["id"],
            "assigned_zone": zone,
            "forecasted_demand": round(trips_est, 2),
            "estimated_profit": round(profit_est, 2),
            "fairness_score": round(current_fairness, 4),
            "repositioning_distance_km": round(dist_km, 2),
            "emission_kg": round(emission_kg, 4)
        })

    df = pd.DataFrame(results)
    logger.info("Assignments completed: %d drivers assigned.", len(df))
    return df

# ------------------------
# WRAPPER FUNCTION (Used by the E2E Orchestrator)
# ------------------------
def get_repositioning_plan():
    """
    Wrapper to fetch all necessary data inputs and execute assign_drivers.
    (Called directly by src/pipeline/run_pipeline.py)
    """
    logger.info("Fetching inputs for Repositioning Plan generation...")
    
    try:
        # Prediction output saved by src/models/infer.py
        forecast_df = load_csv_data(PREDICTION_PATH)
    except Exception as e:
        logger.error(f"Failed to load forecast file: {e}")
        return pd.DataFrame()
        
    if forecast_df.empty:
        logger.error("Forecast data is missing or empty for optimization.")
        return pd.DataFrame()
        
    # Fetch deterministic data inputs from src/data/utils.py
    # FIX: Removed argument (num_drivers) if the util function doesn't support it
    drivers = get_current_available_drivers() 
    zones = get_target_zones()
    anomaly_flags = get_zone_anomaly_flags() 
    driver_history = get_driver_history_final()
    
    # FIX: Corrected function name to match import
    eco_data = get_zone_eco_metrics_final() 

    if len(drivers) == 0 or len(zones) == 0:
        logger.warning("No drivers or zones available for planning.")
        return pd.DataFrame()
    
    # Execute core assignment
    assignments = assign_drivers(drivers, zones, forecast_df, anomaly_flags, driver_history, eco_data)
    return assignments
