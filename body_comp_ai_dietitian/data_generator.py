import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_sample_fitness_data(
    start_weight_kg=85.0,
    target_weight_kg=78.0,
    days=60,
    base_tdee=2400,
    avg_intake=2000,
    seed=42
):
    """
    Generates realistic daily body composition and nutrition sequence data.
    Takes into account energy balance (7700 kcal ~= 1 kg fat) + daily water fluctuations
    caused by sodium, carbs, and hydration.
    """
    np.random.seed(seed)
    start_date = datetime.now() - timedelta(days=days)
    
    dates = [start_date + timedelta(days=i) for i in range(days)]
    
    current_true_weight = start_weight_kg
    weights = []
    calories_consumed = []
    tdee_list = []
    protein_list = []
    carbs_list = []
    fat_list = []
    water_list = []
    steps_list = []
    body_fat_pct_list = []
    
    current_bf = 24.5  # initial body fat %
    
    for i in range(days):
        # Calories with natural daily variation (+/- 250 kcal)
        # Occasional weekend cheat day or higher intake
        is_weekend = dates[i].weekday() >= 5
        weekend_boost = 300 if (is_weekend and np.random.rand() > 0.5) else 0
        intake = int(np.random.normal(avg_intake + weekend_boost, 180))
        intake = max(1300, min(3500, intake))
        
        # Steps
        steps = int(np.random.normal(9000, 2500))
        steps = max(3000, min(20000, steps))
        
        # Daily TDEE based on baseline + step expenditure
        daily_tdee = int(base_tdee + (steps - 8000) * 0.04 + np.random.normal(0, 50))
        
        # Macronutrient breakdown
        # Protein: ~1.8-2.2g per kg
        protein = int(current_true_weight * np.random.uniform(1.8, 2.2))
        # Fat: ~20-25% of calories (9 kcal/g)
        fat = int((intake * np.random.uniform(0.20, 0.28)) / 9)
        # Carbs: remainder (4 kcal/g)
        carbs = max(40, int((intake - (protein * 4 + fat * 9)) / 4))
        
        water = round(float(np.random.uniform(2.2, 3.8)), 2)
        
        # True metabolic weight change (7700 kcal deficit = ~1kg fat loss)
        net_cal = intake - daily_tdee
        true_delta_kg = net_cal / 7700.0
        current_true_weight += true_delta_kg
        
        # Water retention fluctuation influenced by carbs & water intake
        carb_water_shift = (carbs - 200) * 0.003  # carbs hold glycogen water
        hydration_shift = (water - 3.0) * 0.15
        noise = np.random.normal(0, 0.35)  # daily scale noise
        
        scale_weight = round(current_true_weight + carb_water_shift + hydration_shift + noise, 2)
        
        # Body fat percentage trend
        current_bf = max(10.0, current_bf + (true_delta_kg * 0.7 / current_true_weight) * 100)
        
        weights.append(scale_weight)
        calories_consumed.append(intake)
        tdee_list.append(daily_tdee)
        protein_list.append(protein)
        carbs_list.append(carbs)
        fat_list.append(fat)
        water_list.append(water)
        steps_list.append(steps)
        body_fat_pct_list.append(round(current_bf, 1))

    df = pd.DataFrame({
        "Date": [d.strftime("%Y-%m-%d") for d in dates],
        "Weight_kg": weights,
        "Calories_Consumed": calories_consumed,
        "TDEE": tdee_list,
        "Net_Calories": [c - t for c, t in zip(calories_consumed, tdee_list)],
        "Protein_g": protein_list,
        "Carbs_g": carbs_list,
        "Fat_g": fat_list,
        "Water_Liters": water_list,
        "Steps": steps_list,
        "BodyFat_Pct": body_fat_pct_list
    })
    
    return df
