import pandas as pd
import json

# Load Longhaul JSON files
strength_df = pd.read_json("strength.json")
cardio_df = pd.read_json("cardio.json")
longhaul_df = pd.concat([strength_df, cardio_df])

# Load cleaned gym CSV
gym_df = pd.read_csv("cleaned_gym_exercises.csv")

# Optionally align and merge based on columns like 'name', 'equipment', etc.
# Then export
merged_df = pd.concat([gym_df, longhaul_df], ignore_index=True)
merged_df.to_csv("personafit_exercises_dataset.csv", index=False)