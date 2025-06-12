import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import os
import warnings
import kagglehub

# Download latest version
path = kagglehub.dataset_download("cdc/national-health-and-nutrition-examination-survey")

warnings.filterwarnings('ignore')

class DietRecommendationSystem:
    """
    A comprehensive diet recommendation system that provides personalized
    nutrition plans based on user health data, goals, and activity levels.
    Supports multiple training styles and physique goals.
    """
    
    def __init__(self, dataset_path):
        """
        Initialize the diet recommendation system with NHANES dataset
        
        Parameters:
        -----------
        dataset_path : str
            Path to the NHANES dataset files
        """
        self.dataset_path = dataset_path
        self.nhanes_data = None
        self.food_data = None
        self.nutrition_data = None
        self._load_datasets()
        
        # Default values for various activity levels and goals
        self.activity_multipliers = {
            'sedentary': 1.2,
            'lightly_active': 1.375,
            'moderately_active': 1.55,
            'very_active': 1.725,
            'extra_active': 1.9
        }
        
        # Training type adjustments (can be combined)
        self.training_adjustments = {
            'cardio': {'protein': 1.2, 'carbs': 1.3, 'fat': 0.8, 'calories': 1.0},
            'weightlifting': {'protein': 1.6, 'carbs': 1.2, 'fat': 0.9, 'calories': 1.05},
            'bodybuilding': {'protein': 1.8, 'carbs': 1.3, 'fat': 0.7, 'calories': 1.1},
            'endurance': {'protein': 1.4, 'carbs': 1.5, 'fat': 0.9, 'calories': 1.15},
            'crossfit': {'protein': 1.6, 'carbs': 1.4, 'fat': 0.8, 'calories': 1.1},
            'yoga': {'protein': 1.0, 'carbs': 1.1, 'fat': 1.0, 'calories': 0.95},
            'hiit': {'protein': 1.4, 'carbs': 1.3, 'fat': 0.85, 'calories': 1.05},
            'powerlifting': {'protein': 1.7, 'carbs': 1.4, 'fat': 0.9, 'calories': 1.1},
            'sports_specific': {'protein': 1.5, 'carbs': 1.4, 'fat': 0.9, 'calories': 1.1},
            'no_training': {'protein': 0.8, 'carbs': 1.0, 'fat': 1.0, 'calories': 1.0}
        }
        
        # Physique goal adjustments (can be combined)
        self.physique_adjustments = {
            'weight_loss': {'calories': 0.8, 'protein': 1.2, 'carbs': 0.8, 'fat': 0.8},
            'maintenance': {'calories': 1.0, 'protein': 1.0, 'carbs': 1.0, 'fat': 1.0},
            'muscle_gain': {'calories': 1.1, 'protein': 1.2, 'carbs': 1.2, 'fat': 0.9},
            'toning': {'calories': 0.95, 'protein': 1.3, 'carbs': 0.9, 'fat': 0.85},
            'athletic_performance': {'calories': 1.1, 'protein': 1.2, 'carbs': 1.3, 'fat': 0.9},
            'endurance_improvement': {'calories': 1.15, 'protein': 1.3, 'carbs': 1.4, 'fat': 0.85},
            'strength_gain': {'calories': 1.1, 'protein': 1.4, 'carbs': 1.2, 'fat': 0.9},
            'body_recomposition': {'calories': 1.0, 'protein': 1.4, 'carbs': 0.95, 'fat': 0.85},
            'general_health': {'calories': 1.0, 'protein': 1.1, 'carbs': 1.0, 'fat': 1.0}
        }
        
        # Essential micronutrients to track
        self.essential_micros = [
            'vitamin_a', 'vitamin_c', 'vitamin_d', 'vitamin_e', 'vitamin_k',
            'thiamin', 'riboflavin', 'niacin', 'vitamin_b6', 'folate', 'vitamin_b12',
            'calcium', 'iron', 'magnesium', 'phosphorus', 'potassium', 'sodium', 'zinc'
        ]
        
    def _load_datasets(self):
        """Load and preprocess NHANES datasets"""
        try:
            # Try to load actual NHANES data
            possible_files = [
                'demographic.csv', 'DEMO_J.csv', 'P_DEMO.csv',
                'dietary.csv', 'DR1TOT_J.csv', 'P_DR1TOT.csv',
                'nutrition.csv', 'NUTRIENT.csv', 'P_NUTRIENT.csv'
            ]
            
            loaded_files = []
            
            # Check what files exist in the dataset path
            if os.path.exists(self.dataset_path):
                available_files = os.listdir(self.dataset_path)
                print(f"Available files in dataset: {available_files}")
                
                # Try to load demographic data
                demo_file = None
                for file in possible_files[:3]:
                    if file in available_files:
                        demo_file = file
                        break
                
                if demo_file:
                    demo_path = os.path.join(self.dataset_path, demo_file)
                    self.nhanes_data = pd.read_csv(demo_path)
                    loaded_files.append(demo_file)
                    print(f"Loaded demographic data: {demo_file} ({len(self.nhanes_data)} records)")
                
                # Try to load dietary data
                diet_file = None
                for file in possible_files[3:6]:
                    if file in available_files:
                        diet_file = file
                        break
                
                if diet_file:
                    diet_path = os.path.join(self.dataset_path, diet_file)
                    dietary_data = pd.read_csv(diet_path)
                    loaded_files.append(diet_file)
                    print(f"Loaded dietary data: {diet_file} ({len(dietary_data)} records)")
                    
                    # Process dietary data to create food database
                    self._process_nhanes_dietary_data(dietary_data)
                
                # Try to load nutrition data
                nutrition_file = None
                for file in possible_files[6:]:
                    if file in available_files:
                        nutrition_file = file
                        break
                
                if nutrition_file:
                    nutrition_path = os.path.join(self.dataset_path, nutrition_file)
                    self.nutrition_data = pd.read_csv(nutrition_path)
                    loaded_files.append(nutrition_file)
                    print(f"Loaded nutrition data: {nutrition_file} ({len(self.nutrition_data)} records)")
                
                # Merge datasets if we have matching ID columns
                if self.nhanes_data is not None and self.nutrition_data is not None:
                    common_cols = set(self.nhanes_data.columns) & set(self.nutrition_data.columns)
                    id_cols = [col for col in common_cols if 'SEQN' in col or 'ID' in col.upper()]
                    
                    if id_cols:
                        merge_col = id_cols[0]
                        self.nhanes_data = pd.merge(self.nhanes_data, self.nutrition_data, 
                                                  on=merge_col, how='left')
                        print(f"Merged datasets on column: {merge_col}")
                
                if loaded_files:
                    print(f"Successfully loaded NHANES data files: {loaded_files}")
                    # Use actual NHANES data for recommendations
                    self._process_nhanes_data()
                else:
                    raise FileNotFoundError("No recognized NHANES files found")
                    
            else:
                raise FileNotFoundError(f"Dataset path does not exist: {self.dataset_path}")
                
        except Exception as e:
            print(f"Error loading NHANES datasets: {e}")
            print("Creating fallback food database...")
            self._create_fallback_food_database()
    
    def _process_nhanes_dietary_data(self, dietary_data):
        """Process NHANES dietary data to extract food information"""
        try:
            # Common NHANES dietary data columns
            nutrient_cols = {
                'DR1TKCAL': 'calories',
                'DR1TPROT': 'protein', 
                'DR1TCARB': 'carbs',
                'DR1TTFAT': 'fat',
                'DR1TSFAT': 'saturated_fat',
                'DR1TFIBE': 'fiber',
                'DR1TSODI': 'sodium',
                'DR1TPOTA': 'potassium',
                'DR1TVB12': 'vitamin_b12',
                'DR1TFOLR': 'folate',
                'DR1TIRON': 'iron',
                'DR1TCALC': 'calcium',
                'DR1TZINC': 'zinc'
            }
            
            # Check which columns exist in the data
            available_nutrients = {}
            for nhanes_col, standard_col in nutrient_cols.items():
                if nhanes_col in dietary_data.columns:
                    available_nutrients[nhanes_col] = standard_col
            
            if available_nutrients:
                # Create food data from dietary intake records
                food_records = dietary_data[list(available_nutrients.keys())].copy()
                food_records = food_records.rename(columns=available_nutrients)
                
                # Remove records with missing calorie data
                if 'calories' in food_records.columns:
                    food_records = food_records.dropna(subset=['calories'])
                    food_records = food_records[food_records['calories'] > 0]
                
                # Group similar food items if we have food codes/descriptions
                food_desc_cols = [col for col in dietary_data.columns 
                                if 'FDCD' in col or 'DESC' in col or 'FOOD' in col.upper()]
                
                if food_desc_cols:
                    food_records['food_name'] = dietary_data[food_desc_cols[0]]
                else:
                    food_records['food_name'] = 'NHANES_Food_' + food_records.index.astype(str)
                
                self.food_data = food_records
                print(f"Processed {len(self.food_data)} food records from NHANES dietary data")
            else:
                print("No recognized nutrient columns found in dietary data")
                self._create_fallback_food_database()
                
        except Exception as e:
            print(f"Error processing NHANES dietary data: {e}")
            self._create_fallback_food_database()
    
    def _process_nhanes_data(self):
        """Process NHANES demographic data for better recommendations"""
        try:
            if self.nhanes_data is not None:
                # Common NHANES demographic columns
                demo_mapping = {
                    'RIAGENDR': 'gender',  # 1=Male, 2=Female
                    'RIDAGEYR': 'age',
                    'BMXWT': 'weight_kg',
                    'BMXHT': 'height_cm', 
                    'BMXBMI': 'bmi'
                }
                
                # Check which columns exist and create processed data
                processed_cols = {}
                for nhanes_col, standard_col in demo_mapping.items():
                    if nhanes_col in self.nhanes_data.columns:
                        processed_cols[nhanes_col] = standard_col
                
                if processed_cols:
                    self.processed_nhanes = self.nhanes_data[list(processed_cols.keys())].copy()
                    self.processed_nhanes = self.processed_nhanes.rename(columns=processed_cols)
                    
                    # Convert gender coding (1=Male, 2=Female)
                    if 'gender' in self.processed_nhanes.columns:
                        self.processed_nhanes['gender'] = self.processed_nhanes['gender'].map({1: 'male', 2: 'female'})
                    
                    # Remove invalid data
                    numeric_cols = ['age', 'weight_kg', 'height_cm', 'bmi']
                    for col in numeric_cols:
                        if col in self.processed_nhanes.columns:
                            self.processed_nhanes = self.processed_nhanes[self.processed_nhanes[col] > 0]
                    
                    print(f"Processed NHANES demographic data: {len(self.processed_nhanes)} valid records")
                    
        except Exception as e:
            print(f"Error processing NHANES demographic data: {e}")
    
    def _create_fallback_food_database(self):
        """Create a comprehensive fallback food database if NHANES data cannot be loaded"""
        # Expanded food database with more variety
        foods = {
            # Proteins
            'chicken_breast': {'calories': 165, 'protein': 31, 'carbs': 0, 'fat': 3.6, 'vitamin_b6': 0.6, 'niacin': 13.7},
            'salmon': {'calories': 206, 'protein': 22, 'carbs': 0, 'fat': 13, 'vitamin_d': 14, 'omega3': 1.8},
            'lean_beef': {'calories': 250, 'protein': 26, 'carbs': 0, 'fat': 15, 'iron': 2.6, 'zinc': 4.8},
            'eggs': {'calories': 143, 'protein': 12.6, 'carbs': 0.6, 'fat': 9.5, 'vitamin_b12': 0.89, 'choline': 147},
            'greek_yogurt': {'calories': 59, 'protein': 10, 'carbs': 3.6, 'fat': 0.4, 'calcium': 111, 'probiotics': 1},
            'tuna': {'calories': 144, 'protein': 30, 'carbs': 0, 'fat': 1, 'vitamin_d': 5.7, 'selenium': 90},
            'turkey_breast': {'calories': 135, 'protein': 30, 'carbs': 0, 'fat': 1, 'niacin': 11.8, 'vitamin_b6': 0.8},
            'cottage_cheese': {'calories': 98, 'protein': 11, 'carbs': 3.4, 'fat': 4.3, 'calcium': 83, 'phosphorus': 159},
            'tofu': {'calories': 76, 'protein': 8, 'carbs': 1.9, 'fat': 4.8, 'calcium': 350, 'iron': 5.4},
            'lentils': {'calories': 116, 'protein': 9, 'carbs': 20, 'fat': 0.4, 'fiber': 8, 'folate': 181},
            
            # Carbohydrates
            'brown_rice': {'calories': 112, 'protein': 2.6, 'carbs': 23, 'fat': 0.9, 'magnesium': 43, 'fiber': 1.8},
            'sweet_potato': {'calories': 86, 'protein': 1.6, 'carbs': 20, 'fat': 0.1, 'vitamin_a': 14187, 'fiber': 3},
            'oats': {'calories': 389, 'protein': 16.9, 'carbs': 66, 'fat': 6.9, 'fiber': 10.6, 'magnesium': 177},
            'quinoa': {'calories': 120, 'protein': 4.4, 'carbs': 21.3, 'fat': 1.9, 'magnesium': 64, 'fiber': 2.8},
            'whole_wheat_pasta': {'calories': 124, 'protein': 5, 'carbs': 25, 'fat': 1.1, 'fiber': 3.2, 'niacin': 3.4},
            'banana': {'calories': 89, 'protein': 1.1, 'carbs': 22.8, 'fat': 0.3, 'potassium': 358, 'vitamin_b6': 0.4},
            'white_rice': {'calories': 130, 'protein': 2.7, 'carbs': 28, 'fat': 0.3, 'niacin': 2.3, 'folate': 5},
            'potato': {'calories': 77, 'protein': 2, 'carbs': 17, 'fat': 0.1, 'potassium': 425, 'vitamin_c': 19.7},
            'black_beans': {'calories': 127, 'protein': 8.9, 'carbs': 22.8, 'fat': 0.5, 'fiber': 7.5, 'folate': 128},
            
            # Vegetables  
            'spinach': {'calories': 23, 'protein': 2.9, 'carbs': 3.6, 'fat': 0.4, 'vitamin_k': 483, 'folate': 194},
            'broccoli': {'calories': 34, 'protein': 2.8, 'carbs': 6.6, 'fat': 0.4, 'vitamin_c': 89.2, 'vitamin_k': 102},
            'kale': {'calories': 35, 'protein': 2.2, 'carbs': 4.4, 'fat': 1.5, 'vitamin_a': 15376, 'vitamin_c': 93.4},
            'bell_peppers': {'calories': 20, 'protein': 0.9, 'carbs': 4.6, 'fat': 0.2, 'vitamin_c': 127.7, 'vitamin_a': 1624},
            'carrots': {'calories': 41, 'protein': 0.9, 'carbs': 9.6, 'fat': 0.2, 'vitamin_a': 16706, 'fiber': 2.8},
            'tomatoes': {'calories': 18, 'protein': 0.9, 'carbs': 3.9, 'fat': 0.2, 'vitamin_c': 13.7, 'lycopene': 2573},
            
            # Fats
            'avocado': {'calories': 160, 'protein': 2, 'carbs': 8.5, 'fat': 14.7, 'potassium': 485, 'fiber': 6.7},
            'almonds': {'calories': 579, 'protein': 21, 'carbs': 22, 'fat': 49, 'vitamin_e': 25.6, 'magnesium': 270},
            'olive_oil': {'calories': 884, 'protein': 0, 'carbs': 0, 'fat': 100, 'vitamin_e': 14.4, 'vitamin_k': 60},
            'walnuts': {'calories': 654, 'protein': 15, 'carbs': 14, 'fat': 65, 'omega3': 9, 'magnesium': 158},
            'chia_seeds': {'calories': 486, 'protein': 17, 'carbs': 42, 'fat': 31, 'fiber': 34, 'calcium': 631},
            'peanut_butter': {'calories': 588, 'protein': 25, 'carbs': 20, 'fat': 50, 'niacin': 12, 'magnesium': 168},
            
            # Dairy
            'milk_2percent': {'calories': 50, 'protein': 3.3, 'carbs': 4.8, 'fat': 2, 'calcium': 120, 'vitamin_d': 1.2},
            'cheddar_cheese': {'calories': 403, 'protein': 25, 'carbs': 3.4, 'fat': 33, 'calcium': 710, 'vitamin_a': 1242},
            
            # Snacks/Others
            'dark_chocolate': {'calories': 546, 'protein': 5, 'carbs': 61, 'fat': 31, 'iron': 11.9, 'magnesium': 146},
            'green_tea': {'calories': 1, 'protein': 0, 'carbs': 0, 'fat': 0, 'antioxidants': 1, 'caffeine': 25},
        }
        
        # Convert to DataFrame
        self.food_data = pd.DataFrame(foods).T.reset_index()
        self.food_data.rename(columns={'index': 'food_name'}, inplace=True)
        
        # Create more realistic demographic data for recommendations
        np.random.seed(42)  # For reproducible results
        self.nhanes_data = pd.DataFrame({
            'SEQN': range(1000, 2000),
            'RIAGENDR': np.random.choice([1, 2], 1000),  # 1=Male, 2=Female
            'RIDAGEYR': np.random.randint(18, 80, 1000),
            'BMXWT': np.random.normal(70, 15, 1000),   # Weight in kg
            'BMXHT': np.random.normal(170, 10, 1000),  # Height in cm
        })
        
        # Calculate BMI
        self.nhanes_data['BMXBMI'] = (self.nhanes_data['BMXWT'] / 
                                     (self.nhanes_data['BMXHT'] / 100) ** 2)
        
        # Remove outliers
        self.nhanes_data = self.nhanes_data[
            (self.nhanes_data['BMXWT'] > 40) & (self.nhanes_data['BMXWT'] < 150) &
            (self.nhanes_data['BMXHT'] > 140) & (self.nhanes_data['BMXHT'] < 210) &
            (self.nhanes_data['BMXBMI'] > 15) & (self.nhanes_data['BMXBMI'] < 50)
        ]
        
        print(f"Using enhanced fallback database with {len(self.food_data)} foods and {len(self.nhanes_data)} demographic records")
    
    def calculate_bmr(self, weight, height, age, gender):
        """Calculate Basal Metabolic Rate using the Mifflin-St Jeor Equation"""
        if gender.lower() == 'male':
            return (10 * weight) + (6.25 * height) - (5 * age) + 5
        else:
            return (10 * weight) + (6.25 * height) - (5 * age) - 161
    
    def calculate_tdee(self, bmr, activity_level):
        """Calculate Total Daily Energy Expenditure"""
        multiplier = self.activity_multipliers.get(activity_level, 1.2)
        return bmr * multiplier
    
    def combine_adjustments(self, training_types, physique_goals):
        """
        Combine multiple training types and physique goals
        
        Parameters:
        -----------
        training_types : list
            List of training types
        physique_goals : list
            List of physique goals
            
        Returns:
        --------
        dict
            Combined adjustments
        """
        combined_training = {'protein': 1.0, 'carbs': 1.0, 'fat': 1.0, 'calories': 1.0}
        combined_goals = {'protein': 1.0, 'carbs': 1.0, 'fat': 1.0, 'calories': 1.0}
        
        # Combine training adjustments (average)
        if training_types:
            valid_training = [t for t in training_types if t in self.training_adjustments]
            if valid_training:
                for key in combined_training:
                    values = [self.training_adjustments[t][key] for t in valid_training]
                    combined_training[key] = sum(values) / len(values)
        
        # Combine goal adjustments (average)
        if physique_goals:
            valid_goals = [g for g in physique_goals if g in self.physique_adjustments]
            if valid_goals:
                for key in combined_goals:
                    values = [self.physique_adjustments[g][key] for g in valid_goals]
                    combined_goals[key] = sum(values) / len(values)
        
        return combined_training, combined_goals
    
    def calculate_macros(self, tdee, training_types, physique_goals, weight=70):
        """
        Calculate macronutrient distribution based on multiple goals and training types
        
        Parameters:
        -----------
        tdee : float
            Total Daily Energy Expenditure
        training_types : list
            List of training types
        physique_goals : list
            List of physique goals
        weight : float
            Body weight in kg
            
        Returns:
        --------
        dict
            Macronutrient breakdown in grams and percentages
        """
        # Get combined adjustments
        train_adj, goal_adj = self.combine_adjustments(training_types, physique_goals)
        
        # Apply calorie adjustments
        adj_calories = tdee * train_adj['calories'] * goal_adj['calories']
        
        # Calculate protein (g/kg of body weight)
        base_protein_per_kg = 0.8
        protein_per_kg = base_protein_per_kg * train_adj['protein'] * goal_adj['protein']
        
        # Calculate macros in grams
        protein_g = protein_per_kg * weight
        protein_cals = protein_g * 4
        
        # Remaining calories split between carbs and fats
        remaining_cals = adj_calories - protein_cals
        
        # Calculate carbs and fats based on adjustments
        carb_multiplier = train_adj['carbs'] * goal_adj['carbs']
        fat_multiplier = train_adj['fat'] * goal_adj['fat']
        total_multiplier = carb_multiplier + fat_multiplier
        
        carb_cals = remaining_cals * (carb_multiplier / total_multiplier)
        fat_cals = remaining_cals * (fat_multiplier / total_multiplier)
        
        carbs_g = carb_cals / 4
        fat_g = fat_cals / 9
        
        return {
            'total_calories': round(adj_calories),
            'protein': {
                'grams': round(protein_g),
                'calories': round(protein_cals),
                'percentage': round((protein_cals / adj_calories) * 100)
            },
            'carbs': {
                'grams': round(carbs_g),
                'calories': round(carb_cals),
                'percentage': round((carb_cals / adj_calories) * 100)
            },
            'fat': {
                'grams': round(fat_g),
                'calories': round(fat_cals),
                'percentage': round((fat_cals / adj_calories) * 100)
            },
            'training_types_used': training_types,
            'physique_goals_used': physique_goals
        }
    
    def get_micronutrient_targets(self, gender, age, weight, physique_goals):
        """Determine target micronutrient intake based on multiple goals"""
        # Base RDI values
        base_micros = {
            'vitamin_a': 900 if gender == 'male' else 700,
            'vitamin_c': 90 if gender == 'male' else 75,
            'vitamin_d': 15,
            'vitamin_e': 15,
            'vitamin_k': 120 if gender == 'male' else 90,
            'thiamin': 1.2 if gender == 'male' else 1.1,
            'riboflavin': 1.3 if gender == 'male' else 1.1,
            'niacin': 16 if gender == 'male' else 14,
            'vitamin_b6': 1.3,
            'folate': 400,
            'vitamin_b12': 2.4,
            'calcium': 1000,
            'iron': 8 if gender == 'male' else 18,
            'magnesium': 400 if gender == 'male' else 310,
            'phosphorus': 700,
            'potassium': 3400 if gender == 'male' else 2600,
            'sodium': 2300,
            'zinc': 11 if gender == 'male' else 8
        }
        
        # Adjustments based on physique goals
        goal_adjustments = {
            'weight_loss': {'vitamin_d': 1.2, 'calcium': 1.1, 'iron': 1.2, 'magnesium': 1.2},
            'muscle_gain': {'vitamin_d': 1.2, 'zinc': 1.3, 'magnesium': 1.3, 'vitamin_b6': 1.2},
            'athletic_performance': {'vitamin_b6': 1.3, 'vitamin_b12': 1.3, 'iron': 1.3, 'magnesium': 1.3},
            'endurance_improvement': {'iron': 1.4, 'vitamin_b12': 1.3, 'potassium': 1.3, 'magnesium': 1.3},
            'strength_gain': {'zinc': 1.3, 'magnesium': 1.3, 'vitamin_d': 1.2},
            'body_recomposition': {'vitamin_d': 1.1, 'magnesium': 1.2, 'zinc': 1.2}
        }
        
        