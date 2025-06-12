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
        self._load_datasets()
        
        # Default values for various activity levels and goals
        self.activity_multipliers = {
            'sedentary': 1.2,
            'lightly_active': 1.375,
            'moderately_active': 1.55,
            'very_active': 1.725,
            'extra_active': 1.9
        }
        
        # Training type adjustments
        self.training_adjustments = {
            'cardio': {'protein': 1.2, 'carbs': 1.3, 'fat': 0.8},
            'weightlifting': {'protein': 1.6, 'carbs': 1.2, 'fat': 0.9},
            'bodybuilding': {'protein': 1.8, 'carbs': 1.3, 'fat': 0.7},
            'endurance': {'protein': 1.4, 'carbs': 1.5, 'fat': 0.9},
            'crossfit': {'protein': 1.6, 'carbs': 1.4, 'fat': 0.8},
            'yoga': {'protein': 1.0, 'carbs': 1.1, 'fat': 1.0},
            'no_training': {'protein': 0.8, 'carbs': 1.0, 'fat': 1.0}
        }
        
        # Physique goal adjustments
        self.physique_adjustments = {
            'weight_loss': {'calories': 0.8, 'protein': 1.2, 'carbs': 0.8, 'fat': 0.8},
            'maintenance': {'calories': 1.0, 'protein': 1.0, 'carbs': 1.0, 'fat': 1.0},
            'muscle_gain': {'calories': 1.1, 'protein': 1.2, 'carbs': 1.2, 'fat': 0.9},
            'toning': {'calories': 0.95, 'protein': 1.3, 'carbs': 0.9, 'fat': 0.85},
            'athletic_performance': {'calories': 1.1, 'protein': 1.2, 'carbs': 1.3, 'fat': 0.9}
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
            # Load demographic data
            demo_path = os.path.join(self.dataset_path, 'demographic.csv')
            self.nhanes_data = pd.read_csv(demo_path)
            
            # Load dietary data
            diet_path = os.path.join(self.dataset_path, 'dietary.csv')
            self.food_data = pd.read_csv(diet_path)
            
            # Load nutritional data
            nutrition_path = os.path.join(self.dataset_path, 'nutrition.csv')
            nutrition_data = pd.read_csv(nutrition_path)
            
            # Merge datasets as needed
            # This is a simplified example - actual implementation would require
            # more sophisticated merging based on NHANES structure
            if 'SEQN' in self.nhanes_data.columns and 'SEQN' in nutrition_data.columns:
                self.nhanes_data = pd.merge(self.nhanes_data, nutrition_data, on='SEQN', how='left')
            
            print(f"Successfully loaded datasets with {len(self.nhanes_data)} records")
        except Exception as e:
            print(f"Error loading datasets: {e}")
            # Create fallback food database if NHANES data cannot be loaded
            self._create_fallback_food_database()
    
    def _create_fallback_food_database(self):
        """Create a fallback food database if NHANES data cannot be loaded"""
        # Create a basic food database with common foods and their nutritional values
        foods = {
            'chicken_breast': {'calories': 165, 'protein': 31, 'carbs': 0, 'fat': 3.6, 'vitamin_b6': 0.6, 'niacin': 13.7},
            'salmon': {'calories': 206, 'protein': 22, 'carbs': 0, 'fat': 13, 'vitamin_d': 14, 'omega3': 1.8},
            'brown_rice': {'calories': 112, 'protein': 2.6, 'carbs': 23, 'fat': 0.9, 'magnesium': 43, 'fiber': 1.8},
            'sweet_potato': {'calories': 86, 'protein': 1.6, 'carbs': 20, 'fat': 0.1, 'vitamin_a': 14187, 'fiber': 3},
            'spinach': {'calories': 23, 'protein': 2.9, 'carbs': 3.6, 'fat': 0.4, 'vitamin_k': 483, 'folate': 194},
            'eggs': {'calories': 143, 'protein': 12.6, 'carbs': 0.6, 'fat': 9.5, 'vitamin_b12': 0.89, 'choline': 147},
            'greek_yogurt': {'calories': 59, 'protein': 10, 'carbs': 3.6, 'fat': 0.4, 'calcium': 111, 'probiotics': 1},
            'oats': {'calories': 389, 'protein': 16.9, 'carbs': 66, 'fat': 6.9, 'fiber': 10.6, 'magnesium': 177},
            'almonds': {'calories': 579, 'protein': 21, 'carbs': 22, 'fat': 49, 'vitamin_e': 25.6, 'magnesium': 270},
            'avocado': {'calories': 160, 'protein': 2, 'carbs': 8.5, 'fat': 14.7, 'potassium': 485, 'fiber': 6.7},
            'banana': {'calories': 89, 'protein': 1.1, 'carbs': 22.8, 'fat': 0.3, 'potassium': 358, 'vitamin_b6': 0.4},
            'broccoli': {'calories': 34, 'protein': 2.8, 'carbs': 6.6, 'fat': 0.4, 'vitamin_c': 89.2, 'vitamin_k': 102},
            'olive_oil': {'calories': 884, 'protein': 0, 'carbs': 0, 'fat': 100, 'vitamin_e': 14.4, 'vitamin_k': 60},
            'black_beans': {'calories': 127, 'protein': 8.9, 'carbs': 22.8, 'fat': 0.5, 'fiber': 7.5, 'folate': 128},
            'quinoa': {'calories': 120, 'protein': 4.4, 'carbs': 21.3, 'fat': 1.9, 'magnesium': 64, 'fiber': 2.8}
        }
        
        # Convert to DataFrame
        self.food_data = pd.DataFrame(foods).T.reset_index()
        self.food_data.rename(columns={'index': 'food_name'}, inplace=True)
        
        # Create simple demographic data
        self.nhanes_data = pd.DataFrame({
            'SEQN': range(1000, 1100),
            'RIAGENDR': np.random.choice([1, 2], 100),  # 1=Male, 2=Female
            'RIDAGEYR': np.random.randint(18, 80, 100),
            'BMXWT': np.random.uniform(50, 100, 100),   # Weight in kg
            'BMXHT': np.random.uniform(150, 190, 100),  # Height in cm
            'BMXBMI': np.random.uniform(18.5, 35, 100)  # BMI
        })
        
        print("Using fallback food database with limited data")
    
    def calculate_bmr(self, weight, height, age, gender):
        """
        Calculate Basal Metabolic Rate using the Mifflin-St Jeor Equation
        
        Parameters:
        -----------
        weight : float
            Weight in kilograms
        height : float
            Height in centimeters
        age : int
            Age in years
        gender : str
            'male' or 'female'
            
        Returns:
        --------
        float
            BMR in calories per day
        """
        if gender.lower() == 'male':
            return (10 * weight) + (6.25 * height) - (5 * age) + 5
        else:
            return (10 * weight) + (6.25 * height) - (5 * age) - 161
    
    def calculate_tdee(self, bmr, activity_level):
        """
        Calculate Total Daily Energy Expenditure
        
        Parameters:
        -----------
        bmr : float
            Basal Metabolic Rate
        activity_level : str
            One of: 'sedentary', 'lightly_active', 'moderately_active', 
            'very_active', 'extra_active'
            
        Returns:
        --------
        float
            TDEE in calories per day
        """
        multiplier = self.activity_multipliers.get(activity_level, 1.2)
        return bmr * multiplier
    
    def calculate_macros(self, tdee, training_type, physique_goal):
        """
        Calculate macronutrient distribution based on goals
        
        Parameters:
        -----------
        tdee : float
            Total Daily Energy Expenditure
        training_type : str
            Type of training (e.g., 'weightlifting', 'cardio', etc.)
        physique_goal : str
            Goal (e.g., 'weight_loss', 'muscle_gain', etc.)
            
        Returns:
        --------
        dict
            Macronutrient breakdown in grams and percentages
        """
        # Apply goal-specific calorie adjustment
        adj_calories = tdee * self.physique_adjustments.get(physique_goal, {}).get('calories', 1.0)
        
        # Get training and goal adjustments
        train_adj = self.training_adjustments.get(training_type, {'protein': 1.0, 'carbs': 1.0, 'fat': 1.0})
        goal_adj = self.physique_adjustments.get(physique_goal, {'protein': 1.0, 'carbs': 1.0, 'fat': 1.0})
        
        # Calculate protein (g/kg of body weight)
        protein_per_kg = 0.8 * train_adj['protein'] * goal_adj['protein']
        
        # Default weight assumption if not provided - using stat from NHANES average
        avg_weight = 70  # kg
        
        # Calculate macros in grams
        protein_g = protein_per_kg * avg_weight
        protein_cals = protein_g * 4
        
        # Remaining calories split between carbs and fats based on adjustments
        remaining_cals = adj_calories - protein_cals
        
        # Calculate carbs and fats based on remaining calories
        carb_ratio = train_adj['carbs'] * goal_adj['carbs']
        fat_ratio = train_adj['fat'] * goal_adj['fat']
        total_ratio = carb_ratio + fat_ratio
        
        carb_cals = remaining_cals * (carb_ratio / total_ratio)
        fat_cals = remaining_cals * (fat_ratio / total_ratio)
        
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
            }
        }
    
    def get_micronutrient_targets(self, gender, age, weight, physique_goal):
        """
        Determine target micronutrient intake based on gender, age, and goals
        
        Parameters:
        -----------
        gender : str
            'male' or 'female'
        age : int
            Age in years
        weight : float
            Weight in kg
        physique_goal : str
            Goal (e.g., 'weight_loss', 'muscle_gain')
            
        Returns:
        --------
        dict
            Target micronutrient intake values
        """
        # Base RDI values (simplified)
        base_micros = {
            'vitamin_a': 900 if gender == 'male' else 700,  # mcg
            'vitamin_c': 90 if gender == 'male' else 75,    # mg
            'vitamin_d': 15,                                # mcg
            'vitamin_e': 15,                                # mg
            'vitamin_k': 120 if gender == 'male' else 90,   # mcg
            'thiamin': 1.2 if gender == 'male' else 1.1,    # mg
            'riboflavin': 1.3 if gender == 'male' else 1.1, # mg
            'niacin': 16 if gender == 'male' else 14,       # mg
            'vitamin_b6': 1.3,                              # mg
            'folate': 400,                                  # mcg
            'vitamin_b12': 2.4,                             # mcg
            'calcium': 1000,                                # mg
            'iron': 8 if gender == 'male' else 18,          # mg
            'magnesium': 400 if gender == 'male' else 310,  # mg
            'phosphorus': 700,                              # mg
            'potassium': 3400 if gender == 'male' else 2600,# mg
            'sodium': 2300,                                 # mg
            'zinc': 11 if gender == 'male' else 8           # mg
        }
        
        # Adjustments based on physique goal
        adjustments = {
            'weight_loss': {
                'vitamin_d': 1.2,  # Increased for metabolism support
                'calcium': 1.1,    # Bone health during caloric deficit
                'iron': 1.2,       # May decrease with reduced food intake
                'magnesium': 1.2,  # Supports metabolism
                'potassium': 1.1   # Electrolyte balance
            },
            'muscle_gain': {
                'vitamin_d': 1.2,  # Supports muscle function
                'zinc': 1.3,       # Supports protein synthesis
                'magnesium': 1.3,  # Muscle recovery
                'vitamin_b6': 1.2, # Protein metabolism
                'iron': 1.2        # Oxygen transport for recovery
            },
            'athletic_performance': {
                'vitamin_b6': 1.3,  # Energy metabolism
                'vitamin_b12': 1.3, # Red blood cell formation
                'iron': 1.3,        # Oxygen transport
                'magnesium': 1.3,   # Muscle function
                'potassium': 1.3    # Electrolyte balance
            },
            'toning': {
                'vitamin_d': 1.1,  # Muscle function
                'calcium': 1.1,    # Bone health
                'magnesium': 1.2   # Recovery
            }
        }
        
        # Apply goal-specific adjustments
        goal_adj = adjustments.get(physique_goal, {})
        for nutrient, multiplier in goal_adj.items():
            if nutrient in base_micros:
                base_micros[nutrient] *= multiplier
        
        # Apply age adjustments
        if age > 50:
            base_micros['vitamin_d'] *= 1.3  # Increased need
            base_micros['vitamin_b12'] *= 1.2  # Absorption decreases
            base_micros['calcium'] *= 1.2  # Bone health
        
        return {k: round(v, 1) for k, v in base_micros.items()}
    
    def generate_meal_plan(self, macros, micro_targets, food_preferences=None, meal_count=4, allergies=None):
        """
        Generate a personalized meal plan based on macro and micronutrient targets
        
        Parameters:
        -----------
        macros : dict
            Macronutrient targets
        micro_targets : dict
            Micronutrient targets
        food_preferences : list, optional
            List of preferred food items
        meal_count : int, optional
            Number of meals per day (default: 4)
        allergies : list, optional
            List of food allergies to avoid
            
        Returns:
        --------
        dict
            Meal plan with specific food recommendations
        """
        if allergies is None:
            allergies = []
        
        # Filter out allergies
        available_foods = self.food_data
        if allergies and 'food_name' in available_foods.columns:
            for allergy in allergies:
                available_foods = available_foods[~available_foods['food_name'].str.contains(allergy, case=False, na=False)]
        
        # Distribution of calories across meals
        meal_distribution = {
            1: [1.0],
            2: [0.4, 0.6],
            3: [0.3, 0.4, 0.3],
            4: [0.25, 0.35, 0.1, 0.3],
            5: [0.2, 0.25, 0.1, 0.25, 0.2],
            6: [0.15, 0.2, 0.15, 0.15, 0.15, 0.2]
        }
        
        # Get distribution based on meal count
        distribution = meal_distribution.get(meal_count, [1.0/meal_count] * meal_count)
        
        # Create a meal plan
        meal_plan = {
            f"meal_{i+1}": {
                "calories": round(macros['total_calories'] * distribution[i]),
                "protein": round(macros['protein']['grams'] * distribution[i]),
                "carbs": round(macros['carbs']['grams'] * distribution[i]),
                "fat": round(macros['fat']['grams'] * distribution[i]),
                "foods": self._recommend_foods_for_meal(
                    round(macros['protein']['grams'] * distribution[i]),
                    round(macros['carbs']['grams'] * distribution[i]),
                    round(macros['fat']['grams'] * distribution[i]),
                    available_foods
                )
            }
            for i in range(meal_count)
        }
        
        meal_plan['daily_totals'] = macros
        meal_plan['micronutrient_targets'] = micro_targets
        
        return meal_plan
    
    def _recommend_foods_for_meal(self, protein_target, carb_target, fat_target, available_foods):
        """
        Recommend specific foods for a meal based on macro targets
        
        This is a simplified implementation. A real system would use optimization
        to find the best combination of foods to meet the targets.
        
        Parameters:
        -----------
        protein_target : float
            Target protein in grams
        carb_target : float
            Target carbs in grams
        fat_target : float
            Target fat in grams
        available_foods : DataFrame
            Available food items
            
        Returns:
        --------
        list
            Recommended food items with portions
        """
        # This is a simplified approach - an actual implementation would use
        # optimization techniques to better match the targets
        
        # For the fallback dataset
        if 'food_name' in available_foods.columns:
            protein_rich = available_foods[available_foods['protein'] > 15].sample(min(2, len(available_foods[available_foods['protein'] > 15])))
            carb_rich = available_foods[available_foods['carbs'] > 15].sample(min(2, len(available_foods[available_foods['carbs'] > 15])))
            fat_rich = available_foods[available_foods['fat'] > 10].sample(min(1, len(available_foods[available_foods['fat'] > 10])))
            
            recommendations = []
            
            # Simple portion calculation - real system would be more sophisticated
            if not protein_rich.empty:
                for _, food in protein_rich.iterrows():
                    portion = min(2.0, protein_target / max(1, food['protein']))
                    recommendations.append({
                        "food": food['food_name'],
                        "portion": round(portion, 1),
                        "unit": "serving",
                        "nutrients": {
                            "calories": round(food['calories'] * portion),
                            "protein": round(food['protein'] * portion),
                            "carbs": round(food.get('carbs', 0) * portion),
                            "fat": round(food.get('fat', 0) * portion)
                        }
                    })
            
            if not carb_rich.empty:
                for _, food in carb_rich.iterrows():
                    portion = min(2.0, carb_target / max(1, food['carbs']))
                    recommendations.append({
                        "food": food['food_name'],
                        "portion": round(portion, 1),
                        "unit": "serving",
                        "nutrients": {
                            "calories": round(food['calories'] * portion),
                            "protein": round(food.get('protein', 0) * portion),
                            "carbs": round(food['carbs'] * portion),
                            "fat": round(food.get('fat', 0) * portion)
                        }
                    })
            
            if not fat_rich.empty:
                for _, food in fat_rich.iterrows():
                    portion = min(1.5, fat_target / max(1, food['fat']))
                    recommendations.append({
                        "food": food['food_name'],
                        "portion": round(portion, 1),
                        "unit": "serving",
                        "nutrients": {
                            "calories": round(food['calories'] * portion),
                            "protein": round(food.get('protein', 0) * portion),
                            "carbs": round(food.get('carbs', 0) * portion),
                            "fat": round(food['fat'] * portion)
                        }
                    })
        
            return recommendations
        
        # For NHANES dataset (more complex handling)
        else:
            # This would require parsing the complex NHANES food data structure
            # Just returning placeholder for now
            return [
                {"food": "Protein source", "portion": 1, "unit": "serving"},
                {"food": "Carb source", "portion": 1, "unit": "serving"},
                {"food": "Fat source", "portion": 0.5, "unit": "serving"},
                {"food": "Vegetable", "portion": 1, "unit": "cup"}
            ]
    
    def get_user_recommendations(self, user_data):
        """
        Generate comprehensive diet recommendations for a user
        
        Parameters:
        -----------
        user_data : dict
            User data including:
            - weight: Current weight in kg
            - height: Height in cm
            - age: Age in years
            - gender: 'male' or 'female'
            - goal_weight: Target weight in kg
            - physique_goal: One of the supported physique goals
            - training_type: One of the supported training types
            - activity_level: Activity level
            - food_preferences: Optional list of preferred foods
            - allergies: Optional list of food allergies
            - meal_count: Optional number of meals per day
            
        Returns:
        --------
        dict
            Comprehensive diet recommendations
        """
        # Extract user data
        weight = user_data.get('weight', 70)
        height = user_data.get('height', 170)
        age = user_data.get('age', 30)
        gender = user_data.get('gender', 'male')
        goal_weight = user_data.get('goal_weight', weight)
        physique_goal = user_data.get('physique_goal', 'maintenance')
        training_type = user_data.get('training_type', 'no_training')
        activity_level = user_data.get('activity_level', 'moderately_active')
        food_preferences = user_data.get('food_preferences', [])
        allergies = user_data.get('allergies', [])
        meal_count = user_data.get('meal_count', 4)
        
        # Calculate BMR
        bmr = user_data.get('bmr') or self.calculate_bmr(weight, height, age, gender)
        
        # Calculate TDEE
        tdee = user_data.get('total_calories') or self.calculate_tdee(bmr, activity_level)
        
        # Calculate macros
        macros = user_data.get('macros') or self.calculate_macros(tdee, training_type, physique_goal)
        
        # Get micronutrient targets
        micro_targets = self.get_micronutrient_targets(gender, age, weight, physique_goal)
        
        # Generate meal plan
        meal_plan = self.generate_meal_plan(
            macros, 
            micro_targets, 
            food_preferences=food_preferences,
            meal_count=meal_count,
            allergies=allergies
        )
        
        # Calculate estimated time to reach goal weight
        weight_diff = goal_weight - weight
        daily_cal_diff = tdee - macros['total_calories']
        
        if abs(daily_cal_diff) > 0 and weight_diff != 0:
            # 7700 calories roughly equals 1 kg of body weight
            days_to_goal = abs(weight_diff * 7700 / daily_cal_diff)
            weeks_to_goal = days_to_goal / 7
        else:
            days_to_goal = 0
            weeks_to_goal = 0
            
        # Prepare comprehensive recommendations
        recommendations = {
            'user_stats': {
                'current_weight': weight,
                'height': height,
                'age': age,
                'gender': gender,
                'goal_weight': goal_weight,
                'bmr': round(bmr),
                'tdee': round(tdee),
                'weight_difference': round(weight_diff, 1),
                'estimated_days_to_goal': round(days_to_goal) if days_to_goal > 0 else "N/A",
                'estimated_weeks_to_goal': round(weeks_to_goal, 1) if weeks_to_goal > 0 else "N/A"
            },
            'diet_plan': {
                'physique_goal': physique_goal,
                'training_type': training_type,
                'activity_level': activity_level,
                'daily_calories': macros['total_calories'],
                'macronutrients': {
                    'protein': macros['protein'],
                    'carbs': macros['carbs'],
                    'fat': macros['fat']
                },
                'micronutrient_targets': micro_targets,
                'meal_plan': meal_plan
            },
            'recommendations': self.generate_recommendations(weight_diff, physique_goal, training_type)
        }
        
        return recommendations
    
    def generate_recommendations(self, weight_diff, physique_goal, training_type):
        """
        Generate personalized recommendations based on goals
        
        Parameters:
        -----------
        weight_diff : float
            Difference between current and goal weight
        physique_goal : str
            User's physique goal
        training_type : str
            User's training type
            
        Returns:
        --------
        dict
            Personalized recommendations
        """
        recommendations = {
            'general': [],
            'nutrition': [],
            'training': [],
            'supplements': []
        }
        
        # General recommendations
        recommendations['general'].append("Stay hydrated by drinking at least 2-3 liters of water daily")
        recommendations['general'].append("Aim for 7-9 hours of quality sleep each night for optimal recovery")
        recommendations['general'].append("Track your progress weekly by measuring weight and body composition")
        
        # Goal-specific recommendations
        if weight_diff < -2:  # Weight loss
            recommendations['nutrition'].append("Focus on protein intake to preserve muscle mass during caloric deficit")
            recommendations['nutrition'].append("Emphasize fiber-rich foods to increase satiety during caloric restriction")
            recommendations['nutrition'].append("Consider time-restricted eating (like 16:8 intermittent fasting) if it suits your schedule")
            recommendations['training'].append("Include both resistance training and cardio for optimal fat loss")
            recommendations['supplements'].append("Consider a protein supplement to help meet daily protein targets")
            
        elif weight_diff > 2:  # Weight gain
            recommendations['nutrition'].append("Eat calorie-dense foods like nuts, avocados, and olive oil")
            recommendations['nutrition'].append("Consume protein-rich meals every 3-4 hours for optimal muscle protein synthesis")
            recommendations['nutrition'].append("Consider liquid calories (smoothies) if struggling to meet calorie needs")
            recommendations['training'].append("Focus on progressive overload in resistance training")
            recommendations['supplements'].append("Protein and creatine supplementation may support muscle development")
            
        else:  # Maintenance
            recommendations['nutrition'].append("Focus on food quality and nutrient density rather than strict calorie counting")
            recommendations['nutrition'].append("Maintain consistent meal timing to support metabolic health")
            recommendations['training'].append("Incorporate variety in your training to prevent plateaus")
            recommendations['supplements'].append("A high-quality multivitamin may help fill micronutrient gaps in your diet")
            
        # Training-specific recommendations
        if training_type == 'weightlifting' or training_type == 'bodybuilding':
            recommendations['training'].append("Ensure adequate recovery between training muscle groups (48-72 hours)")
            recommendations['training'].append("Focus on compound movements for maximum growth stimulus")
            recommendations['nutrition'].append("Time protein intake around your workouts for optimal recovery")
            
        elif training_type == 'cardio' or training_type == 'endurance':
            recommendations['training'].append("Include adequate warm-up and cool-down in each session")
            recommendations['training'].append("Consider low-intensity active recovery between harder sessions")
            recommendations['nutrition'].append("Prioritize carbohydrate intake before and after longer sessions")
            recommendations['nutrition'].append("Consider electrolyte replacement for sessions longer than 60 minutes")
            
        elif training_type == 'crossfit':
            recommendations['training'].append("Balance high-intensity days with recovery workouts")
            recommendations['training'].append("Pay special attention to mobility and recovery techniques")
            recommendations['nutrition'].append("Focus on post-workout nutrition with both protein and carbs")
            
        return recommendations


# User interface for the diet recommendation system
class DietRecommendationApp:
    """
    Interactive interface for the diet recommendation system
    """
    
    def __init__(self, dataset_path):
        """
        Initialize the diet recommendation app
        
        Parameters:
        -----------
        dataset_path : str
            Path to the NHANES dataset files
        """
        self.recommendation_system = DietRecommendationSystem(dataset_path)
        
    def collect_user_data(self):
        """Collect user data through console input"""
        print("\n===== Personalized Diet Recommendation System =====\n")
        print("Please enter your information to receive customized recommendations.")
        
        user_data = {}
        
        # Basic information
        try:
            user_data['weight'] = float(input("Current weight (kg): "))
            user_data['height'] = float(input("Height (cm): "))
            user_data['age'] = int(input("Age (years): "))
            
            gender = input("Gender (male/female): ").lower()
            user_data['gender'] = gender if gender in ['male', 'female'] else 'male'
            
            user_data['goal_weight'] = float(input("Goal weight (kg): "))
            
            # Physique goals
            print("\nPhysique goals:")
            print("1. Weight loss")
            print("2. Maintenance")
            print("3. Muscle gain")
            print("4. Toning")
            print("5. Athletic performance")
            goal_choice = int(input("Select your primary goal (1-5): "))
            
            goal_map = {
                1: 'weight_loss', 
                2: 'maintenance', 
                3: 'muscle_gain',
                4: 'toning',
                5: 'athletic_performance'
            }
            user_data['physique_goal'] = goal_map.get(goal_choice, 'maintenance')
            
            # Training type
            print("\nTraining type:")
            print("1. Cardio focused")
            print("2. Weightlifting")
            print("3. Bodybuilding")
            print("4. Endurance training")
            print("5. CrossFit")
            print("6. Yoga/Flexibility")
            print("7. No specific training")
            training_choice = int(input("Select your primary training type (1-7): "))
            
            training_map = {
                1: 'cardio',
                2: 'weightlifting',
                3: 'bodybuilding',
                4: 'endurance',
                5: 'crossfit',
                6: 'yoga',
                7: 'no_training'
            }
            user_data['training_type'] = training_map.get(training_choice, 'no_training')
            
            # Activity level
            print("\nActivity level:")
            print("1. Sedentary (little or no exercise)")
            print("2. Lightly active (light exercise/sports 1-3 days/week)")
            print("3. Moderately active (moderate exercise/sports 3-5 days/week)")
            print("4. Very active (hard exercise/sports 6-7 days/week)")
            print("5. Extra active (very hard exercise & physical job or training twice/day)")
            activity_choice = int(input("Select your activity level (1-5): "))
            
            activity_map = {
                1: 'sedentary',
                2: 'lightly_active',
                3: 'moderately_active',
                4: 'very_active',
                5: 'extra_active'
            }
            user_data['activity_level'] = activity_map.get(activity_choice, 'moderately_active')
            
            # Food preferences and allergies
            food_prefs = input("\nEnter any food preferences (comma separated): ")
            if food_prefs.strip():
                user_data['food_preferences'] = [p.strip() for p in food_prefs.split(',')]
            
            allergies = input("Enter any food allergies (comma separated): ")
            if allergies.strip():
                user_data['allergies'] = [a.strip() for a in allergies.split(',')]
            
            # Meal count
            user_data['meal_count'] = int(input("\nHow many meals per day do you prefer? (2-6): "))
            if user_data['meal_count'] < 2 or user_data['meal_count'] > 6:
                user_data['meal_count'] = 4
                print("Using default of 4 meals per day.")
                
        except ValueError:
            print("Error in input. Using default values.")
            # Default values if input fails
            user_data = {
                'weight': 70,
                'height': 170,
                'age': 30,
                'gender': 'male',
                'goal_weight': 70,
                'physique_goal': 'maintenance',
                'training_type': 'no_training',
                'activity_level': 'moderately_active',
                'meal_count': 4
            }
            
        return user_data
    
    def run(self):
        """Run the interactive diet recommendation app"""
        user_data = self.collect_user_data()
        recommendations = self.recommendation_system.get_user_recommendations(user_data)
        self.display_recommendations(recommendations)
    
    def display_recommendations(self, recommendations):
        """Display recommendations to the user"""
        print("\n\n=========================================")
        print("     PERSONALIZED DIET RECOMMENDATIONS     ")
        print("=========================================\n")
        
        # Display user stats
        stats = recommendations['user_stats']
        print("USER PROFILE:")
        print(f"• Current weight: {stats['current_weight']} kg")
        print(f"• Height: {stats['height']} cm")
        print(f"• Age: {stats['age']} years")
        print(f"• Gender: {stats['gender'].capitalize()}")
        print(f"• Goal weight: {stats['goal_weight']} kg ({stats['weight_difference']} kg difference)")
        print(f"• Basal Metabolic Rate (BMR): {stats['bmr']} calories/day")
        print(f"• Total Daily Energy Expenditure (TDEE): {stats['tdee']} calories/day")
        
        if stats['estimated_weeks_to_goal'] != "N/A":
            print(f"\nEstimated time to reach goal: {stats['estimated_weeks_to_goal']} weeks")
        
        # Display diet plan
        diet = recommendations['diet_plan']
        print("\nDIET PLAN:")
        print(f"• Goal: {diet['physique_goal'].replace('_', ' ').capitalize()}")
        print(f"• Training focus: {diet['training_type'].replace('_', ' ').capitalize()}")
        print(f"• Activity level: {diet['activity_level'].replace('_', ' ').capitalize()}")
        print(f"• Daily calorie target: {diet['daily_calories']} calories")
        
        # Display macros
        print("\nMACRONUTRIENT BREAKDOWN:")
        print(f"• Protein: {diet['macronutrients']['protein']['grams']}g " +
              f"({diet['macronutrients']['protein']['percentage']}%)")
        print(f"• Carbohydrates: {diet['macronutrients']['carbs']['grams']}g " +
              f"({diet['macronutrients']['carbs']['percentage']}%)")
        print(f"• Fat: {diet['macronutrients']['fat']['grams']}g " +
              f"({diet['macronutrients']['fat']['percentage']}%)")
        
        # Display meal plan
        print("\nMEAL PLAN:")
        meal_plan = diet['meal_plan']
        for meal_key, meal in meal_plan.items():
            if meal_key == 'daily_totals' or meal_key == 'micronutrient_targets':
                continue
                
            print(f"\n{meal_key.replace('_', ' ').title()} - {meal['calories']} calories " +
                  f"(P: {meal['protein']}g, C: {meal['carbs']}g, F: {meal['fat']}g)")
            
            if 'foods' in meal:
                for food in meal['foods']:
                    if isinstance(food, dict) and 'food' in food:
                        food_name = food['food'].replace('_', ' ').title()
                        portion = food.get('portion', 1)
                        unit = food.get('unit', 'serving')
                        
                        nutrients = food.get('nutrients', {})
                        if nutrients:
                            print(f"  • {food_name} - {portion} {unit} " +
                                  f"({nutrients.get('calories', '?')} cal, " +
                                  f"P: {nutrients.get('protein', '?')}g, " +
                                  f"C: {nutrients.get('carbs', '?')}g, " +
                                  f"F: {nutrients.get('fat', '?')}g)")
                        else:
                            print(f"  • {food_name} - {portion} {unit}")
        
        # Display recommendations
        rec = recommendations['recommendations']
        print("\nPERSONALIZED RECOMMENDATIONS:")
        
        print("\nGeneral Health:")
        for item in rec['general']:
            print(f"• {item}")
            
        print("\nNutrition:")
        for item in rec['nutrition']:
            print(f"• {item}")
            
        print("\nTraining:")
        for item in rec['training']:
            print(f"• {item}")
            
        print("\nSupplements to Consider:")
        for item in rec['supplements']:
            print(f"• {item}")
            
        print("\n=========================================")
        print("Remember: These recommendations are based on general principles and may need")
        print("adjustment based on individual response. Monitor progress and adjust as needed.")
        print("=========================================\n")


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        # Use the path from the initial code snippet
        # If running from a different script, replace with the actual path
        dataset_path = path if 'path' in globals() else "./dataset"
        
    print(f"Using dataset path: {dataset_path}")
    
    # Create an instance of the recommendation system
    recommendation_app = DietRecommendationApp(dataset_path)
    
    # Run the interactive console app
    recommendation_app.run()
    
    # Or to get recommendations programmatically:
    """
    # Create the recommendation system
    system = DietRecommendationSystem(dataset_path)
    
    # Example user data
    user_data = {
        'weight': 75,
        'height': 175,
        'age': 30,
        'gender': 'male',
        'goal_weight': 70,
        'physique_goal': 'weight_loss',
        'training_type': 'weightlifting',
        'activity_level': 'moderately_active',
        'food_preferences': ['chicken', 'rice', 'vegetables'],
        'allergies': ['nuts', 'shellfish'],
        'meal_count': 4
    }
    
    # Get recommendations
    recommendations = system.get_user_recommendations(user_data)
    
    # Process the recommendations as needed
    print(f"Daily calorie target: {recommendations['diet_plan']['daily_calories']}")
    """