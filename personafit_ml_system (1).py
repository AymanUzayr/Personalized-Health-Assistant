import pandas as pd
import numpy as np
import requests
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class PersonaFITMLSystem:
    def __init__(self):
        self.workout_recommender = None
        self.progression_predictor = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.exercise_data = None
        self.equipment_normalizer = {
            'dumbbell': 'dumbbells',
            'dumbbells': 'dumbbells',
            'barbell': 'barbell',
            'body only': 'bodyweight',
            'bodyweight': 'bodyweight',
            'machine': 'machine',
            'cable': 'cable',
            'kettlebell': 'kettlebell',
            'resistance band': 'resistance_band',
            'medicine ball': 'medicine_ball',
            'foam roll': 'foam_roll',
            'e-z curl bar': 'barbell',
            'other': 'other'
        }
        
    def load_real_exercise_dataset(self):
        """Load and clean the Free Exercise Database from GitHub"""
        print("Loading Free Exercise Database from GitHub...")
        
        try:
            # Download the dataset
            url = "https://raw.githubusercontent.com/yuhonas/free-exercise-db/main/dist/exercises.json"
            response = requests.get(url)
            response.raise_for_status()
            
            exercises_data = response.json()
            print(f"Downloaded {len(exercises_data)} exercises")
            
            # Convert to DataFrame and clean
            self.exercise_data = self.clean_exercise_dataset(exercises_data)
            
            return self.exercise_data
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Falling back to sample dataset...")
            return self.create_sample_exercise_dataset()
    
    def clean_exercise_dataset(self, exercises_data):
        """Clean and normalize the exercise dataset"""
        print("Cleaning and normalizing exercise dataset...")
        
        cleaned_exercises = []
        seen_exercises = set()  # For duplicate removal
        
        for exercise in exercises_data:
            # Skip if essential fields are missing
            if not exercise.get('name') or not exercise.get('primaryMuscles'):
                continue
                
            # Create normalized exercise name for duplicate detection
            normalized_name = exercise['name'].lower().strip().replace(' ', '_')
            if normalized_name in seen_exercises:
                continue
            seen_exercises.add(normalized_name)
            
            # Clean and normalize the exercise data
            cleaned_exercise = {
                'id': exercise.get('id', normalized_name),
                'name': exercise['name'].strip(),
                'primaryMuscles': self.normalize_muscle_group(exercise['primaryMuscles']),
                'secondaryMuscles': self.normalize_secondary_muscles(exercise.get('secondaryMuscles', [])),
                'equipment': self.normalize_equipment(exercise.get('equipment', 'bodyweight')),
                'difficulty': self.normalize_difficulty(exercise.get('level', 'beginner')),
                'compound': self.determine_compound_movement(exercise),
                'category': exercise.get('category', 'strength'),
                'force': exercise.get('force', ''),
                'mechanic': exercise.get('mechanic', ''),
                'instructions': self.clean_instructions(exercise.get('instructions', []))
            }
            
            cleaned_exercises.append(cleaned_exercise)
        
        df = pd.DataFrame(cleaned_exercises)
        
        # Additional cleaning
        df = self.post_process_dataset(df)
        
        print(f"Cleaned dataset: {len(df)} exercises")
        print(f"Equipment types: {df['equipment'].value_counts().to_dict()}")
        print(f"Difficulty levels: {df['difficulty'].value_counts().to_dict()}")
        print(f"Primary muscles: {df['primaryMuscles'].value_counts().to_dict()}")
        
        return df
    
    def normalize_muscle_group(self, muscles):
        """Normalize muscle group names"""
        if isinstance(muscles, list):
            muscles = muscles[0] if muscles else 'other'
        
        muscle_mapping = {
            'biceps': 'biceps',
            'triceps': 'triceps',
            'chest': 'chest',
            'shoulders': 'shoulders',
            'back': 'back',
            'lats': 'back',
            'latissimus dorsi': 'back',
            'middle back': 'back',
            'lower back': 'back',
            'quadriceps': 'legs',
            'hamstrings': 'legs',
            'glutes': 'legs',
            'calves': 'legs',
            'abdominals': 'core',
            'abs': 'core',
            'core': 'core',
            'forearms': 'forearms',
            'traps': 'back',
            'trapezius': 'back'
        }
        
        muscle_lower = muscles.lower().strip()
        return muscle_mapping.get(muscle_lower, muscle_lower)
    
    def normalize_secondary_muscles(self, muscles):
        """Normalize secondary muscle groups"""
        if not muscles:
            return ""
        
        if isinstance(muscles, list):
            normalized = [self.normalize_muscle_group(m) for m in muscles]
            return ",".join(normalized)
        
        return self.normalize_muscle_group(muscles)
    
    def normalize_equipment(self, equipment):
        """Normalize equipment names"""
        if not equipment:
            return 'bodyweight'
        
        equipment_lower = equipment.lower().strip()
        return self.equipment_normalizer.get(equipment_lower, equipment_lower)
    
    def normalize_difficulty(self, level):
        """Normalize difficulty levels"""
        if not level:
            return 'beginner'
        
        level_mapping = {
            'beginner': 'beginner',
            'intermediate': 'intermediate', 
            'advanced': 'advanced',
            'expert': 'advanced'
        }
        
        level_lower = level.lower().strip()
        return level_mapping.get(level_lower, 'beginner')
    
    def determine_compound_movement(self, exercise):
        """Determine if exercise is compound based on secondary muscles and name"""
        # Check if has secondary muscles
        secondary_muscles = exercise.get('secondaryMuscles', [])
        if secondary_muscles and len(secondary_muscles) > 0:
            return True
        
        # Check exercise name for compound movement indicators
        compound_indicators = [
            'squat', 'deadlift', 'press', 'pull-up', 'chin-up', 'row',
            'lunge', 'clean', 'snatch', 'thruster', 'burpee'
        ]
        
        name_lower = exercise.get('name', '').lower()
        return any(indicator in name_lower for indicator in compound_indicators)
    
    def clean_instructions(self, instructions):
        """Clean and format exercise instructions"""
        if not instructions:
            return ""
        
        if isinstance(instructions, list):
            # Join instructions and clean
            cleaned = " ".join(instructions)
        else:
            cleaned = str(instructions)
        
        # Basic cleaning
        cleaned = cleaned.replace('\n', ' ').replace('\r', ' ')
        cleaned = ' '.join(cleaned.split())  # Remove extra whitespace
        
        return cleaned[:500]  # Limit length
    
    def post_process_dataset(self, df):
        """Additional post-processing of the dataset"""
        # Remove exercises with missing critical information
        df = df.dropna(subset=['name', 'primaryMuscles', 'equipment', 'difficulty'])
        
        # Add additional features
        df['has_instructions'] = df['instructions'].str.len() > 10
        df['muscle_count'] = df['secondaryMuscles'].str.count(',') + 1
        df['muscle_count'] = df['muscle_count'].fillna(0)
        
        # Create difficulty score
        difficulty_scores = {'beginner': 1, 'intermediate': 2, 'advanced': 3}
        df['difficulty_score'] = df['difficulty'].map(difficulty_scores)
        
        return df
    
    def create_enhanced_workout_logs(self, n_users=100, n_sessions_per_user=40):
        """Generate realistic workout logs using the real exercise dataset"""
        if self.exercise_data is None:
            self.load_real_exercise_dataset()
        
        np.random.seed(42)
        logs = []
        
        # Get exercises by difficulty and equipment
        exercises_by_difficulty = self.exercise_data.groupby('difficulty')['name'].apply(list).to_dict()
        exercises_by_equipment = self.exercise_data.groupby('equipment')['name'].apply(list).to_dict()
        
        for user_id in range(1, n_users + 1):
            # User characteristics
            experience_level = np.random.choice(['beginner', 'intermediate', 'advanced'], 
                                              p=[0.4, 0.4, 0.2])
            goal = np.random.choice(['muscle_gain', 'fat_loss', 'maintenance'], 
                                  p=[0.5, 0.3, 0.2])
            
            # Available equipment for this user
            all_equipment = list(exercises_by_equipment.keys())
            user_equipment = self.generate_user_equipment(experience_level)
            
            # Filter exercises available to this user
            available_exercises = self.exercise_data[
                (self.exercise_data['equipment'].isin(user_equipment)) &
                (self.exercise_data['difficulty'] == experience_level)
            ]['name'].tolist()
            
            # If too few exercises available, expand criteria
            if len(available_exercises) < 10:
                available_exercises = self.exercise_data[
                    self.exercise_data['equipment'].isin(user_equipment)
                ]['name'].tolist()
            
            if len(available_exercises) < 5:
                available_exercises = self.exercise_data['name'].tolist()
            
            # Starting date
            start_date = datetime.now() - timedelta(days=n_sessions_per_user * 3)
            
            # Track user's progression for each exercise
            user_exercise_progress = {}
            
            for session in range(n_sessions_per_user):
                session_date = start_date + timedelta(days=session * 3)
                
                # Select exercises for this session based on goal
                session_exercises = self.select_session_exercises(
                    available_exercises, goal, experience_level, session
                )
                
                for exercise in session_exercises:
                    # Get exercise details
                    exercise_info = self.exercise_data[
                        self.exercise_data['name'] == exercise
                    ].iloc[0]
                    
                    # Initialize or get current progress
                    if exercise not in user_exercise_progress:
                        base_weight = self._get_base_weight_realistic(exercise_info, experience_level)
                        base_reps = self._get_base_reps(goal, experience_level)
                        user_exercise_progress[exercise] = {
                            'weight': base_weight,
                            'reps': base_reps,
                            'sets': self._get_base_sets(goal, exercise_info['compound'])
                        }
                    
                    # Apply progression logic
                    current = user_exercise_progress[exercise]
                    
                    # Simulate progression with some randomness
                    progression_factor = self._get_progression_factor(experience_level, session)
                    fatigue_factor = np.random.uniform(0.9, 1.1)
                    
                    # Weight progression (every 4 sessions for compound, 6 for isolation)
                    progression_frequency = 4 if exercise_info['compound'] else 6
                    if session > 0 and session % progression_frequency == 0:
                        current['weight'] *= (1 + progression_factor * 0.05)
                    
                    # Rep progression
                    if session > 0 and np.random.random() < 0.3:
                        current['reps'] = min(current['reps'] + 1, 20)
                    
                    # Apply daily variation
                    actual_reps = max(int(current['reps'] * fatigue_factor), 
                                    max(current['reps'] - 3, 1))
                    actual_weight = max(current['weight'] * fatigue_factor, 0)
                    
                    # RPE calculation
                    target_reps = current['reps']
                    rpe = self._calculate_rpe_realistic(actual_reps, target_reps, 
                                                      experience_level, exercise_info)
                    
                    # Create log entry
                    log_entry = {
                        'user_id': user_id,
                        'date': session_date.strftime('%Y-%m-%d'),
                        'exercise_name': exercise,
                        'sets': current['sets'],
                        'reps': actual_reps,
                        'weight': round(actual_weight, 2),
                        'target_reps': target_reps,
                        'target_weight': round(current['weight'], 2),
                        'RPE': rpe,
                        'experience_level': experience_level,
                        'goal': goal,
                        'session_number': session + 1,
                        'days_since_last': 3 if session > 0 else 0,
                        'volume': actual_reps * current['sets'] * actual_weight,
                        'primary_muscle': exercise_info['primaryMuscles'],
                        'equipment': exercise_info['equipment'],
                        'is_compound': exercise_info['compound'],
                        'difficulty': exercise_info['difficulty']
                    }
                    
                    logs.append(log_entry)
        
        return pd.DataFrame(logs)
    
    def generate_user_equipment(self, experience_level):
        """Generate realistic equipment availability for user"""
        base_equipment = ['bodyweight']
        
        if experience_level in ['intermediate', 'advanced']:
            base_equipment.extend(['dumbbells'])
            
        if experience_level == 'advanced':
            base_equipment.extend(['barbell', 'machine'])
            
        # Add random additional equipment
        additional_equipment = ['kettlebell', 'cable', 'resistance_band']
        n_additional = np.random.randint(0, len(additional_equipment))
        if n_additional > 0:
            base_equipment.extend(
                np.random.choice(additional_equipment, n_additional, replace=False)
            )
        
        return base_equipment
    
    def select_session_exercises(self, available_exercises, goal, experience_level, session):
        """Select exercises for a training session"""
        # Filter exercises based on primary muscle groups
        if self.exercise_data is not None:
            available_df = self.exercise_data[
                self.exercise_data['name'].isin(available_exercises)
            ]
            
            # Create balanced workout
            muscle_groups = ['chest', 'back', 'legs', 'shoulders']
            if goal == 'muscle_gain':
                muscle_groups.extend(['biceps', 'triceps'])
            if session % 3 == 0:  # Every third session include core
                muscle_groups.append('core')
            
            selected_exercises = []
            for muscle in muscle_groups:
                muscle_exercises = available_df[
                    available_df['primaryMuscles'] == muscle
                ]['name'].tolist()
                
                if muscle_exercises:
                    selected_exercises.append(np.random.choice(muscle_exercises))
            
            # Add random exercises to reach target count
            target_count = {'beginner': 6, 'intermediate': 8, 'advanced': 10}[experience_level]
            while len(selected_exercises) < target_count and len(selected_exercises) < len(available_exercises):
                remaining = [ex for ex in available_exercises if ex not in selected_exercises]
                if remaining:
                    selected_exercises.append(np.random.choice(remaining))
                else:
                    break
            
            return selected_exercises
        
        # Fallback if no exercise data
        return np.random.choice(available_exercises, 
                               min(6, len(available_exercises)), 
                               replace=False).tolist()
    
    def _get_base_weight_realistic(self, exercise_info, experience_level):
        """Get realistic base weight based on exercise and experience"""
        equipment = exercise_info['equipment']
        is_compound = exercise_info['compound']
        muscle = exercise_info['primaryMuscles']
        
        if equipment == 'bodyweight':
            return 0
        
        # Base weights by experience level
        experience_multipliers = {
            'beginner': 0.7,
            'intermediate': 1.0,
            'advanced': 1.4
        }
        
        # Base weights by muscle group and exercise type
        base_weights = {
            'chest': {'compound': 60, 'isolation': 25},
            'back': {'compound': 70, 'isolation': 30},
            'legs': {'compound': 80, 'isolation': 35},
            'shoulders': {'compound': 40, 'isolation': 15},
            'biceps': {'compound': 30, 'isolation': 15},
            'triceps': {'compound': 35, 'isolation': 20},
            'core': {'compound': 20, 'isolation': 0}
        }
        
        exercise_type = 'compound' if is_compound else 'isolation'
        base_weight = base_weights.get(muscle, {'compound': 40, 'isolation': 20})[exercise_type]
        
        # Apply experience multiplier
        final_weight = base_weight * experience_multipliers[experience_level]
        
        # Add some randomness
        final_weight *= np.random.uniform(0.8, 1.2)
        
        return max(final_weight, 0)
    
    def _get_base_sets(self, goal, is_compound):
        """Get base number of sets"""
        if goal == 'muscle_gain':
            return 4 if is_compound else 3
        elif goal == 'fat_loss':
            return 3
        else:  # maintenance
            return 3
    
    def _calculate_rpe_realistic(self, actual_reps, target_reps, experience_level, exercise_info):
        """Calculate realistic RPE based on multiple factors"""
        performance_ratio = actual_reps / target_reps if target_reps > 0 else 1
        
        # Base RPE from performance
        if performance_ratio >= 1.1:
            base_rpe = np.random.uniform(6, 7)
        elif performance_ratio >= 1.0:
            base_rpe = np.random.uniform(7, 8)
        elif performance_ratio >= 0.9:
            base_rpe = np.random.uniform(8, 9)
        else:
            base_rpe = np.random.uniform(9, 10)
        
        # Adjust for exercise difficulty
        difficulty_adjustment = {
            'beginner': -0.5,
            'intermediate': 0,
            'advanced': 0.5
        }
        base_rpe += difficulty_adjustment[exercise_info['difficulty']]
        
        # Adjust for compound vs isolation
        if exercise_info['compound']:
            base_rpe += 0.3  # Compound exercises feel harder
        
        # Experience adjustment
        experience_adjustment = {'beginner': 0.5, 'intermediate': 0, 'advanced': -0.3}
        final_rpe = base_rpe + experience_adjustment[experience_level]
        
        return round(np.clip(final_rpe, 6, 10), 1)
    
    # Keep all the original methods for backward compatibility
    def create_sample_exercise_dataset(self):
        """Fallback method - create sample dataset if real data unavailable"""
        exercises = [
            {"name": "Push-ups", "primaryMuscles": "chest", "secondaryMuscles": "triceps,shoulders", 
             "equipment": "bodyweight", "difficulty": "beginner", "compound": True},
            {"name": "Bench Press", "primaryMuscles": "chest", "secondaryMuscles": "triceps,shoulders", 
             "equipment": "barbell", "difficulty": "intermediate", "compound": True},
            {"name": "Pull-ups", "primaryMuscles": "back", "secondaryMuscles": "biceps", 
             "equipment": "bodyweight", "difficulty": "intermediate", "compound": True},
            {"name": "Squats", "primaryMuscles": "legs", "secondaryMuscles": "glutes,core", 
             "equipment": "bodyweight", "difficulty": "beginner", "compound": True},
            {"name": "Deadlift", "primaryMuscles": "back", "secondaryMuscles": "hamstrings,glutes", 
             "equipment": "barbell", "difficulty": "advanced", "compound": True},
        ]
        
        self.exercise_data = pd.DataFrame(exercises)
        return self.exercise_data

    # Keep all original methods unchanged for compatibility
    def create_sample_workout_logs(self, n_users=50, n_sessions_per_user=30):
        """Original method kept for backward compatibility"""
        return self.create_enhanced_workout_logs(n_users, n_sessions_per_user)
    
    def _get_base_weight(self, exercise, experience_level):
        """Original method kept for backward compatibility"""
        weight_mapping = {
            'beginner': {'bodyweight': 0, 'light': 10, 'medium': 20, 'heavy': 40},
            'intermediate': {'bodyweight': 0, 'light': 15, 'medium': 35, 'heavy': 70},
            'advanced': {'bodyweight': 0, 'light': 25, 'medium': 50, 'heavy': 100}
        }
        
        if 'push-up' in exercise.lower() or 'plank' in exercise.lower():
            category = 'bodyweight'
        elif any(word in exercise.lower() for word in ['curl', 'raise', 'fly']):
            category = 'light'
        elif any(word in exercise.lower() for word in ['press', 'row', 'lunge']):
            category = 'medium'
        else:
            category = 'heavy'
            
        return weight_mapping[experience_level][category] * np.random.uniform(0.8, 1.2)
    
    def _get_base_reps(self, goal, experience_level):
        """Get base reps based on goal and experience"""
        rep_mapping = {
            'muscle_gain': {'beginner': 10, 'intermediate': 8, 'advanced': 6},
            'fat_loss': {'beginner': 15, 'intermediate': 12, 'advanced': 10},
            'maintenance': {'beginner': 12, 'intermediate': 10, 'advanced': 8}
        }
        return rep_mapping[goal][experience_level]
    
    def _get_progression_factor(self, experience_level, session):
        """Get progression factor based on experience and session"""
        base_factors = {'beginner': 0.8, 'intermediate': 0.6, 'advanced': 0.4}
        decay_factor = 0.99 ** session
        return base_factors[experience_level] * decay_factor
    
    def _calculate_rpe(self, actual_reps, target_reps, experience_level):
        """Original RPE calculation method"""
        performance_ratio = actual_reps / target_reps if target_reps > 0 else 1
        
        if performance_ratio >= 1.1:
            base_rpe = np.random.uniform(6, 7)
        elif performance_ratio >= 1.0:
            base_rpe = np.random.uniform(7, 8)
        elif performance_ratio >= 0.9:
            base_rpe = np.random.uniform(8, 9)
        else:
            base_rpe = np.random.uniform(9, 10)
            
        experience_adjustment = {'beginner': 0.5, 'intermediate': 0, 'advanced': -0.5}
        final_rpe = base_rpe + experience_adjustment[experience_level]
        
        return round(np.clip(final_rpe, 6, 10), 1)
    
    
    def save_models(self, directory):
        """Save trained models and scalers to specified directory"""
        import os
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Save each progression model
        for target, model in self.progression_predictor.items():
            with open(os.path.join(directory, f"{target}_model.pkl"), 'wb') as f:
                pickle.dump(model, f)

        # Save scaler and label encoders
        with open(os.path.join(directory, "scaler.pkl"), 'wb') as f:
            pickle.dump(self.scaler, f)

        with open(os.path.join(directory, "label_encoders.pkl"), 'wb') as f:
            pickle.dump(self.label_encoders, f)

        print(f"Models and preprocessors saved to '{directory}'")

# Keep all other original methods unchanged...
    def prepare_features(self, df):
        """Prepare features for ML model"""
        df = df.sort_values(['user_id', 'date', 'exercise_name'])
        
        df['date'] = pd.to_datetime(df['date'])
        
        # Create lag features
        df['prev_weight'] = df.groupby(['user_id', 'exercise_name'])['weight'].shift(1)
        df['prev_reps'] = df.groupby(['user_id', 'exercise_name'])['reps'].shift(1)
        df['prev_volume'] = df.groupby(['user_id', 'exercise_name'])['volume'].shift(1)
        df['prev_rpe'] = df.groupby(['user_id', 'exercise_name'])['RPE'].shift(1)
        
        # Performance ratios
        df['weight_change'] = (df['weight'] - df['prev_weight']) / df['prev_weight'].fillna(1)
        df['reps_change'] = (df['reps'] - df['prev_reps']) / df['prev_reps'].fillna(1)
        df['volume_change'] = (df['volume'] - df['prev_volume']) / df['prev_volume'].fillna(1)
        
        # Rolling averages
        for col in ['weight', 'reps', 'RPE', 'volume']:
            df[f'{col}_rolling_3'] = (
                df.groupby(['user_id', 'exercise_name'])[col]
                .rolling(3, min_periods=1)
                .mean()
                .reset_index(level=[0, 1], drop=True)
            )
        
        # Days since last session
        df['days_since_exercise'] = df.groupby(['user_id', 'exercise_name'])['date'].diff().dt.days.fillna(0)
        
        # Encode categorical variables
        categorical_cols = ['exercise_name', 'experience_level', 'goal']
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col])
            else:
                df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col])
        
        return df
    
    def train_progression_model(self, df):
        """Train the progression prediction model"""
        print("Training progression prediction model...")
        
        df = self.prepare_features(df)
        df = df.dropna(subset=['prev_weight', 'prev_reps'])
        
        feature_cols = [
            'prev_weight', 'prev_reps', 'prev_volume', 'prev_rpe',
            'weight_rolling_3', 'reps_rolling_3', 'RPE_rolling_3', 'volume_rolling_3',
            'session_number', 'days_since_exercise',
            'exercise_name_encoded', 'experience_level_encoded', 'goal_encoded',
            'weight_change', 'reps_change', 'volume_change'
        ]
        
        targets = ['weight', 'reps', 'RPE']
        X = df[feature_cols].fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        
        self.progression_predictor = {}
        
        for target in targets:
            print(f"Training model for {target}...")
            
            y = df[target]
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            model = RandomForestRegressor(
                n_estimators=100, max_depth=10, min_samples_split=5,
                min_samples_leaf=2, random_state=42, n_jobs=-1
            )
            
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            print(f"{target} - MSE: {mse:.4f}, R²: {r2:.4f}")
            self.progression_predictor[target] = model
        
        self.feature_cols = feature_cols
        print("Progression model training completed!")
        return self.progression_predictor

# ... [Keep all other original methods unchanged] ...

def main():
    """Enhanced main training function with real exercise data"""
    print("=== PersonaFIT ML System Training (Enhanced) ===")
    
    system = PersonaFITMLSystem()
    
    print("\n1. Loading real exercise dataset...")
    exercise_df = system.load_real_exercise_dataset()
    print(f"Loaded {len(exercise_df)} exercises")
    
    print("\n2. Generating enhanced workout logs...")
    workout_logs = system.create_enhanced_workout_logs(n_users=100, n_sessions_per_user=40)
    print(f"Generated {len(workout_logs)} workout log entries")
    
    # Save enhanced data
    workout_logs.to_csv("enhanced_workout_logs.csv", index=False)
    exercise_df.to_csv("cleaned_exercise_data.csv", index=False)
    print("Enhanced data saved to CSV files")
    
    print("\n3. Training progression prediction model...")
    system.train_progression_model(workout_logs)
    
    print("\n4. Saving models...")
    system.save_models("enhanced_models")
    
    print("\n=== Enhanced Training Complete ===")
    print("Real exercise dataset integrated successfully!")
    print(f"Dataset statistics:")
    print(f"- Total exercises: {len(exercise_df)}")
    print(f"- Equipment types: {exercise_df['equipment'].nunique()}")
    print(f"- Muscle groups: {exercise_df['primaryMuscles'].nunique()}")
    print(f"- Difficulty levels: {exercise_df['difficulty'].nunique()}")
    
    return system

if __name__ == "__main__":
    enhanced_system = main()