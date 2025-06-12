import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class PersonaFITPredictor:
    def __init__(self, model_path="models"):
        self.model_path = model_path
        self.progression_predictor = None
        self.workout_recommender = None
        self.scaler = None
        self.label_encoders = {}
        self.exercise_data = None
        self.feature_cols = []
        self.recommender_features = []
        
        # Load models
        self.load_models()
    
    def load_models(self):
        """Load all trained models"""
        try:
            # Load progression predictor models individually
            self.progression_predictor = {}
            for target in ['weight', 'reps', 'RPE']:
                with open(f"{self.model_path}/{target}_model.pkl", "rb") as f:
                    self.progression_predictor[target] = pickle.load(f)

            # Load workout recommender
            with open(f"{self.model_path}/workout_recommender.pkl", "rb") as f:
                self.workout_recommender = pickle.load(f)

            # Load scaler
            with open(f"{self.model_path}/scaler.pkl", "rb") as f:
                self.scaler = pickle.load(f)

            # Load label encoders
            with open(f"{self.model_path}/label_encoders.pkl", "rb") as f:
                self.label_encoders = pickle.load(f)

            # Load exercise data
            self.exercise_data = pd.read_csv("cleaned_exercise_data.csv")

            # Load feature metadata
            with open(f"{self.model_path}/metadata.pkl", "rb") as f:
                metadata = pickle.load(f)
                self.feature_cols = metadata.get('feature_cols', [])
                self.recommender_features = metadata.get('recommender_features', [])

            print("All models loaded successfully!")

        except Exception as e:
            print(f"Error loading models: {e}")
            print("Please run the training script first to generate the models.")

    def predict_next_session(self, user_workout_history, exercise_name):
        """
        Predict next session parameters for a specific exercise
        
        Args:
            user_workout_history: DataFrame with user's workout history
            exercise_name: Name of the exercise to predict
        
        Returns:
            Dictionary with predicted weight, reps, and RPE
        """
        if self.progression_predictor is None:
            return {"error": "Progression predictor not loaded"}
        
        # Filter history for the specific exercise
        exercise_history = user_workout_history[
            user_workout_history['exercise_name'] == exercise_name
        ].copy()
        
        if len(exercise_history) == 0:
            return {"error": f"No history found for exercise: {exercise_name}"}
        
        # Sort by date
        exercise_history = exercise_history.sort_values('date')
        
        # Get the most recent session
        latest_session = exercise_history.iloc[-1]
        
        # Prepare features for prediction
        try:
            # Create a temporary dataframe for feature engineering
            temp_df = exercise_history.copy()
            temp_df = self._prepare_prediction_features(temp_df)
            
            # Get the latest row with all features
            latest_features = temp_df.iloc[-1]
            
            # Extract feature values
            feature_values = []
            for col in self.feature_cols:
                if col in latest_features:
                    feature_values.append(latest_features[col])
                else:
                    feature_values.append(0)  # Default value for missing features
            
            # Scale features
            feature_array = np.array(feature_values).reshape(1, -1)
            scaled_features = self.scaler.transform(feature_array)
            
            # Make predictions
            predictions = {}
            for target in ['weight', 'reps', 'RPE']:
                if target in self.progression_predictor:
                    pred = self.progression_predictor[target].predict(scaled_features)[0]
                    predictions[target] = round(pred, 2)
            
            # Add some business logic constraints
            predictions = self._apply_progression_constraints(predictions, latest_session)
            
            return {
                "exercise": exercise_name,
                "predicted_weight": predictions.get('weight', latest_session['weight']),
                "predicted_reps": int(predictions.get('reps', latest_session['reps'])),
                "predicted_rpe": predictions.get('RPE', latest_session['RPE']),
                "last_session": {
                    "weight": latest_session['weight'],
                    "reps": latest_session['reps'],
                    "rpe": latest_session.get('RPE', 'N/A')
                },
                "recommendation": self._generate_progression_recommendation(predictions, latest_session)
            }
            
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}
    
    def _prepare_prediction_features(self, df):
        """Prepare features for prediction (similar to training)"""
        # Create lag features
        df['prev_weight'] = df['weight'].shift(1)
        df['prev_reps'] = df['reps'].shift(1)
        df['prev_volume'] = (df['weight'] * df['reps'] * df['sets']).shift(1)
        df['prev_rpe'] = df.get('RPE', 7).shift(1)
        
        # Performance ratios
        df['weight_change'] = (df['weight'] - df['prev_weight']) / df['prev_weight'].fillna(1)
        df['reps_change'] = (df['reps'] - df['prev_reps']) / df['prev_reps'].fillna(1)
        df['volume'] = df['weight'] * df['reps'] * df['sets']
        df['volume_change'] = (df['volume'] - df['prev_volume']) / df['prev_volume'].fillna(1)
        
        # Rolling averages (last 3 sessions)
        for col in ['weight', 'reps', 'volume']:
            df[f'{col}_rolling_3'] = df[col].rolling(3, min_periods=1).mean()
        
        df['RPE_rolling_3'] = df.get('RPE', 7).rolling(3, min_periods=1).mean()
        
        # Days since last session
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df['days_since_exercise'] = df['date'].diff().dt.days.fillna(0)
        else:
            df['days_since_exercise'] = 3  # Default
        
        # Session number
        df['session_number'] = range(1, len(df) + 1)
        
        # Encode categorical variables
        for col in ['exercise_name', 'experience_level', 'goal']:
            if col in df.columns and col in self.label_encoders:
                df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col])
            else:
                df[f'{col}_encoded'] = 0  # Default encoding
        
        return df.fillna(0)
    
    def _apply_progression_constraints(self, predictions, last_session):
        """Apply business logic constraints to predictions"""
        # Weight progression constraints
        if 'weight' in predictions:
            max_weight_increase = last_session['weight'] * 1.1  # Max 10% increase
            min_weight = last_session['weight'] * 0.9  # Min 10% decrease (deload)
            predictions['weight'] = np.clip(predictions['weight'], min_weight, max_weight_increase)
        
        # Reps constraints
        if 'reps' in predictions:
            predictions['reps'] = max(1, min(20, predictions['reps']))  # Between 1-20 reps
        
        # RPE constraints
        if 'RPE' in predictions:
            predictions['RPE'] = np.clip(predictions['RPE'], 6, 10)
        
        return predictions
    
    def _generate_progression_recommendation(self, predictions, last_session):
        """Generate human-readable progression recommendation"""
        recommendations = []
        
        # Weight recommendation
        if 'weight' in predictions:
            weight_change = predictions['weight'] - last_session['weight']
            if weight_change > 2:
                recommendations.append(f"Increase weight by {weight_change:.1f}kg")
            elif weight_change < -2:
                recommendations.append(f"Reduce weight by {abs(weight_change):.1f}kg (deload)")
            else:
                recommendations.append("Maintain current weight")
        
        # Reps recommendation
        if 'reps' in predictions:
            rep_change = predictions['reps'] - last_session['reps']
            if rep_change > 1:
                recommendations.append(f"Aim for {int(predictions['reps'])} reps")
            elif rep_change < -1:
                recommendations.append(f"Reduce to {int(predictions['reps'])} reps")
        
        # RPE guidance
        if 'RPE' in predictions:
            if predictions['RPE'] > 8.5:
                recommendations.append("High intensity - consider deload if struggling")
            elif predictions['RPE'] < 7:
                recommendations.append("Low intensity - room for progression")
        
        return "; ".join(recommendations) if recommendations else "Maintain current parameters"
    
    def recommend_workout_plan(self, user_profile, days_per_week=3):
        """
        Generate a complete workout plan for a user
        
        Args:
            user_profile: Dictionary with user info (goal, experience, equipment, etc.)
            days_per_week: Number of training days per week
        
        Returns:
            Complete workout plan
        """
        if self.workout_recommender is None:
            return {"error": "Workout recommender not loaded"}
        
        # Get suitable exercises
        suitable_exercises = self._get_suitable_exercises(user_profile)
        
        if len(suitable_exercises) == 0:
            return {"error": "No suitable exercises found for this profile"}
        
        # Generate workout split
        workout_plan = self._generate_workout_split(suitable_exercises, user_profile, days_per_week)
        
        return workout_plan
    
    def _get_suitable_exercises(self, user_profile):
        """Get exercises suitable for user profile using ML model"""
        if self.exercise_data is None:
            return pd.DataFrame()
        
        # Filter by equipment availability
        available_exercises = self.exercise_data[
            self.exercise_data['equipment'].isin(user_profile.get('equipment', []))
        ].copy()
        
        if len(available_exercises) == 0:
            return pd.DataFrame()
        
        # Prepare features for each exercise
        exercise_features = []
        
        for _, exercise in available_exercises.iterrows():
            features = {
                'goal_muscle_gain': 1 if user_profile.get('goal') == 'muscle_gain' else 0,
                'goal_fat_loss': 1 if user_profile.get('goal') == 'fat_loss' else 0,
                'goal_maintenance': 1 if user_profile.get('goal') == 'maintenance' else 0,
                'exp_beginner': 1 if user_profile.get('experience') == 'beginner' else 0,
                'exp_intermediate': 1 if user_profile.get('experience') == 'intermediate' else 0,
                'exp_advanced': 1 if user_profile.get('experience') == 'advanced' else 0,
                'has_equipment': 1 if exercise['equipment'] in user_profile.get('equipment', []) else 0,
                'is_compound': 1 if exercise['compound'] else 0,
                'difficulty_match': 1 if self._difficulty_matches(exercise['difficulty'], user_profile.get('experience')) else 0,
                'primary_muscle_chest': 1 if exercise['primaryMuscles'] == 'chest' else 0,
                'primary_muscle_back': 1 if exercise['primaryMuscles'] == 'back' else 0,
                'primary_muscle_legs': 1 if exercise['primaryMuscles'] == 'legs' else 0,
                'primary_muscle_shoulders': 1 if exercise['primaryMuscles'] == 'shoulders' else 0,
                'primary_muscle_arms': 1 if exercise['primaryMuscles'] in ['biceps', 'triceps'] else 0,
                'primary_muscle_core': 1 if exercise['primaryMuscles'] == 'core' else 0,
            }
            exercise_features.append(features)
        
        # Convert to DataFrame and predict suitability
        features_df = pd.DataFrame(exercise_features)
        
        # Ensure all required features are present
        for feature in self.recommender_features:
            if feature not in features_df.columns:
                features_df[feature] = 0
        
        # Predict suitability scores
        feature_matrix = features_df[self.recommender_features].values
        suitability_scores = self.workout_recommender.predict(feature_matrix)
        
        # Add scores to exercises
        available_exercises = available_exercises.copy()
        available_exercises['suitability_score'] = suitability_scores
        
        # Filter and sort by suitability
        suitable_exercises = available_exercises[
            available_exercises['suitability_score'] > 0.3
        ].sort_values('suitability_score', ascending=False)
        
        return suitable_exercises
    
    def _difficulty_matches(self, exercise_difficulty, user_experience):
        """Check if exercise difficulty matches user experience"""
        match_map = {
            'beginner': ['beginner'],
            'intermediate': ['beginner', 'intermediate'],
            'advanced': ['beginner', 'intermediate', 'advanced']
        }
        return exercise_difficulty in match_map.get(user_experience, [])
    
    def _generate_workout_split(self, suitable_exercises, user_profile, days_per_week):
        """Generate workout split based on days per week and user preferences"""
        split_type = user_profile.get('split_preference', 'auto')
        
        # Auto-determine split based on days per week
        if split_type == 'auto':
            if days_per_week <= 3:
                split_type = 'full_body'
            elif days_per_week == 4:
                split_type = 'upper_lower'
            else:
                split_type = 'push_pull_legs'
        
        if split_type == 'full_body':
            return self._generate_full_body_split(suitable_exercises, user_profile, days_per_week)
        elif split_type == 'upper_lower':
            return self._generate_upper_lower_split(suitable_exercises, user_profile)
        elif split_type == 'push_pull_legs':
            return self._generate_ppl_split(suitable_exercises, user_profile)
        else:
            return self._generate_full_body_split(suitable_exercises, user_profile, days_per_week)
    
    def _generate_full_body_split(self, exercises, profile, days):
        """Generate full body workout split"""
        # Group exercises by muscle group
        muscle_groups = ['chest', 'back', 'legs', 'shoulders', 'biceps', 'triceps', 'core']
        
        workout_plan = {
            'split_type': 'Full Body',
            'days_per_week': days,
            'workouts': []
        }
        
        for day in range(days):
            day_workout = {
                'day': f'Day {day + 1}',
                'exercises': []
            }
            
            # Select 1-2 exercises per major muscle group
            for muscle in ['chest', 'back', 'legs', 'shoulders']:
                muscle_exercises = exercises[exercises['primaryMuscles'] == muscle]
                if len(muscle_exercises) > 0:
                    selected = muscle_exercises.head(1 if muscle == 'legs' else 1)
                    for _, ex in selected.iterrows():
                        sets, reps = self._get_sets_reps(profile, ex)
                        day_workout['exercises'].append({
                            'name': ex['name'],
                            'muscle': ex['primaryMuscles'],
                            'sets': sets,
                            'reps': reps,
                            'equipment': ex['equipment']
                        })
            
            # Add some arms and core
            for muscle in ['biceps', 'core']:
                muscle_exercises = exercises[exercises['primaryMuscles'] == muscle]
                if len(muscle_exercises) > 0:
                    selected = muscle_exercises.head(1)
                    for _, ex in selected.iterrows():
                        sets, reps = self._get_sets_reps(profile, ex)
                        day_workout['exercises'].append({
                            'name': ex['name'],
                            'muscle': ex['primaryMuscles'],
                            'sets': sets,
                            'reps': reps,
                            'equipment': ex['equipment']
                        })
            
            workout_plan['workouts'].append(day_workout)
        
        return workout_plan
    
    def _generate_upper_lower_split(self, exercises, profile):
        """Generate upper/lower body split"""
        workout_plan = {
            'split_type': 'Upper/Lower',
            'days_per_week': 4,
            'workouts': []
        }
        
        # Upper body day
        upper_workout = {
            'day': 'Upper Body',
            'exercises': []
        }
        
        upper_muscles = ['chest', 'back', 'shoulders', 'biceps', 'triceps']
        for muscle in upper_muscles:
            muscle_exercises = exercises[exercises['primaryMuscles'] == muscle]
            if len(muscle_exercises) > 0:
                selected = muscle_exercises.head(2 if muscle in ['chest', 'back'] else 1)
                for _, ex in selected.iterrows():
                    sets, reps = self._get_sets_reps(profile, ex)
                    upper_workout['exercises'].append({
                        'name': ex['name'],
                        'muscle': ex['primaryMuscles'],
                        'sets': sets,
                        'reps': reps,
                        'equipment': ex['equipment']
                    })
        
        # Lower body day
        lower_workout = {
            'day': 'Lower Body',
            'exercises': []
        }
        
        lower_muscles = ['legs', 'hamstrings', 'glutes', 'core']
        for muscle in lower_muscles:
            muscle_exercises = exercises[exercises['primaryMuscles'] == muscle]
            if len(muscle_exercises) > 0:
                selected = muscle_exercises.head(2 if muscle == 'legs' else 1)
                for _, ex in selected.iterrows():
                    sets, reps = self._get_sets_reps(profile, ex)
                    lower_workout['exercises'].append({
                        'name': ex['name'],
                        'muscle': ex['primaryMuscles'],
                        'sets': sets,
                        'reps': reps,
                        'equipment': ex['equipment']
                    })
        
        workout_plan['workouts'] = [upper_workout, lower_workout]
        return workout_plan
    
    def _generate_ppl_split(self, exercises, profile):
        """Generate Push/Pull/Legs split"""
        workout_plan = {
            'split_type': 'Push/Pull/Legs',
            'days_per_week': 6,
            'workouts': []
        }
        
        # Push day (chest, shoulders, triceps)
        push_workout = {
            'day': 'Push',
            'exercises': []
        }
        
        push_muscles = ['chest', 'shoulders', 'triceps']
        for muscle in push_muscles:
            muscle_exercises = exercises[exercises['primaryMuscles'] == muscle]
            if len(muscle_exercises) > 0:
                selected = muscle_exercises.head(2 if muscle in ['chest', 'shoulders'] else 1)
                for _, ex in selected.iterrows():
                    sets, reps = self._get_sets_reps(profile, ex)
                    push_workout['exercises'].append({
                        'name': ex['name'],
                        'muscle': ex['primaryMuscles'],
                        'sets': sets,
                        'reps': reps,
                        'equipment': ex['equipment']
                    })
        
        # Pull day (back, biceps)
        pull_workout = {
            'day': 'Pull',
            'exercises': []
        }
        
        pull_muscles = ['back', 'biceps']
        for muscle in pull_muscles:
            muscle_exercises = exercises[exercises['primaryMuscles'] == muscle]
            if len(muscle_exercises) > 0:
                selected = muscle_exercises.head(2 if muscle == 'back' else 1)
                for _, ex in selected.iterrows():
                    sets, reps = self._get_sets_reps(profile, ex)
                    pull_workout['exercises'].append({
                        'name': ex['name'],
                        'muscle': ex['primaryMuscles'],
                        'sets': sets,
                        'reps': reps,
                        'equipment': ex['equipment']
                    })
        
        # Legs day
        legs_workout = {
            'day': 'Legs',
            'exercises': []
        }
        
        leg_muscles = ['legs', 'hamstrings', 'glutes', 'core']
        for muscle in leg_muscles:
            muscle_exercises = exercises[exercises['primaryMuscles'] == muscle]
            if len(muscle_exercises) > 0:
                selected = muscle_exercises.head(2 if muscle == 'legs' else 1)
                for _, ex in selected.iterrows():
                    sets, reps = self._get_sets_reps(profile, ex)
                    legs_workout['exercises'].append({
                        'name': ex['name'],
                        'muscle': ex['primaryMuscles'],
                        'sets': sets,
                        'reps': reps,
                        'equipment': ex['equipment']
                    })
        
        workout_plan['workouts'] = [push_workout, pull_workout, legs_workout]
        return workout_plan
    
    def _get_sets_reps(self, profile, exercise):
        """Get appropriate sets and reps based on profile and exercise"""
        goal = profile.get('goal', 'maintenance')
        experience = profile.get('experience', 'beginner')
        
        # Sets/reps mapping based on goal and experience
        mapping = {
            'muscle_gain': {
                'beginner': {'sets': 3, 'reps': '8-12'},
                'intermediate': {'sets': 4, 'reps': '6-10'},
                'advanced': {'sets': 4, 'reps': '6-8'}
            },
            'fat_loss': {
                'beginner': {'sets': 3, 'reps': '12-15'},
                'intermediate': {'sets': 3, 'reps': '12-20'},
                'advanced': {'sets': 4, 'reps': '15-20'}
            },
            'maintenance': {
                'beginner': {'sets': 2, 'reps': '10-12'},
                'intermediate': {'sets': 3, 'reps': '8-12'},
                'advanced': {'sets': 3, 'reps': '8-10'}
            }
        }
        
        config = mapping[goal][experience]
        
        # Adjust for compound vs isolation exercises
        if exercise.get('compound', False):
            sets = config['sets']
        else:
            sets = max(2, config['sets'] - 1)  # Fewer sets for isolation
        
        return sets, config['reps']

# Example usage and testing
def demo_usage():
    """Demonstrate how to use the PersonaFIT predictor"""
    print("=== PersonaFIT Predictor Demo ===")
    
    # Initialize predictor
    predictor = PersonaFITPredictor()
    
    # Example user profile
    user_profile = {
        'goal': 'muscle_gain',
        'experience': 'intermediate',
        'equipment': ['bodyweight', 'dumbbells'],
        'age': 25,
        'split_preference': 'auto'
    }
    
    print("\n1. Generating workout recommendation...")
    workout_plan = predictor.recommend_workout_plan(user_profile, days_per_week=3)
    
    if 'error' not in workout_plan:
        print(f"Generated {workout_plan['split_type']} split:")
        for i, workout in enumerate(workout_plan['workouts']):
            print(f"\n{workout['day']}:")
            for ex in workout['exercises']:
                print(f"  - {ex['name']}: {ex['sets']}x{ex['reps']} ({ex['muscle']})")
    else:
        print(f"Error: {workout_plan['error']}")
    
    # Example workout history for progression prediction
    print("\n2. Progression prediction example...")
    
    # Create sample workout history
    sample_history = pd.DataFrame([
        {'date': '2024-01-01', 'exercise_name': 'Push-ups', 'sets': 3, 'reps': 8, 'weight': 0, 'RPE': 7, 'experience_level': 'intermediate', 'goal': 'muscle_gain'},
        {'date': '2024-01-03', 'exercise_name': 'Push-ups', 'sets': 3, 'reps': 10, 'weight': 0, 'RPE': 8, 'experience_level': 'intermediate', 'goal': 'muscle_gain'},
        {'date': '2024-01-05', 'exercise_name': 'Push-ups', 'sets': 3, 'reps': 12, 'weight': 0, 'RPE': 8.5, 'experience_level': 'intermediate', 'goal': 'muscle_gain'},
    ])
    
    prediction = predictor.predict_next_session(sample_history, 'Push-ups')
    
    if 'error' not in prediction:
        print(f"Prediction for {prediction['exercise']}:")
        print(f"  Next session: {prediction['predicted_reps']} reps (RPE: {prediction['predicted_rpe']})")
        print(f"  Last session: {prediction['last_session']['reps']} reps")
        print(f"  Recommendation: {prediction['recommendation']}")
    else:
        print(f"Prediction error: {prediction['error']}")
    
    return predictor

if __name__ == "__main__":
    demo_usage()