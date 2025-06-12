import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, silhouette_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import warnings
warnings.filterwarnings('ignore')

class PersonaFitMLSuite:
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.model_performances = {}
        
    def create_synthetic_targets(self, df):
        """Create realistic target variables for PersonaFit use cases"""
        print("🎯 Creating synthetic target variables for PersonaFit models...")
        
        synthetic_targets = {}
        
        # 1. Workout Difficulty Score (0-10 scale)
        if 'Age_Group' in df.columns and 'BMI_Category' in df.columns:
            # Base difficulty on age and fitness level
            age_factor = df['Age_Group'].map({
                'Youth': 1.2, 'Young_Adult': 1.0, 'Adult': 0.9, 
                'Middle_Age': 0.7, 'Senior': 0.5
            }).fillna(0.8)
            
            bmi_factor = df['BMI_Category'].map({
                'Underweight': 0.6, 'Normal': 1.0, 
                'Overweight': 0.7, 'Obese': 0.4
            }).fillna(0.8)
            
            # Add some randomness for realism
            noise = np.random.normal(0, 0.1, len(df))
            workout_difficulty = np.clip(
                (age_factor * bmi_factor * 7) + noise, 1, 10
            )
            synthetic_targets['workout_difficulty'] = workout_difficulty
        
        # 2. Recommended Daily Calories
        if any(col for col in df.columns if 'weight' in col.lower()):
            weight_col = [col for col in df.columns if 'weight' in col.lower()][0]
            age_col = [col for col in df.columns if 'age' in col.lower()]
            
            # Harris-Benedict equation estimation
            base_calories = df[weight_col] * 15  # Simplified calculation
            if age_col:
                age_adjustment = (30 - df[age_col[0]]) * 5  # Adjust for age
                base_calories += age_adjustment
            
            # Add activity level adjustment
            if 'Activity_Level' in df.columns:
                activity_multiplier = df['Activity_Level'].map({
                    'Low': 1.2, 'Moderate': 1.5, 'High': 1.8
                }).fillna(1.4)
                base_calories *= activity_multiplier
            
            synthetic_targets['daily_calories'] = np.clip(base_calories, 1200, 4000)
        
        # 3. Recovery Time (hours needed between workouts)
        if 'Health_Risk_Score' in df.columns:
            # Higher health risk = longer recovery time
            base_recovery = 24 + (df['Health_Risk_Score'] * 24)  # 24-48 hours
            
            # Age adjustment
            if 'Age_Group' in df.columns:
                age_recovery_factor = df['Age_Group'].map({
                    'Youth': 0.8, 'Young_Adult': 1.0, 'Adult': 1.2, 
                    'Middle_Age': 1.5, 'Senior': 2.0
                }).fillna(1.2)
                base_recovery *= age_recovery_factor
            
            synthetic_targets['recovery_hours'] = np.clip(base_recovery, 12, 72)
        
        # 4. Fitness Progress Score (improvement potential 0-100)
        consistency_factor = np.random.uniform(0.3, 1.0, len(df))  # Simulated consistency
        genetic_factor = np.random.normal(0.7, 0.2, len(df))  # Simulated genetic potential
        
        if 'BMI_Category' in df.columns:
            bmi_progress_factor = df['BMI_Category'].map({
                'Underweight': 0.8, 'Normal': 1.0, 
                'Overweight': 0.9, 'Obese': 0.7
            }).fillna(0.8)
            
            progress_score = (consistency_factor * genetic_factor * bmi_progress_factor * 100)
            synthetic_targets['fitness_progress_score'] = np.clip(progress_score, 10, 100)
        
        # 5. Diet Type Recommendation (categorical)
        diet_types = ['Mediterranean', 'Low_Carb', 'High_Protein', 'Balanced', 'Plant_Based']
        
        if 'Health_Risk_Score' in df.columns and 'BMI_Category' in df.columns:
            # Logic-based diet assignment
            diet_probs = np.zeros((len(df), len(diet_types)))
            
            for i, (health_risk, bmi_cat) in enumerate(zip(df['Health_Risk_Score'], df['BMI_Category'])):
                if health_risk > 0.7:  # High health risk
                    diet_probs[i] = [0.4, 0.3, 0.1, 0.1, 0.1]  # Mediterranean preferred
                elif bmi_cat == 'Obese':
                    diet_probs[i] = [0.2, 0.4, 0.3, 0.1, 0.0]  # Low-carb/High-protein
                elif bmi_cat == 'Normal':
                    diet_probs[i] = [0.2, 0.2, 0.2, 0.3, 0.1]  # Balanced preferred
                else:
                    diet_probs[i] = [0.2, 0.2, 0.2, 0.2, 0.2]  # Equal probability
            
            diet_recommendations = []
            for prob_row in diet_probs:
                diet_recommendations.append(np.random.choice(diet_types, p=prob_row))
            
            synthetic_targets['diet_type'] = diet_recommendations
        
        return synthetic_targets
    
    def train_workout_difficulty_predictor(self, df, target_col='workout_difficulty'):
        """Train model to predict optimal workout difficulty"""
        print("\n🏋️ Training Workout Difficulty Predictor...")
        
        # Prepare features
        feature_cols = [col for col in df.columns if col not in [target_col]]
        X = df[feature_cols].select_dtypes(include=[np.number]).fillna(0)
        y = df[target_col]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train multiple models
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Neural Network': MLPRegressor(hidden_layer_sizes=(64, 32), random_state=42, max_iter=500)
        }
        
        best_model = None
        best_score = float('inf')
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            print(f"  {name}: MSE={mse:.3f}, MAE={mae:.3f}")
            
            if mse < best_score:
                best_score = mse
                best_model = model
        
        self.models['workout_difficulty'] = best_model
        self.model_performances['workout_difficulty'] = {'mse': best_score, 'features': list(X.columns)}
        
        # Feature importance
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            print(f"  Top features: {feature_importance.head(3)['feature'].tolist()}")
        
        return best_model
    
    def train_calorie_recommender(self, df, target_col='daily_calories'):
        """Train model to recommend daily calorie intake"""
        print("\n🍎 Training Calorie Recommender...")
        
        feature_cols = [col for col in df.columns if col not in [target_col]]
        X = df[feature_cols].select_dtypes(include=[np.number]).fillna(0)
        y = df[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Use ensemble approach
        model = RandomForestRegressor(n_estimators=150, random_state=42, max_depth=10)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"  Calorie Recommender: MSE={mse:.1f}, MAE={mae:.1f}")
        
        self.models['calorie_recommendation'] = model
        self.model_performances['calorie_recommendation'] = {'mse': mse, 'features': list(X.columns)}
        
        return model
    
    def train_recovery_predictor(self, df, target_col='recovery_hours'):
        """Train model to predict recovery time needed"""
        print("\n😴 Training Recovery Time Predictor...")
        
        feature_cols = [col for col in df.columns if col not in [target_col]]
        X = df[feature_cols].select_dtypes(include=[np.number]).fillna(0)
        y = df[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Use SVR for recovery prediction
        model = SVR(kernel='rbf', C=100, gamma='scale')
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"  Recovery Predictor: MSE={mse:.2f}, MAE={mae:.2f} hours")
        
        self.models['recovery_prediction'] = model
        self.model_performances['recovery_prediction'] = {'mse': mse, 'features': list(X.columns)}
        
        return model
    
    def train_progress_forecaster(self, df, target_col='fitness_progress_score'):
        """Train model to forecast fitness progress"""
        print("\n📈 Training Fitness Progress Forecaster...")
        
        feature_cols = [col for col in df.columns if col not in [target_col]]
        X = df[feature_cols].select_dtypes(include=[np.number]).fillna(0)
        y = df[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Gradient boosting for progress prediction
        model = GradientBoostingRegressor(
            n_estimators=200, 
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"  Progress Forecaster: MSE={mse:.2f}, MAE={mae:.2f}")
        
        self.models['progress_forecasting'] = model
        self.model_performances['progress_forecasting'] = {'mse': mse, 'features': list(X.columns)}
        
        return model
    
    def train_diet_type_classifier(self, df, target_col='diet_type'):
        """Train model to classify optimal diet type"""
        print("\n🥗 Training Diet Type Classifier...")
        
        feature_cols = [col for col in df.columns if col not in [target_col]]
        X = df[feature_cols].select_dtypes(include=[np.number]).fillna(0)
        y = df[target_col]
        
        # Encode target variable
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        self.encoders['diet_type'] = le
        
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
        
        # Random Forest Classifier
        model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"  Diet Classifier Accuracy: {accuracy:.3f}")
        
        self.models['diet_classification'] = model
        self.model_performances['diet_classification'] = {'accuracy': accuracy, 'features': list(X.columns)}
        
        return model
    
    def train_user_clustering(self, df):
        """Train clustering model for user segmentation"""
        print("\n👥 Training User Clustering Model...")
        
        # Select features for clustering
        cluster_features = df.select_dtypes(include=[np.number]).fillna(0)
        
        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(cluster_features)
        self.scalers['clustering'] = scaler
        
        # Determine optimal number of clusters
        silhouette_scores = []
        K_range = range(2, 8)
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(scaled_features)
            silhouette_avg = silhouette_score(scaled_features, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        
        optimal_k = K_range[np.argmax(silhouette_scores)]
        
        # Train final clustering model
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(scaled_features)
        
        print(f"  Optimal clusters: {optimal_k}")
        print(f"  Silhouette score: {max(silhouette_scores):.3f}")
        
        self.models['user_clustering'] = kmeans
        self.model_performances['user_clustering'] = {
            'n_clusters': optimal_k, 
            'silhouette_score': max(silhouette_scores),
            'features': list(cluster_features.columns)
        }
        
        return kmeans
    
    def create_ensemble_recommender(self, df):
        """Create ensemble model for comprehensive recommendations"""
        print("\n🎯 Creating Ensemble Recommendation System...")
        
        # Combine predictions from multiple models for better recommendations
        class PersonaFitEnsemble:
            def __init__(self, models, encoders, scalers):
                self.models = models
                self.encoders = encoders
                self.scalers = scalers
            
            def predict_comprehensive(self, user_data):
                """Make comprehensive predictions for a user"""
                predictions = {}
                
                # Prepare user data
                user_numeric = user_data.select_dtypes(include=[np.number]).fillna(0)
                
                # Individual predictions
                if 'workout_difficulty' in self.models:
                    predictions['workout_difficulty'] = self.models['workout_difficulty'].predict([user_numeric.iloc[0]])[0]
                
                if 'calorie_recommendation' in self.models:
                    predictions['daily_calories'] = self.models['calorie_recommendation'].predict([user_numeric.iloc[0]])[0]
                
                if 'recovery_prediction' in self.models:
                    predictions['recovery_hours'] = self.models['recovery_prediction'].predict([user_numeric.iloc[0]])[0]
                
                if 'progress_forecasting' in self.models:
                    predictions['progress_score'] = self.models['progress_forecasting'].predict([user_numeric.iloc[0]])[0]
                
                if 'diet_classification' in self.models:
                    diet_pred = self.models['diet_classification'].predict([user_numeric.iloc[0]])[0]
                    predictions['recommended_diet'] = self.encoders['diet_type'].inverse_transform([diet_pred])[0]
                
                if 'user_clustering' in self.models:
                    scaled_data = self.scalers['clustering'].transform([user_numeric.iloc[0]])
                    predictions['user_cluster'] = self.models['user_clustering'].predict(scaled_data)[0]
                
                return predictions
        
        ensemble = PersonaFitEnsemble(self.models, self.encoders, self.scalers)
        self.models['ensemble'] = ensemble
        
        print("  ✅ Ensemble recommender created!")
        return ensemble
    
    def save_models(self, save_path='personafit_models/'):
        """Save all trained models"""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        print(f"\n💾 Saving models to {save_path}...")
        
        # Save individual models
        for model_name, model in self.models.items():
            if model_name != 'ensemble':  # Skip ensemble for separate handling
                joblib.dump(model, f"{save_path}{model_name}_model.pkl")
        
        # Save encoders and scalers
        joblib.dump(self.encoders, f"{save_path}encoders.pkl")
        joblib.dump(self.scalers, f"{save_path}scalers.pkl")
        
        # Save performance metrics
        joblib.dump(self.model_performances, f"{save_path}model_performances.pkl")
        
        print("  ✅ All models saved successfully!")
    
    def generate_model_report(self):
        """Generate comprehensive model performance report"""
        print("\n📊 PERSONAFIT ML MODELS PERFORMANCE REPORT")
        print("=" * 60)
        
        for model_name, performance in self.model_performances.items():
            print(f"\n🔸 {model_name.upper().replace('_', ' ')}")
            for metric, value in performance.items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.4f}")
                elif isinstance(value, list):
                    print(f"  {metric}: {len(value)} features")
                else:
                    print(f"  {metric}: {value}")
        
        print(f"\n✅ Total models trained: {len(self.models)}")
        print("🚀 PersonaFit ML Suite is ready for deployment!")

# Main training pipeline
def train_personafit_models(preprocessor):
    """Complete training pipeline for PersonaFit"""
    print("🚀 STARTING PERSONAFIT ML TRAINING PIPELINE")
    print("=" * 60)
    
    # Get processed data
    df = preprocessor.processed_data.copy()
    
    # Initialize ML suite
    ml_suite = PersonaFitMLSuite(preprocessor)
    
    # Create synthetic targets
    synthetic_targets = ml_suite.create_synthetic_targets(df)
    
    # Add synthetic targets to dataframe
    for target_name, target_values in synthetic_targets.items():
        df[target_name] = target_values
    
    # Train all models
    ml_suite.train_workout_difficulty_predictor(df)
    ml_suite.train_calorie_recommender(df)
    ml_suite.train_recovery_predictor(df)
    ml_suite.train_progress_forecaster(df)
    ml_suite.train_diet_type_classifier(df)
    ml_suite.train_user_clustering(df)
    
    # Create ensemble
    ml_suite.create_ensemble_recommender(df)
    
    # Save models
    ml_suite.save_models()
    
    # Generate report
    ml_suite.generate_model_report()
    
    return ml_suite

# Demo prediction function
def demo_personafit_prediction(ml_suite, sample_user_data):
    """Demonstrate PersonaFit predictions for a sample user"""
    print("\n🧪 DEMO: PersonaFit Predictions for Sample User")
    print("=" * 50)
    
    if 'ensemble' in ml_suite.models:
        predictions = ml_suite.models['ensemble'].predict_comprehensive(sample_user_data)
        
        print("📋 Personalized Recommendations:")
        for pred_type, value in predictions.items():
            if isinstance(value, float):
                print(f"  • {pred_type.replace('_', ' ').title()}: {value:.2f}")
            else:
                print(f"  • {pred_type.replace('_', ' ').title()}: {value}")
    
    return predictions

# Usage example
if __name__ == "__main__":
    # Assuming you have the preprocessor from the previous code
    # ml_suite = train_personafit_models(preprocessor)
    
    # Create sample user for demo
    # sample_user = pd.DataFrame({
    #     'age': [28], 'weight': [70], 'height': [175], 
    #     'BMI_Category': ['Normal'], 'Activity_Level': ['Moderate']
    # })
    # predictions = demo_personafit_prediction(ml_suite, sample_user)
    
    print("🎉 PersonaFit ML Training Suite Ready!")
    print("Use: ml_suite = train_personafit_models(preprocessor) to start training!")