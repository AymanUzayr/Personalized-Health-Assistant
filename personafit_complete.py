import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, IsolationForest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, classification_report, silhouette_score
import joblib
import kagglehub
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class PersonaFitMLPipeline:
    def __init__(self):
        """Initialize PersonaFit ML Pipeline for Streamlit deployment"""
        self.raw_data = None
        self.processed_data = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.model_performances = {}
        self.is_trained = False
        self.feature_columns = []
        
    # ==================== DATA LOADING & PREPROCESSING ====================
    
    def download_and_load_data(self):
        """Download NHANES dataset from Kaggle and load it"""
        try:
            print("📂 Downloading NHANES Dataset from Kaggle...")
            path = kagglehub.dataset_download("cdc/national-health-and-nutrition-examination-survey")
            
            # Find CSV files in the downloaded path
            csv_files = []
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith('.csv'):
                        csv_files.append(os.path.join(root, file))
            
            if not csv_files:
                raise FileNotFoundError("No CSV files found in the dataset")
            
            print(f"Found {len(csv_files)} CSV files")
            
            # Load the main dataset (usually the largest file)
            main_file = max(csv_files, key=lambda x: os.path.getsize(x))
            print(f"Loading main file: {os.path.basename(main_file)}")
            
            self.raw_data = pd.read_csv(main_file)
            
            print(f"✅ Dataset loaded successfully!")
            print(f"Shape: {self.raw_data.shape}")
            print(f"Columns: {len(self.raw_data.columns)}")
            
            return self.raw_data
            
        except Exception as e:
            print(f"❌ Error loading dataset: {str(e)}")
            # Fallback: create synthetic data for demo
            return self._create_demo_data()
    
    def _create_demo_data(self):
        """Create synthetic NHANES-like data for demo purposes"""
        print("🔄 Creating synthetic demo data...")
        
        np.random.seed(42)
        n_samples = 5000
        
        # Generate synthetic health data
        age = np.random.randint(18, 80, n_samples)
        gender = np.random.choice(['Male', 'Female'], n_samples)
        height = np.random.normal(170, 10, n_samples)  # cm
        weight = np.random.normal(70, 15, n_samples)   # kg
        
        # Calculate BMI
        bmi = weight / ((height/100) ** 2)
        
        # Generate correlated health metrics
        systolic_bp = 90 + (age * 0.5) + np.random.normal(0, 10, n_samples)
        diastolic_bp = 60 + (age * 0.3) + np.random.normal(0, 8, n_samples)
        cholesterol = 150 + (age * 1.2) + (bmi * 2) + np.random.normal(0, 30, n_samples)
        glucose = 80 + (age * 0.4) + (bmi * 1.5) + np.random.normal(0, 15, n_samples)
        
        # Lifestyle factors
        physical_activity = np.random.choice(['Low', 'Moderate', 'High'], n_samples, p=[0.4, 0.4, 0.2])
        smoking_status = np.random.choice(['Never', 'Former', 'Current'], n_samples, p=[0.6, 0.25, 0.15])
        education = np.random.choice(['Less than High School', 'High School', 'Some College', 'College+'], 
                                   n_samples, p=[0.2, 0.3, 0.3, 0.2])
        
        self.raw_data = pd.DataFrame({
            'age': age,
            'gender': gender,
            'height_cm': height,
            'weight_kg': weight,
            'bmi': bmi,
            'systolic_bp': systolic_bp,
            'diastolic_bp': diastolic_bp,
            'total_cholesterol': cholesterol,
            'glucose': glucose,
            'physical_activity': physical_activity,
            'smoking_status': smoking_status,
            'education_level': education
        })
        
        print(f"✅ Demo data created: {self.raw_data.shape}")
        return self.raw_data
    
    def explore_data(self):
        """Comprehensive data exploration"""
        if self.raw_data is None:
            print("❌ No data loaded. Please run download_and_load_data() first.")
            return None
            
        print("\n🔍 EXPLORATORY DATA ANALYSIS")
        print("=" * 50)
        
        # Basic info
        print("📊 Dataset Overview:")
        print(f"Total records: {len(self.raw_data):,}")
        print(f"Total features: {len(self.raw_data.columns)}")
        
        # Missing values analysis
        missing_percent = (self.raw_data.isnull().sum() / len(self.raw_data)) * 100
        missing_df = pd.DataFrame({
            'Column': self.raw_data.columns,
            'Missing_Count': self.raw_data.isnull().sum(),
            'Missing_Percentage': missing_percent
        }).sort_values('Missing_Percentage', ascending=False)
        
        print("\n❓ Top 10 Columns with Missing Values:")
        print(missing_df.head(10))
        
        # Data types
        print(f"\n📈 Data Types:")
        print(self.raw_data.dtypes.value_counts())
        
        return missing_df
    
    def clean_and_preprocess_data(self):
        """Complete data cleaning and preprocessing pipeline - OPTIMIZED VERSION"""
        if self.raw_data is None:
            print("❌ No data loaded. Please run download_and_load_data() first.")
            return None
            
        print("\n🧹 DATA CLEANING & PREPROCESSING PIPELINE")
        print("=" * 50)
        
        df = self.raw_data.copy()
        
        # 1. Remove completely empty rows/columns
        print("1️⃣ Removing empty rows and columns...")
        initial_shape = df.shape
        df = df.dropna(axis=0, how='all')
        df = df.dropna(axis=1, how='all')
        print(f"   Shape: {initial_shape} → {df.shape}")
        
        # 2. Handle extreme outliers
        print("2️⃣ Handling extreme outliers...")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df[col].notna().sum() > 0:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                
                outliers_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                if outliers_count > 0:
                    df[col] = np.where((df[col] < lower_bound) | (df[col] > upper_bound), 
                                     np.nan, df[col])
        
        # 3. OPTIMIZED missing value imputation
        print("3️⃣ Handling missing values...")
        
        # Separate numeric and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        # Check missing value percentage and data size
        total_samples = len(df)
        total_features = len(numeric_cols)
        
        print(f"   📊 Dataset size: {total_samples:,} rows, {total_features} numeric features")
        
        # SMART IMPUTATION STRATEGY based on data size and missing percentage
        if len(numeric_cols) > 0:
            missing_percentages = df[numeric_cols].isnull().sum() / len(df)
            high_missing_cols = missing_percentages[missing_percentages > 0.5].index
            
            if len(high_missing_cols) > 0:
                print(f"   ⚠️  Dropping {len(high_missing_cols)} columns with >50% missing values")
                df = df.drop(columns=high_missing_cols)
                numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            # Use different strategies based on dataset size
            if total_samples > 10000 or total_features > 20:
                print("   🔄 Using Simple Imputation (dataset too large for KNN)")
                # Use median/mean imputation for large datasets
                imputer = SimpleImputer(strategy='median')
                df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
            else:
                print("   🔄 Using KNN Imputation (small dataset)")
                try:
                    # Limit KNN neighbors based on dataset size
                    n_neighbors = min(5, max(2, int(total_samples * 0.01)))
                    imputer = KNNImputer(n_neighbors=n_neighbors)
                    
                    # Process in smaller chunks if needed
                    if total_samples > 5000:
                        chunk_size = 1000
                        imputed_chunks = []
                        
                        for i in range(0, total_samples, chunk_size):
                            chunk = df[numeric_cols].iloc[i:i+chunk_size]
                            if not chunk.empty:
                                imputed_chunk = imputer.fit_transform(chunk)
                                imputed_chunks.append(pd.DataFrame(imputed_chunk, 
                                                                 columns=numeric_cols, 
                                                                 index=chunk.index))
                        
                        df[numeric_cols] = pd.concat(imputed_chunks)
                    else:
                        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                        
                except Exception as e:
                    print(f"   ⚠️  KNN failed ({str(e)}), falling back to Simple Imputation")
                    imputer = SimpleImputer(strategy='median')
                    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        
        # Mode imputation for categorical data
        print("   🔄 Imputing categorical variables...")
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                mode_value = df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown'
                df[col] = df[col].fillna(mode_value)
        
        # 4. Feature engineering
        print("4️⃣ Creating PersonaFit features...")
        
        # BMI categories (if BMI exists or can be calculated)
        if 'bmi' in df.columns:
            df['BMI_Category'] = pd.cut(df['bmi'], 
                                      bins=[0, 18.5, 25, 30, 100], 
                                      labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
        elif 'height_cm' in df.columns and 'weight_kg' in df.columns:
            df['bmi'] = df['weight_kg'] / ((df['height_cm']/100) ** 2)
            df['BMI_Category'] = pd.cut(df['bmi'], 
                                      bins=[0, 18.5, 25, 30, 100], 
                                      labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
        
        # Age groups
        if 'age' in df.columns:
            df['Age_Group'] = pd.cut(df['age'], 
                                   bins=[0, 25, 35, 45, 55, 100], 
                                   labels=['Youth', 'Young_Adult', 'Adult', 'Middle_Age', 'Senior'])
        
        # Health risk score
        health_cols = [col for col in df.columns 
                      if any(keyword in col.lower() 
                           for keyword in ['cholesterol', 'bp', 'glucose', 'pressure'])]
        
        if len(health_cols) >= 2:
            health_data = df[health_cols].select_dtypes(include=[np.number])
            if not health_data.empty:
                scaler = MinMaxScaler()
                normalized_health = scaler.fit_transform(health_data.fillna(health_data.mean()))
                df['Health_Risk_Score'] = np.mean(normalized_health, axis=1)
        
        # Activity level mapping
        if 'physical_activity' in df.columns:
            df['Activity_Level'] = df['physical_activity'].map({
                'Low': 1, 'Moderate': 2, 'High': 3
            }).fillna(2)
        
        self.processed_data = df
        print(f"✅ Data preprocessing completed! Final shape: {df.shape}")
        
        return df
    
    # ==================== MACHINE LEARNING MODELS ====================
    
    def create_synthetic_targets(self):
        """Create realistic target variables for PersonaFit use cases"""
        if self.processed_data is None:
            print("❌ No processed data available. Run clean_and_preprocess_data() first.")
            return None
            
        print("\n🎯 Creating synthetic target variables...")
        
        df = self.processed_data.copy()
        
        # 1. Workout Difficulty Score (1-10 scale)
        if 'Age_Group' in df.columns and 'BMI_Category' in df.columns:
            age_factor = df['Age_Group'].map({
                'Youth': 1.2, 'Young_Adult': 1.0, 'Adult': 0.9, 
                'Middle_Age': 0.7, 'Senior': 0.5
            }).fillna(0.8)
            
            bmi_factor = df['BMI_Category'].map({
                'Underweight': 0.6, 'Normal': 1.0, 
                'Overweight': 0.7, 'Obese': 0.4
            }).fillna(0.8)
            
            noise = np.random.normal(0, 0.1, len(df))
            workout_difficulty = np.clip((age_factor * bmi_factor * 7) + noise, 1, 10)
            df['workout_difficulty'] = workout_difficulty
        
        # 2. Recommended Daily Calories
        if 'weight_kg' in df.columns:
            base_calories = df['weight_kg'] * 15
            if 'age' in df.columns:
                age_adjustment = (30 - df['age']) * 5
                base_calories += age_adjustment
            
            if 'Activity_Level' in df.columns:
                base_calories *= df['Activity_Level']
            
            df['daily_calories'] = np.clip(base_calories, 1200, 4000)
        
        # 3. Recovery Time (hours)
        if 'Health_Risk_Score' in df.columns:
            base_recovery = 24 + (df['Health_Risk_Score'] * 24)
            if 'Age_Group' in df.columns:
                age_recovery_factor = df['Age_Group'].map({
                    'Youth': 0.8, 'Young_Adult': 1.0, 'Adult': 1.2, 
                    'Middle_Age': 1.5, 'Senior': 2.0
                }).fillna(1.2)
                base_recovery *= age_recovery_factor
            
            df['recovery_hours'] = np.clip(base_recovery, 12, 72)
        
        # 4. Fitness Progress Score (0-100)
        consistency_factor = np.random.uniform(0.3, 1.0, len(df))
        genetic_factor = np.random.normal(0.7, 0.2, len(df))
        
        if 'BMI_Category' in df.columns:
            bmi_progress_factor = df['BMI_Category'].map({
                'Underweight': 0.8, 'Normal': 1.0, 
                'Overweight': 0.9, 'Obese': 0.7
            }).fillna(0.8)
            
            progress_score = (consistency_factor * genetic_factor * bmi_progress_factor * 100)
            df['fitness_progress_score'] = np.clip(progress_score, 10, 100)
        
        # 5. Diet Type Recommendation
        diet_types = ['Mediterranean', 'Low_Carb', 'High_Protein', 'Balanced', 'Plant_Based']
        
        if 'Health_Risk_Score' in df.columns and 'BMI_Category' in df.columns:
            diet_recommendations = []
            for _, row in df.iterrows():
                if row['Health_Risk_Score'] > 0.7:
                    diet_recommendations.append(np.random.choice(diet_types, p=[0.4, 0.3, 0.1, 0.1, 0.1]))
                elif row['BMI_Category'] == 'Obese':
                    diet_recommendations.append(np.random.choice(diet_types, p=[0.2, 0.4, 0.3, 0.1, 0.0]))
                else:
                    diet_recommendations.append(np.random.choice(diet_types, p=[0.2, 0.2, 0.2, 0.3, 0.1]))
            
            df['diet_type'] = diet_recommendations
        
        self.processed_data = df
        return df
    
    def train_all_models(self):
        """Train all PersonaFit ML models"""
        if self.processed_data is None:
            print("❌ No processed data available.")
            return False
            
        print("\n🤖 TRAINING PERSONAFIT ML MODELS")
        print("=" * 50)
        
        df = self.processed_data.copy()
        
        # Create synthetic targets if they don't exist
        if 'workout_difficulty' not in df.columns:
            df = self.create_synthetic_targets()
        
        # Prepare features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Encode categorical variables
        for col in categorical_cols:
            if col not in ['diet_type']:  # Skip target variables
                le = LabelEncoder()
                df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
                self.encoders[col] = le
                numeric_cols.append(col + '_encoded')
        
        # Store feature columns for prediction
        target_cols = ['workout_difficulty', 'daily_calories', 'recovery_hours', 'fitness_progress_score', 'diet_type']
        self.feature_columns = [col for col in numeric_cols if col not in target_cols]
        
        X = df[self.feature_columns].fillna(0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        self.scalers['main'] = self.scaler
        
        # Train individual models
        models_to_train = [
            ('workout_difficulty', 'Workout Difficulty', RandomForestRegressor(n_estimators=100, random_state=42)),
            ('daily_calories', 'Calorie Recommendation', RandomForestRegressor(n_estimators=100, random_state=42)),
            ('recovery_hours', 'Recovery Time', SVR(kernel='rbf', C=100)),
            ('fitness_progress_score', 'Progress Forecasting', GradientBoostingRegressor(n_estimators=100, random_state=42))
        ]
        
        for target_col, model_name, model in models_to_train:
            if target_col in df.columns:
                print(f"\n🏋️ Training {model_name} Model...")
                
                y = df[target_col]
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                print(f"  ✅ {model_name}: MSE={mse:.3f}, MAE={mae:.3f}")
                
                self.models[target_col] = model
                self.model_performances[target_col] = {'mse': mse, 'mae': mae}
        
        # Train diet type classifier
        if 'diet_type' in df.columns:
            print(f"\n🥗 Training Diet Type Classifier...")
            
            le_diet = LabelEncoder()
            y_diet = le_diet.fit_transform(df['diet_type'])
            self.encoders['diet_type'] = le_diet
            
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_diet, test_size=0.2, random_state=42)
            
            diet_model = RandomForestClassifier(n_estimators=100, random_state=42)
            diet_model.fit(X_train, y_train)
            
            y_pred = diet_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"  ✅ Diet Classifier Accuracy: {accuracy:.3f}")
            
            self.models['diet_type'] = diet_model
            self.model_performances['diet_type'] = {'accuracy': accuracy}
        
        # Train user clustering
        print(f"\n👥 Training User Clustering...")
        
        scaler_cluster = StandardScaler()
        X_cluster = scaler_cluster.fit_transform(X)
        self.scalers['clustering'] = scaler_cluster
        
        # Find optimal clusters
        silhouette_scores = []
        K_range = range(2, 8)
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_cluster)
            score = silhouette_score(X_cluster, cluster_labels)
            silhouette_scores.append(score)
        
        optimal_k = K_range[np.argmax(silhouette_scores)]
        
        kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        kmeans_final.fit(X_cluster)
        
        print(f"  ✅ User Clustering: {optimal_k} clusters, Silhouette Score: {max(silhouette_scores):.3f}")
        
        self.models['clustering'] = kmeans_final
        self.model_performances['clustering'] = {'n_clusters': optimal_k, 'silhouette_score': max(silhouette_scores)}
        
        self.is_trained = True
        print(f"\n🎉 All models trained successfully!")
        
        return True
    
    def predict_for_user(self, user_data):
        """Make predictions for a single user"""
        if not self.is_trained:
            print("❌ Models not trained yet. Run train_all_models() first.")
            return None
        
        try:
            # Prepare user data
            user_df = pd.DataFrame([user_data])
            
            # Encode categorical variables
            for col, encoder in self.encoders.items():
                if col in user_df.columns and col != 'diet_type':
                    user_df[col + '_encoded'] = encoder.transform([str(user_df[col].iloc[0])])[0]
            
            # Select features
            user_features = []
            for col in self.feature_columns:
                if col in user_df.columns:
                    user_features.append(user_df[col].iloc[0])
                else:
                    user_features.append(0)  # Default value for missing features
            
            user_features = np.array(user_features).reshape(1, -1)
            user_scaled = self.scalers['main'].transform(user_features)
            
            # Make predictions
            predictions = {}
            
            if 'workout_difficulty' in self.models:
                predictions['workout_difficulty'] = round(self.models['workout_difficulty'].predict(user_scaled)[0], 1)
            
            if 'daily_calories' in self.models:
                predictions['daily_calories'] = round(self.models['daily_calories'].predict(user_scaled)[0], 0)
            
            if 'recovery_hours' in self.models:
                predictions['recovery_hours'] = round(self.models['recovery_hours'].predict(user_scaled)[0], 1)
            
            if 'fitness_progress_score' in self.models:
                predictions['fitness_progress_score'] = round(self.models['fitness_progress_score'].predict(user_scaled)[0], 1)
            
            if 'diet_type' in self.models:
                diet_pred = self.models['diet_type'].predict(user_scaled)[0]
                predictions['recommended_diet'] = self.encoders['diet_type'].inverse_transform([diet_pred])[0]
            
            if 'clustering' in self.models:
                user_cluster_scaled = self.scalers['clustering'].transform(user_features)
                predictions['user_cluster'] = int(self.models['clustering'].predict(user_cluster_scaled)[0])
            
            return predictions
            
        except Exception as e:
            print(f"❌ Error making predictions: {str(e)}")
            return None
    
    def get_model_summary(self):
        """Get summary of all trained models"""
        if not self.is_trained:
            return "Models not trained yet."
        
        summary = {
            'total_models': len(self.models),
            'data_shape': self.processed_data.shape if self.processed_data is not None else (0, 0),
            'feature_count': len(self.feature_columns),
            'performance': self.model_performances
        }
        
        return summary
    
    def save_pipeline(self, filepath='personafit_pipeline.pkl'):
        """Save the entire pipeline"""
        if not self.is_trained:
            print("❌ Cannot save untrained pipeline.")
            return False
        
        try:
            pipeline_data = {
                'models': self.models,
                'scalers': self.scalers,
                'encoders': self.encoders,
                'feature_columns': self.feature_columns,
                'model_performances': self.model_performances
            }
            
            joblib.dump(pipeline_data, filepath)
            print(f"✅ Pipeline saved to {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving pipeline: {str(e)}")
            return False
    
    def load_pipeline(self, filepath='personafit_pipeline.pkl'):
        """Load a saved pipeline"""
        try:
            pipeline_data = joblib.load(filepath)
            
            self.models = pipeline_data['models']
            self.scalers = pipeline_data['scalers']
            self.encoders = pipeline_data['encoders']
            self.feature_columns = pipeline_data['feature_columns']
            self.model_performances = pipeline_data['model_performances']
            self.is_trained = True
            
            print(f"✅ Pipeline loaded from {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading pipeline: {str(e)}")
            return False

# ==================== STREAMLIT HELPER FUNCTIONS ====================

def initialize_personafit():
    """Initialize PersonaFit pipeline for Streamlit"""
    pipeline = PersonaFitMLPipeline()
    return pipeline

def run_complete_pipeline():
    """Run the complete PersonaFit pipeline"""
    print("🚀 PERSONAFIT COMPLETE ML PIPELINE")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = PersonaFitMLPipeline()
    
    # Download and load data
    pipeline.download_and_load_data()
    
    # Explore data
    pipeline.explore_data()
    
    # Clean and preprocess
    pipeline.clean_and_preprocess_data()
    
    # Train all models
    pipeline.train_all_models()
    
    # Save pipeline
    pipeline.save_pipeline()
    
    print("\n🎉 PersonaFit Pipeline Complete!")
    print("Ready for Streamlit deployment! 🚀")
    
    return pipeline

# Example usage for Streamlit
if __name__ == "__main__":
    # For standalone testing
    pipeline = run_complete_pipeline()
    
    # Example prediction
    sample_user = {
        'age': 30,
        'gender': 'Male',
        'height_cm': 175,
        'weight_kg': 70,
        'bmi': 22.9,
        'physical_activity': 'Moderate'
    }
    
    predictions = pipeline.predict_for_user(sample_user)
    print(f"\n🎯 Sample Predictions: {predictions}")