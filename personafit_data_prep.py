import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.impute import KNNImputer
import warnings
warnings.filterwarnings('ignore')

class PersonaFitDataPreprocessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.raw_data = None
        self.processed_data = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_dataset(self):
        """Load and explore the NHANES dataset"""
        print("📂 Loading NHANES Dataset...")
        
        # List all files in the dataset
        files = os.listdir(self.data_path)
        print(f"Available files: {files}")
        
        # Assuming the main file is CSV - adjust filename as needed
        main_file = [f for f in files if f.endswith('.csv')][0]
        self.raw_data = pd.read_csv(os.path.join(self.data_path, main_file))
        
        print(f"✅ Dataset loaded successfully!")
        print(f"Shape: {self.raw_data.shape}")
        print(f"Columns: {len(self.raw_data.columns)}")
        
        return self.raw_data
    
    def explore_data(self):
        """Comprehensive data exploration for PersonaFit features"""
        print("\n🔍 EXPLORATORY DATA ANALYSIS")
        print("=" * 50)
        
        # Basic info
        print("📊 Dataset Overview:")
        print(f"Total records: {len(self.raw_data):,}")
        print(f"Total features: {len(self.raw_data.columns)}")
        print(f"Memory usage: {self.raw_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Missing values analysis
        print("\n❓ Missing Values Analysis:")
        missing_percent = (self.raw_data.isnull().sum() / len(self.raw_data)) * 100
        missing_df = pd.DataFrame({
            'Column': self.raw_data.columns,
            'Missing_Count': self.raw_data.isnull().sum(),
            'Missing_Percentage': missing_percent
        }).sort_values('Missing_Percentage', ascending=False)
        
        print(missing_df.head(10))
        
        # Data types
        print(f"\n📈 Data Types Distribution:")
        print(self.raw_data.dtypes.value_counts())
        
        return missing_df
    
    def identify_personafit_features(self):
        """Identify relevant features for PersonaFit application"""
        print("\n🎯 IDENTIFYING PERSONAFIT-RELEVANT FEATURES")
        print("=" * 50)
        
        # Define feature categories for PersonaFit
        feature_categories = {
            'demographics': ['age', 'gender', 'race', 'education'],
            'physical_metrics': ['height', 'weight', 'bmi', 'waist', 'hip'],
            'health_indicators': ['blood_pressure', 'cholesterol', 'diabetes', 'heart_rate'],
            'activity_lifestyle': ['physical_activity', 'sedentary_time', 'sleep'],
            'nutrition': ['calories', 'protein', 'carbs', 'fat', 'fiber', 'sodium'],
            'lab_values': ['glucose', 'hemoglobin', 'vitamin_d'],
            'fitness_related': ['muscle_strength', 'flexibility', 'endurance']
        }
        
        # Search for relevant columns in the dataset
        relevant_features = {}
        all_columns = [col.lower() for col in self.raw_data.columns]
        
        for category, keywords in feature_categories.items():
            found_features = []
            for keyword in keywords:
                matching_cols = [col for col in self.raw_data.columns 
                               if keyword.lower() in col.lower()]
                found_features.extend(matching_cols)
            
            if found_features:
                relevant_features[category] = list(set(found_features))
        
        print("🔍 PersonaFit Relevant Features Found:")
        for category, features in relevant_features.items():
            print(f"\n{category.upper()}:")
            for feature in features[:5]:  # Show first 5
                print(f"  • {feature}")
            if len(features) > 5:
                print(f"  ... and {len(features)-5} more")
        
        return relevant_features
    
    def clean_data(self, target_features=None):
        """Comprehensive data cleaning pipeline"""
        print("\n🧹 DATA CLEANING PIPELINE")
        print("=" * 50)
        
        df = self.raw_data.copy()
        
        # 1. Remove completely empty rows/columns
        print("1️⃣ Removing empty rows and columns...")
        initial_shape = df.shape
        df = df.dropna(axis=0, how='all')  # Remove empty rows
        df = df.dropna(axis=1, how='all')  # Remove empty columns
        print(f"   Shape: {initial_shape} → {df.shape}")
        
        # 2. Handle extreme outliers using IQR method
        print("2️⃣ Handling extreme outliers...")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df[col].notna().sum() > 0:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR  # More conservative than 1.5
                upper_bound = Q3 + 3 * IQR
                
                outliers_before = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                df[col] = np.where((df[col] < lower_bound) | (df[col] > upper_bound), 
                                 np.nan, df[col])
                
                if outliers_before > 0:
                    print(f"   {col}: {outliers_before} outliers removed")
        
        # 3. Smart missing value imputation
        print("3️⃣ Intelligent missing value imputation...")
        
        # Separate numeric and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        # KNN imputation for numeric data (better than mean/median)
        if len(numeric_cols) > 0:
            knn_imputer = KNNImputer(n_neighbors=5)
            df[numeric_cols] = knn_imputer.fit_transform(df[numeric_cols])
        
        # Mode imputation for categorical data
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                mode_value = df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown'
                df[col] = df[col].fillna(mode_value)
        
        print(f"   ✅ Missing values resolved")
        
        # 4. Data type optimization
        print("4️⃣ Optimizing data types...")
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col], errors='ignore')
                except:
                    pass
        
        self.processed_data = df
        print(f"✅ Data cleaning completed! Final shape: {df.shape}")
        
        return df
    
    def feature_engineering(self):
        """Create PersonaFit-specific engineered features"""
        print("\n⚙️ FEATURE ENGINEERING FOR PERSONAFIT")
        print("=" * 50)
        
        df = self.processed_data.copy()
        
        # Create PersonaFit-specific features
        engineered_features = []
        
        # 1. BMI categories
        if 'BMI' in df.columns or any('bmi' in col.lower() for col in df.columns):
            bmi_col = [col for col in df.columns if 'bmi' in col.lower()][0]
            df['BMI_Category'] = pd.cut(df[bmi_col], 
                                      bins=[0, 18.5, 25, 30, 100], 
                                      labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
            engineered_features.append('BMI_Category')
        
        # 2. Age groups for fitness planning
        if 'age' in df.columns or any('age' in col.lower() for col in df.columns):
            age_col = [col for col in df.columns if 'age' in col.lower()][0]
            df['Age_Group'] = pd.cut(df[age_col], 
                                   bins=[0, 25, 35, 45, 55, 100], 
                                   labels=['Youth', 'Young_Adult', 'Adult', 'Middle_Age', 'Senior'])
            engineered_features.append('Age_Group')
        
        # 3. Health risk score (composite feature)
        health_indicators = [col for col in df.columns 
                           if any(keyword in col.lower() 
                                for keyword in ['cholesterol', 'blood_pressure', 'glucose', 'diabetes'])]
        
        if len(health_indicators) >= 2:
            # Normalize health indicators and create composite score
            health_data = df[health_indicators].select_dtypes(include=[np.number])
            if not health_data.empty:
                scaler = MinMaxScaler()
                normalized_health = scaler.fit_transform(health_data.fillna(health_data.mean()))
                df['Health_Risk_Score'] = np.mean(normalized_health, axis=1)
                engineered_features.append('Health_Risk_Score')
        
        # 4. Activity level classification
        activity_cols = [col for col in df.columns 
                        if any(keyword in col.lower() 
                             for keyword in ['physical', 'activity', 'exercise', 'sedentary'])]
        
        if activity_cols:
            # Create activity level feature based on available activity data
            activity_data = df[activity_cols].select_dtypes(include=[np.number])
            if not activity_data.empty:
                df['Activity_Level'] = pd.qcut(activity_data.mean(axis=1), 
                                             q=3, labels=['Low', 'Moderate', 'High'])
                engineered_features.append('Activity_Level')
        
        print("🔧 Engineered Features Created:")
        for feature in engineered_features:
            print(f"  • {feature}")
        
        self.processed_data = df
        return df
    
    def prepare_ml_features(self, target_column=None):
        """Prepare features for machine learning models"""
        print("\n🤖 PREPARING FEATURES FOR ML MODELS")
        print("=" * 50)
        
        df = self.processed_data.copy()
        
        # 1. Encode categorical variables
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            if col != target_column:
                le = LabelEncoder()
                df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
                print(f"  ✅ Encoded: {col}")
        
        # 2. Scale numerical features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if target_column and target_column in numeric_cols:
            numeric_cols = numeric_cols.drop(target_column)
        
        df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        print(f"  ✅ Scaled {len(numeric_cols)} numerical features")
        
        # 3. Feature selection based on importance
        print("  🎯 Feature importance analysis ready for model training")
        
        self.processed_data = df
        return df
    
    def create_personafit_datasets(self):
        """Create specific datasets for PersonaFit use cases"""
        print("\n📊 CREATING PERSONAFIT-SPECIFIC DATASETS")
        print("=" * 50)
        
        df = self.processed_data.copy()
        
        datasets = {}
        
        # 1. Fitness Recommendation Dataset
        fitness_features = [col for col in df.columns 
                          if any(keyword in col.lower() 
                               for keyword in ['age', 'bmi', 'weight', 'height', 'activity', 'physical'])]
        
        if len(fitness_features) >= 3:
            datasets['fitness_recommendation'] = df[fitness_features].copy()
            print(f"  🏃 Fitness Recommendation Dataset: {len(fitness_features)} features")
        
        # 2. Nutrition Planning Dataset
        nutrition_features = [col for col in df.columns 
                            if any(keyword in col.lower() 
                                 for keyword in ['calorie', 'protein', 'carb', 'fat', 'fiber', 'sodium', 'vitamin'])]
        
        if len(nutrition_features) >= 3:
            datasets['nutrition_planning'] = df[nutrition_features + ['Age_Group', 'BMI_Category']].copy()
            print(f"  🍎 Nutrition Planning Dataset: {len(nutrition_features)} features")
        
        # 3. Health Risk Assessment Dataset
        health_features = [col for col in df.columns 
                         if any(keyword in col.lower() 
                              for keyword in ['health', 'risk', 'cholesterol', 'blood', 'glucose', 'diabetes'])]
        
        if len(health_features) >= 3:
            datasets['health_assessment'] = df[health_features].copy()
            print(f"  ❤️ Health Assessment Dataset: {len(health_features)} features")
        
        return datasets
    
    def generate_data_summary(self):
        """Generate comprehensive data summary for PersonaFit"""
        print("\n📋 PERSONAFIT DATA SUMMARY REPORT")
        print("=" * 60)
        
        df = self.processed_data
        
        summary = {
            'total_records': len(df),
            'total_features': len(df.columns),
            'numeric_features': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_features': len(df.select_dtypes(include=['object', 'category']).columns),
            'missing_values': df.isnull().sum().sum(),
            'data_quality_score': ((df.notna().sum().sum()) / (len(df) * len(df.columns))) * 100
        }
        
        print(f"📊 Dataset Size: {summary['total_records']:,} records")
        print(f"🔢 Total Features: {summary['total_features']}")
        print(f"📈 Numeric Features: {summary['numeric_features']}")
        print(f"🏷️ Categorical Features: {summary['categorical_features']}")
        print(f"❓ Missing Values: {summary['missing_values']}")
        print(f"✅ Data Quality Score: {summary['data_quality_score']:.2f}%")
        
        return summary

# Usage Example
def main():
    # Initialize the preprocessor
    preprocessor = PersonaFitDataPreprocessor(path)
    
    # Step 1: Load and explore
    data = preprocessor.load_dataset()
    missing_analysis = preprocessor.explore_data()
    
    # Step 2: Identify PersonaFit features
    relevant_features = preprocessor.identify_personafit_features()
    
    # Step 3: Clean the data
    cleaned_data = preprocessor.clean_data()
    
    # Step 4: Feature engineering
    engineered_data = preprocessor.feature_engineering()
    
    # Step 5: Prepare for ML
    ml_ready_data = preprocessor.prepare_ml_features()
    
    # Step 6: Create specific datasets
    personafit_datasets = preprocessor.create_personafit_datasets()
    
    # Step 7: Generate summary
    summary = preprocessor.generate_data_summary()
    
    return preprocessor, personafit_datasets

# Run the pipeline
if __name__ == "__main__":
    preprocessor, datasets = main()
    print("\n🎉 PersonaFit data preprocessing pipeline completed successfully!")
    print("Your datasets are ready for ML model training! 🚀")