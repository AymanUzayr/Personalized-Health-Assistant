import streamlit as st
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import time
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="PersonaFit - AI Fitness Assistant",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #FF6B6B;
    }
    .recommendation-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin: 1rem 0;
    }
    .explanation-box {
        background: #e8f5e8;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class PersonaFitEngine:
    def __init__(self):
        self.models = self._initialize_models()
        self.user_sessions = {}
    
    def _initialize_models(self):
        """Initialize mock ML models for demonstration"""
        return {
            'workout_difficulty': RandomForestRegressor(n_estimators=50, random_state=42),
            'calorie_recommendation': RandomForestRegressor(n_estimators=50, random_state=42),
            'recovery_prediction': RandomForestRegressor(n_estimators=50, random_state=42),
            'progress_forecasting': RandomForestRegressor(n_estimators=50, random_state=42),
            'diet_classification': RandomForestClassifier(n_estimators=50, random_state=42)
        }
    
    def calculate_user_metrics(self, age, weight, height, activity_level, fitness_goal):
        """Calculate user metrics and make predictions"""
        # Calculate BMI
        bmi = weight / ((height/100) ** 2)
        
        # Determine BMI category
        if bmi < 18.5:
            bmi_category = "Underweight"
        elif bmi < 25:
            bmi_category = "Normal"
        elif bmi < 30:
            bmi_category = "Overweight"
        else:
            bmi_category = "Obese"
        
        # Age group classification
        if age < 25:
            age_group = "Youth"
        elif age < 35:
            age_group = "Young Adult"
        elif age < 45:
            age_group = "Adult"
        elif age < 55:
            age_group = "Middle Age"
        else:
            age_group = "Senior"
        
        return {
            'bmi': round(bmi, 1),
            'bmi_category': bmi_category,
            'age_group': age_group
        }
    
    def predict_workout_difficulty(self, user_data):
        """Predict optimal workout difficulty (1-10 scale)"""
        # Mock prediction based on user data
        age_factor = max(0.4, 1.2 - (user_data['age'] - 20) * 0.02)
        bmi_factor = 1.0 if user_data['bmi_category'] == 'Normal' else 0.7
        activity_factor = {'Low': 0.6, 'Moderate': 1.0, 'High': 1.3}.get(user_data['activity_level'], 1.0)
        
        base_difficulty = 5.0
        difficulty = base_difficulty * age_factor * bmi_factor * activity_factor
        difficulty += np.random.normal(0, 0.5)  # Add some realistic variation
        
        return max(1.0, min(10.0, round(difficulty, 1)))
    
    def predict_calorie_needs(self, user_data):
        """Predict daily calorie requirements"""
        # Harris-Benedict equation (simplified)
        if user_data.get('gender', 'Male') == 'Male':
            bmr = 88.362 + (13.397 * user_data['weight']) + (4.799 * user_data['height']) - (5.677 * user_data['age'])
        else:
            bmr = 447.593 + (9.247 * user_data['weight']) + (3.098 * user_data['height']) - (4.330 * user_data['age'])
        
        # Activity multiplier
        activity_multipliers = {'Low': 1.2, 'Moderate': 1.55, 'High': 1.9}
        total_calories = bmr * activity_multipliers.get(user_data['activity_level'], 1.55)
        
        # Adjust for fitness goal
        if user_data['fitness_goal'] == 'Weight Loss':
            total_calories *= 0.85
        elif user_data['fitness_goal'] == 'Muscle Gain':
            total_calories *= 1.15
        
        return round(total_calories)
    
    def predict_recovery_time(self, user_data, workout_difficulty):
        """Predict recovery time needed"""
        base_recovery = 24  # Base 24 hours
        
        # Age adjustment
        age_factor = 1 + (user_data['age'] - 25) * 0.02
        
        # Difficulty adjustment
        difficulty_factor = workout_difficulty / 5.0
        
        # Fitness level adjustment
        fitness_factor = {'Low': 1.5, 'Moderate': 1.0, 'High': 0.7}.get(user_data['activity_level'], 1.0)
        
        recovery_hours = base_recovery * age_factor * difficulty_factor * fitness_factor
        
        return round(max(12, min(72, recovery_hours)))
    
    def predict_diet_type(self, user_data):
        """Recommend optimal diet type"""
        diet_types = ['Mediterranean', 'Low Carb', 'High Protein', 'Balanced', 'Plant Based']
        
        # Logic-based recommendation
        if user_data['fitness_goal'] == 'Muscle Gain':
            return 'High Protein'
        elif user_data['fitness_goal'] == 'Weight Loss':
            return 'Low Carb' if user_data['bmi_category'] in ['Overweight', 'Obese'] else 'Mediterranean'
        elif user_data['age'] > 50:
            return 'Mediterranean'
        else:
            return 'Balanced'
    
    def predict_progress_score(self, user_data):
        """Predict fitness progress potential (0-100)"""
        base_score = 60
        
        # Age factor (younger = higher potential)
        age_factor = max(0.5, 1.0 - (user_data['age'] - 25) * 0.01)
        
        # BMI factor
        bmi_factor = 1.0 if user_data['bmi_category'] == 'Normal' else 0.8
        
        # Activity factor
        activity_factor = {'Low': 0.7, 'Moderate': 1.0, 'High': 1.2}.get(user_data['activity_level'], 1.0)
        
        score = base_score * age_factor * bmi_factor * activity_factor
        score += np.random.normal(0, 5)  # Add variation
        
        return round(max(10, min(100, score)), 1)
    
    def generate_time_series_forecast(self, user_id, current_progress=None):
        """Generate 30-day progress forecast"""
        if current_progress is None:
            current_progress = np.random.uniform(40, 80)
        
        dates = []
        progress_values = []
        weight_values = []
        
        # Generate trend
        daily_improvement = np.random.uniform(0.2, 0.8)
        weight_change_rate = np.random.uniform(-0.1, 0.1)  # kg per day
        
        for i in range(30):
            date = datetime.now() + timedelta(days=i)
            dates.append(date.strftime('%Y-%m-%d'))
            
            # Progress with some realistic fluctuation
            progress = current_progress + (i * daily_improvement) + np.random.normal(0, 2)
            progress = max(0, min(100, progress))
            progress_values.append(round(progress, 1))
            
            # Weight projection
            weight_change = 70 + (i * weight_change_rate) + np.random.normal(0, 0.3)
            weight_values.append(round(weight_change, 1))
        
        return {
            'dates': dates,
            'fitness_progress': progress_values,
            'weight_projection': weight_values,
            'trend': 'Improving' if daily_improvement > 0.4 else 'Stable'
        }
    
    def explain_recommendations(self, user_data, predictions):
        """Generate explanations for recommendations"""
        explanations = {}
        
        # Workout difficulty explanation
        difficulty = predictions['workout_difficulty']
        if difficulty <= 3:
            explanations['workout'] = f"🔰 **Beginner Level ({difficulty}/10)**: Starting with low intensity based on your current fitness level and experience. This helps build a solid foundation safely."
        elif difficulty <= 6:
            explanations['workout'] = f"⚡ **Moderate Level ({difficulty}/10)**: Your age ({user_data['age']}) and activity level ({user_data['activity_level']}) suggest you can handle moderate intensity workouts effectively."
        else:
            explanations['workout'] = f"🔥 **Advanced Level ({difficulty}/10)**: Your high activity level and good fitness metrics indicate you're ready for challenging workouts that will maximize results."
        
        # Calorie explanation
        calories = predictions['daily_calories']
        explanations['calories'] = f"🍽️ **{calories} calories/day**: Based on your BMR calculation considering your weight ({user_data['weight']}kg), height ({user_data['height']}cm), age ({user_data['age']}), and {user_data['activity_level'].lower()} activity level. Adjusted for your goal: {user_data['fitness_goal']}."
        
        # Recovery explanation
        recovery = predictions['recovery_hours']
        explanations['recovery'] = f"😴 **{recovery} hours recovery**: Your age group ({user_data.get('age_group', 'Adult')}) and planned workout intensity require this recovery time for optimal muscle repair and growth."
        
        # Diet explanation
        diet = predictions['diet_type']
        explanations['diet'] = f"🥗 **{diet} Diet**: Recommended based on your fitness goal ({user_data['fitness_goal']}) and current BMI category ({user_data.get('bmi_category', 'Normal')}). This diet type aligns best with your objectives."
        
        return explanations

# Initialize the PersonaFit engine
@st.cache_resource
def load_personafit_engine():
    return PersonaFitEngine()

engine = load_personafit_engine()

# Main App Interface
def main():
    # Header
    st.markdown('<h1 class="main-header">💪 PersonaFit - AI Fitness Assistant</h1>', unsafe_allow_html=True)
    
    # Sidebar for user input
    st.sidebar.header("👤 Your Profile")
    
    # User inputs
    age = st.sidebar.slider("Age", 16, 80, 28)
    weight = st.sidebar.number_input("Weight (kg)", 40.0, 200.0, 70.0, 0.1)
    height = st.sidebar.number_input("Height (cm)", 140.0, 220.0, 175.0, 0.1)
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    activity_level = st.sidebar.selectbox("Current Activity Level", ["Low", "Moderate", "High"])
    fitness_goal = st.sidebar.selectbox("Primary Fitness Goal", 
                                      ["Weight Loss", "Muscle Gain", "General Fitness", "Endurance"])
    
    # Calculate user metrics
    user_data = {
        'age': age,
        'weight': weight,
        'height': height,
        'gender': gender,
        'activity_level': activity_level,
        'fitness_goal': fitness_goal
    }
    
    # Add calculated metrics
    calculated_metrics = engine.calculate_user_metrics(age, weight, height, activity_level, fitness_goal)
    user_data.update(calculated_metrics)
    
    # Generate predictions button
    if st.sidebar.button("🎯 Generate My PersonaFit Plan", type="primary"):
        with st.spinner("🤖 AI is analyzing your profile..."):
            time.sleep(2)  # Simulate processing time
            
            # Make predictions
            predictions = {
                'workout_difficulty': engine.predict_workout_difficulty(user_data),
                'daily_calories': engine.predict_calorie_needs(user_data),
                'recovery_hours': engine.predict_recovery_time(user_data, engine.predict_workout_difficulty(user_data)),
                'diet_type': engine.predict_diet_type(user_data),
                'progress_score': engine.predict_progress_score(user_data)
            }
            
            # Store in session state
            st.session_state.user_data = user_data
            st.session_state.predictions = predictions
            st.session_state.show_results = True
    
    # Main content area
    if st.session_state.get('show_results', False):
        show_personafit_results()
    else:
        show_welcome_screen()

def show_welcome_screen():
    """Display welcome screen with app features"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🏋️ Smart Workouts</h3>
            <p>AI-powered workout difficulty prediction based on your fitness level and goals</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>🍎 Nutrition Plans</h3>
            <p>Personalized calorie recommendations and diet type suggestions</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>📈 Progress Tracking</h3>
            <p>30-day forecasting with recovery time optimization</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🚀 Get Started")
    st.info("Fill out your profile in the sidebar and click 'Generate My PersonaFit Plan' to receive personalized AI recommendations!")

def show_personafit_results():
    """Display PersonaFit results and recommendations"""
    user_data = st.session_state.user_data
    predictions = st.session_state.predictions
    
    # Generate explanations
    explanations = engine.explain_recommendations(user_data, predictions)
    
    # User Overview
    st.markdown("## 📊 Your PersonaFit Profile")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("BMI", f"{user_data['bmi']}", user_data['bmi_category'])
    with col2:
        st.metric("Age Group", user_data['age_group'])
    with col3:
        st.metric("Activity Level", user_data['activity_level'])
    with col4:
        st.metric("Fitness Goal", user_data['fitness_goal'])
    
    st.markdown("---")
    
    # Recommendations Section
    st.markdown("## 🎯 Your Personalized Recommendations")
    
    # Main recommendations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="recommendation-box">
            <h3>🏋️ Workout Plan</h3>
            <h2>{predictions['workout_difficulty']}/10 Intensity</h2>
            <p>Optimal workout difficulty level</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="recommendation-box">
            <h3>🍽️ Nutrition Plan</h3>
            <h2>{predictions['daily_calories']} cal/day</h2>
            <p>Recommended daily intake</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="recommendation-box">
            <h3>😴 Recovery Time</h3>
            <h2>{predictions['recovery_hours']} hours</h2>
            <p>Between intense workouts</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="recommendation-box">
            <h3>🥗 Diet Type</h3>
            <h2>{predictions['diet_type']}</h2>
            <p>Best suited for your goals</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Explanations Section
    st.markdown("## 🧠 Why These Recommendations?")
    
    for key, explanation in explanations.items():
        st.markdown(f"""
        <div class="explanation-box">
            {explanation}
        </div>
        """, unsafe_allow_html=True)
    
    # Progress Forecasting
    st.markdown("---")
    st.markdown("## 📈 30-Day Progress Forecast")
    
    # Generate time series data
    forecast_data = engine.generate_time_series_forecast("user_123", predictions['progress_score'])
    
    # Create progress chart
    fig_progress = go.Figure()
    fig_progress.add_trace(go.Scatter(
        x=forecast_data['dates'],
        y=forecast_data['fitness_progress'],
        mode='lines+markers',
        name='Fitness Progress (%)',
        line=dict(color='#FF6B6B', width=3)
    ))
    
    fig_progress.update_layout(
        title="Predicted Fitness Progress Over 30 Days",
        xaxis_title="Date",
        yaxis_title="Fitness Progress (%)",
        height=400,
        showlegend=True
    )
    
    st.plotly_chart(fig_progress, use_container_width=True)
    
    # Weight projection chart
    fig_weight = go.Figure()
    fig_weight.add_trace(go.Scatter(
        x=forecast_data['dates'],
        y=forecast_data['weight_projection'],
        mode='lines+markers',
        name='Weight (kg)',
        line=dict(color='#4ECDC4', width=3)
    ))
    
    fig_weight.update_layout(
        title="Weight Projection Over 30 Days",
        xaxis_title="Date",
        yaxis_title="Weight (kg)",
        height=400,
        showlegend=True
    )
    
    st.plotly_chart(fig_weight, use_container_width=True)
    
    # Additional Insights
    st.markdown("---")
    st.markdown("## 💡 Additional Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"📊 **Progress Trend**: {forecast_data['trend']}")
    
    with col2:
        weekly_improvement = round((forecast_data['fitness_progress'][6] - forecast_data['fitness_progress'][0]), 1)
        st.success(f"📈 **Weekly Improvement**: +{weekly_improvement}%")
    
    with col3:
        monthly_goal = round(forecast_data['fitness_progress'][-1], 1)
        st.warning(f"🎯 **30-Day Goal**: {monthly_goal}% fitness level")
    
    # API Endpoints Section
    st.markdown("---")
    st.markdown("## 🔌 API Integration")
    
    with st.expander("View API Endpoints for Developers"):
        st.code(f"""
# PersonaFit API Endpoints (for integration)

POST /api/predict/workout
{{
    "age": {user_data['age']},
    "weight": {user_data['weight']},
    "height": {user_data['height']},
    "activity_level": "{user_data['activity_level']}"
}}
Response: {{"workout_difficulty": {predictions['workout_difficulty']}}}

POST /api/predict/calories
{{
    "user_profile": {json.dumps(user_data, indent=2)}
}}
Response: {{"daily_calories": {predictions['daily_calories']}}}

GET /api/forecast/progress?user_id=123&days=30
Response: {json.dumps(forecast_data, indent=2)[:200]}...

POST /api/explain/recommendations
{{
    "user_data": {json.dumps(user_data, indent=2)[:100]}...,
    "predictions": {json.dumps(predictions, indent=2)}
}}
""", language="json")
    
    # Action buttons
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📱 Export to Mobile App", type="primary"):
            st.success("✅ Recommendations exported to PersonaFit mobile app!")
    
    with col2:
        if st.button("📧 Email My Plan"):
            st.success("✅ Personalized plan sent to your email!")
    
    with col3:
        if st.button("🔄 Generate New Plan"):
            st.session_state.show_results = False
            st.rerun()

# Initialize session state
if 'show_results' not in st.session_state:
    st.session_state.show_results = False

# Run the app
if __name__ == "__main__":
    main()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🚀 PersonaFit - Powered by AI & Machine Learning | Built with Streamlit</p>
    <p>💡 Your personalized fitness journey starts here!</p>
</div>
""", unsafe_allow_html=True)