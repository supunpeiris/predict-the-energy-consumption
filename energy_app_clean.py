import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import mean_absolute_error, r2_score

# Set page configuration
st.set_page_config(
    page_title="Energy Consumption Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: yellow;
        text-align: center;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: purple;
        padding: 25px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 20px 0;
    }
    .feature-importance {
        background-color: green;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .metric-box {
        background-color: blue;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid red;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_components():
    """Load the pre-trained model and components"""
    try:
        model = joblib.load('best_energy_model.pkl')
        scaler = joblib.load('feature_scaler.pkl')
        feature_names = joblib.load('feature_names.pkl')
        return model, scaler, feature_names
    except FileNotFoundError as e:
        st.error(f"❌ Model files not found: {e}")
        st.info("Please make sure you've saved the model files in the same directory")
        return None, None, None

def get_feature_ranges():
    """Define reasonable ranges for each feature based on your data"""
    return {
        'lights': (0, 100, 30),
        'T1': (15, 30, 20),
        'RH_1': (20, 80, 50),
        'T2': (15, 30, 20),
        'RH_2': (20, 80, 50),
        'T3': (15, 30, 20),
        'RH_3': (20, 80, 50),
        'T4': (15, 30, 20),
        'RH_4': (20, 80, 50),
        'T5': (15, 30, 20),
        'RH_5': (20, 80, 50),
        'T6': (15, 30, 20),
        'RH_6': (20, 80, 50),
        'T7': (15, 30, 20),
        'RH_7': (20, 80, 50),
        'T8': (15, 30, 20),
        'RH_8': (20, 80, 50),
        'T9': (15, 30, 20),
        'RH_9': (20, 80, 50),
        'T_out': (-5, 35, 15),
        'Press_mm_hg': (720, 780, 730),
        'RH_out': (20, 100, 70),
        'Windspeed': (0, 15, 5),
        'Visibility': (0, 50, 25),
        'Tdewpoint': (-5, 25, 10),
        'rv1': (0, 50, 25),
        'rv2': (0, 50, 25),
        'hour': (0, 23, 12),
        'day_of_week': (0, 6, 3),
        'is_weekend': (0, 1, 0),
        'temp_indoor_outdoor_diff': (-10, 10, 0),
        'avg_indoor_temp': (15, 30, 22),
        'avg_indoor_humidity': (20, 80, 50),
        'lights_to_appliances_ratio': (0, 5, 1),
        'is_working_hours': (0, 1, 1),
        'is_sleeping_hours': (0, 1, 0)
    }

def main():
    # Header
    st.markdown('<h1 class="main-header">⚡ Building Energy Consumption Predictor</h1>', unsafe_allow_html=True)
    
    # Load model
    model, scaler, feature_names = load_model_and_components()
    if model is None:
        st.stop()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose Section",
        ["🏠 Home", "🔮 Make Prediction", "📊 Model Info", "🔍 Feature Analysis"]
    )
    
    # Home Page
    if app_mode == "🏠 Home":
        st.header("Welcome to the Energy Consumption Predictor!")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### About This App
            This application predicts building energy consumption using a **Random Forest** machine learning model.
            
            **Model Performance:**
            - 🎯 **R² Score: 0.7444** (74.44% variance explained)
            - 📊 **MAE: 20.56 Wh** (Mean Absolute Error)
            - 🤖 **Algorithm: Random Forest**
            
            **Key Features:**
            - Real-time energy consumption predictions
            - Interactive feature analysis
            - Model performance insights
            - User-friendly interface
            """)
        
        with col2:
            st.markdown("""
            <div style='background-color: purple; padding: 20px; border-radius: 10px;'>
            <h4>🚀 Quick Start</h4>
            <ol>
            <li>Go to <b>Make Prediction</b></li>
            <li>Adjust feature values</li>
            <li>Get instant energy prediction</li>
            </ol>
            </div>
            """, unsafe_allow_html=True)
            
            # Quick metrics
            st.markdown("""
            <div class='metric-box'>
            <h4>📈 Model Metrics</h4>
            <p>R² Score: <b>0.7444</b></p>
            <p>MAE: <b>20.56 Wh</b></p>
            <p>Features: <b>36</b></p>
            </div>
            """, unsafe_allow_html=True)
    
    # Prediction Page
    elif app_mode == "🔮 Make Prediction":
        st.header("🔮 Energy Consumption Prediction")
        
        st.markdown("""
        <div class='prediction-box'>
        <h3>Adjust the features below to predict energy consumption</h3>
        <p>The model will predict energy usage based on your input values.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Get feature ranges
        feature_ranges = get_feature_ranges()
        
        # Organize features by category
        col1, col2, col3 = st.columns(3)
        
        input_features = {}
        
        with col1:
            st.subheader("🏠 Indoor Environment")
            important_features = ['lights', 'T1', 'RH_1', 'T3', 'RH_3', 'T8']
            for feature in important_features:
                if feature in feature_ranges:
                    min_val, max_val, default_val = feature_ranges[feature]
                    input_features[feature] = st.slider(
                        f"{feature}",
                        min_val, max_val, default_val,
                        help=f"Adjust {feature} value"
                    )
        
        with col2:
            st.subheader("🌤️ Outdoor & Weather")
            outdoor_features = ['T_out', 'RH_out', 'Press_mm_hg', 'Windspeed']
            for feature in outdoor_features:
                if feature in feature_ranges:
                    min_val, max_val, default_val = feature_ranges[feature]
                    input_features[feature] = st.slider(
                        f"{feature}",
                        min_val, max_val, default_val,
                        help=f"Adjust {feature} value"
                    )
        
        with col3:
            st.subheader("⏰ Time & Engineered")
            time_features = ['hour', 'is_working_hours', 'is_sleeping_hours', 'lights_to_appliances_ratio']
            for feature in time_features:
                if feature in feature_ranges:
                    min_val, max_val, default_val = feature_ranges[feature]
                    
                    if feature in ['is_working_hours', 'is_sleeping_hours']:
                        input_features[feature] = st.selectbox(
                            f"{feature.replace('_', ' ').title()}",
                            [0, 1],
                            index=default_val,
                            format_func=lambda x: "Yes" if x == 1 else "No"
                        )
                    else:
                        input_features[feature] = st.slider(
                            f"{feature.replace('_', ' ').title()}",
                            min_val, max_val, default_val,
                            help=f"Adjust {feature} value"
                        )
        
        # Fill remaining features with defaults
        for feature in feature_names:
            if feature not in input_features and feature in feature_ranges:
                min_val, max_val, default_val = feature_ranges[feature]
                input_features[feature] = default_val
        
        # Prediction button
        if st.button("🚀 Predict Energy Consumption", type="primary", use_container_width=True):
            try:
                # Prepare input data
                input_df = pd.DataFrame([input_features])[feature_names]
                
                # Make prediction
                prediction = model.predict(input_df)[0]
                
                # Display result
                st.markdown(f"""
                <div style='background-color: #d4edda; padding: 30px; border-radius: 15px; border-left: 5px solid #28a745; text-align: center;'>
                <h2 style='color: #155724; margin: 0;'>🎯 Prediction Result</h2>
                <h1 style='color: #155724; margin: 20px 0;'>{prediction:.1f} Wh</h1>
                <p style='color: #155724; font-size: 1.2rem; margin: 0;'>Estimated Energy Consumption</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Interpretation
                st.subheader("💡 Interpretation")
                if prediction < 30:
                    st.success("**Very Low Consumption** - Minimal appliance usage")
                elif prediction < 60:
                    st.info("**Low Consumption** - Typical for efficient usage")
                elif prediction < 120:
                    st.warning("**Moderate Consumption** - Normal household activity")
                else:
                    st.error("**High Consumption** - Heavy appliance usage")
                
            except Exception as e:
                st.error(f"Prediction error: {e}")
    
    # Model Info Page
    elif app_mode == "📊 Model Info":
        st.header("📊 Model Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Performance")
            st.metric("R² Score", "0.7444")
            st.metric("Mean Absolute Error", "20.56 Wh")
            st.metric("Algorithm", "Random Forest")
            st.metric("Number of Features", "36")
        
        with col2:
            st.subheader("Model Details")
            st.markdown("""
            **Random Forest Regressor:**
            - **n_estimators**: 100 trees
            - **max_depth**: None (unlimited)
            - **random_state**: 42
            
            **Training Data:**
            - UCI Appliances Energy Dataset
            - 19,735 total records
            - 80/20 train/test split
            
            **Interpretation:**
            - R² = 0.7444 means the model explains 74.44% of energy variance
            - MAE = 20.56 Wh means average prediction error is about 20.56 Watt-hours
            """)
        
        # Feature importance visualization
        st.subheader("🔍 Feature Importance")
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            top_15 = feature_importance.head(15)
            
            colors = plt.cm.viridis(np.linspace(0, 1, len(top_15)))
            bars = ax.barh(top_15['feature'], top_15['importance'], color=colors)
            ax.set_xlabel('Feature Importance')
            ax.set_title('Top 15 Most Important Features', fontweight='bold')
            ax.grid(True, alpha=0.3)
            plt.gca().invert_yaxis()
            
            st.pyplot(fig)
    
    # Feature Analysis Page
    elif app_mode == "🔍 Feature Analysis":
        st.header("🔍 Feature Analysis")
        
        st.subheader("Top 10 Most Important Features")
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Display top 10 features
            top_10 = feature_importance.head(10)
            
            for i, (idx, row) in enumerate(top_10.iterrows(), 1):
                importance_pct = (row['importance'] / feature_importance['importance'].sum()) * 100
                
                st.markdown(f"""
                <div class='feature-importance'>
                <h4>#{i}: {row['feature']}</h4>
                <p>Importance: <b>{row['importance']:.4f}</b> ({importance_pct:.1f}% of total)</p>
                </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
