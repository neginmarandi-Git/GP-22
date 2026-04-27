import pandas as pd
import xgboost as xgb
import streamlit as st
from sklearn.model_selection import train_test_split

# --- 1. DATA LOADING & MODEL TRAINING ---
@st.cache_data
def load_and_train_model():
    # Load the Ames Housing Dataset
    df = pd.read_csv('AmesHousing.csv')
    
    # Filter outliers to improve engineering model accuracy
    df = df[df['Gr Liv Area'] < 4000]
    
    # Precise list of 38 features required by the model (as identified in the error log)
    features = [
        'MS SubClass', 'Lot Frontage', 'Lot Area', 'Overall Qual', 'Overall Cond', 
        'Year Built', 'Year Remod/Add', 'Mas Vnr Area', 'BsmtFin SF 1', 'BsmtFin SF 2', 
        'Bsmt Unf SF', 'Total Bsmt SF', '1st Flr SF', '2nd Flr SF', 'Low Qual Fin SF', 
        'Gr Liv Area', 'Bsmt Full Bath', 'Bsmt Half Bath', 'Full Bath', 'Half Bath', 
        'Bedroom AbvGr', 'Kitchen AbvGr', 'TotRms AbvGrd', 'Fireplaces', 'Garage Yr Blt', 
        'Garage Cars', 'Garage Area', 'Wood Deck SF', 'Open Porch SF', 'Enclosed Porch', 
        '3Ssn Porch', 'Screen Porch', 'Pool Area', 'Misc Val', 'Mo Sold', 'Yr Sold',
        'Heating QC', 'Kitchen Qual'
    ]
    
    # Map qualitative features to numerical values for the mathematical model
    qual_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}
    df['Heating QC'] = df['Heating QC'].map(qual_map).fillna(3)
    df['Kitchen Qual'] = df['Kitchen Qual'].map(qual_map).fillna(3)
    
    # Handling missing values and defining X and y
    X = df[features].fillna(0)
    y = df['SalePrice']
    
    # Training the XGBoost Regressor
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
    xgb_model.fit(X_train, y_train)
    
    return xgb_model, features, df

# Initialize Model and Data
model, feature_names, original_df = load_and_train_model()

# --- 2. STREAMLIT UI SETUP ---
st.title("🏗️ Building Retrofit ROI Predictor")
st.markdown("### Integrated Project Management Framework")

st.sidebar.header("Building Parameters")
# Primary inputs for the simulation
area = st.sidebar.slider("Living Area (sq ft)", 500, 4000, 1500)
overall_q = st.sidebar.slider("Overall Construction Quality (1-10)", 1, 10, 6)
year_built = st.sidebar.number_input("Year of Construction", 1900, 2025, 2000)

st.sidebar.header("Retrofit Scenario Analysis")
h_qual = st.sidebar.selectbox("Current Heating System Quality", [1,2,3,4,5], index=2, 
                              format_func=lambda x: ['Poor','Fair','Typical','Good','Excellent'][x-1])
k_qual = st.sidebar.selectbox("Current Kitchen/Interior Quality", [1,2,3,4,5], index=2, 
                              format_func=lambda x: ['Poor','Fair','Typical','Good','Excellent'][x-1])

# --- 3. PREDICTION LOGIC (Ensuring Feature Alignment) ---
# Create a baseline using the mean values of the dataset for the remaining 33 features
base_values = original_df[feature_names].mean().to_dict()
base_values.update({
    'Gr Liv Area': area,
    'Overall Qual': overall_q,
    'Year Built': year_built,
    'Heating QC': h_qual,
    'Kitchen Qual': k_qual
})

# Construct the input DataFrame with exact feature ordering
base_house = pd.DataFrame([base_values])[feature_names]

# Predict current market value
predicted_price = model.predict(base_house)[0]

# --- 4. RETROFIT IMPACT SIMULATION ---
# Simulate an upgrade to 'Excellent' (Level 5) for both targeted features
upgrade_house = base_house.copy()
upgrade_house['Heating QC'] = 5  
upgrade_house['Kitchen Qual'] = 5 
upgraded_price = model.predict(upgrade_house)[0]

# --- 5. RESULTS DASHBOARD ---
col1, col2 = st.columns(2)
col1.metric("Predicted Current Value", f"${predicted_price:,.0f}")
col2.metric("Post-Retrofit Value", f"${upgraded_price:,.0f}", delta=f"${upgraded_price - predicted_price:,.0f}")

st.info(f"Summary: The simulated retrofit adds approximately **${upgraded_price - predicted_price:,.0f}** to the asset value.")

