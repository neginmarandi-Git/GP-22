import pandas as pd
import numpy as np
import xgboost as xgb
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Building Retrofit ROI Predictor", layout="wide")

# --- 1. DATA LOADING & MODEL TRAINING ---
@st.cache_data
def load_and_train_model():
    # Load the Ames Housing Dataset
    df = pd.read_csv('AmesHousing.csv')
    
    # Filter outliers to improve engineering model accuracy
    df = df[df['Gr Liv Area'] < 4000]
    
    # List of 38 key features for the predictive engine
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
    
    # Map qualitative features to numerical values (Sustainability & Quality Metrics)
    qual_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}
    df['Heating QC'] = df['Heating QC'].map(qual_map).fillna(3)
    df['Kitchen Qual'] = df['Kitchen Qual'].map(qual_map).fillna(3)
    
    X = df[features].fillna(0)
    y = df['SalePrice']
    
    # 80/20 Train-Test Split for scientific validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train XGBoost Regressor
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    return model, features, df

# Initialize Model and Data
model, feature_names, original_df = load_and_train_model()

# --- 2. USER INTERFACE (SIDEBAR) ---
st.title("🏗️ Building Retrofit ROI Predictor")
st.markdown("### Integrated Framework for Sustainable Building Management")

st.sidebar.header("📍 Property Parameters")
area = st.sidebar.slider("Living Area (sq ft)", 500, 4000, 1500)
overall_q = st.sidebar.slider("Overall Construction Quality (1-10)", 1, 10, 6)
year_built = st.sidebar.number_input("Year of Construction", 1900, 2026, 2000)

st.sidebar.header("🔧 Retrofit Scenarios")
h_qual = st.sidebar.selectbox("Heating System Quality", [1,2,3,4,5], index=2, 
                             format_func=lambda x: ['Poor','Fair','Typical','Good','Excellent'][x-1])
k_qual = st.sidebar.selectbox("Kitchen/Interior Quality", [1,2,3,4,5], index=2, 
                             format_func=lambda x: ['Poor','Fair','Typical','Good','Excellent'][x-1])

st.sidebar.markdown("---")
st.sidebar.header("💰 Investment Analysis")
renovation_cost = st.sidebar.number_input("Estimated Retrofit Cost ($)", min_value=0, value=15000)

# --- 3. PREDICTION LOGIC ---
# Create baseline using dataset averages for non-user-defined features
base_values = original_df[feature_names].mean().to_dict()
base_values.update({
    'Gr Liv Area': area, 
    'Overall Qual': overall_q, 
    'Year Built': year_built,
    'Heating QC': h_qual, 
    'Kitchen Qual': k_qual
})

# Predict Current Market Value
current_house = pd.DataFrame([base_values])[feature_names]
predicted_price = model.predict(current_house)[0]

# Predict Post-Retrofit Value (Assuming upgrade to 'Excellent' - Level 5)
upgrade_values = base_values.copy()
upgrade_values['Heating QC'] = 5
upgrade_values['Kitchen Qual'] = 5
upgrade_house = pd.DataFrame([upgrade_values])[feature_names]
upgraded_price = model.predict(upgrade_house)[0]

# --- 4. FINANCIAL DASHBOARD ---
profit = (upgraded_price - predicted_price) - renovation_cost
roi = (profit / renovation_cost) * 100 if renovation_cost > 0 else 0

st.markdown("---")
st.markdown("### 📊 Financial & Sustainability Metrics")
col1, col2, col3 = st.columns(3)

col1.metric("Current Asset Value", f"${predicted_price:,.0f}")
col2.metric("Post-Retrofit Value", f"${upgraded_price:,.0f}", delta=f"${upgraded_price - predicted_price:,.0f}")
col3.metric("Net Investment Profit", f"${profit:,.0f}")

# Investment Recommendation with Visual Feedback
if profit > 0:
    st.success(f"✅ **Viable Project:** The predicted Return on Investment (ROI) is **{roi:.1f}%**.")
else:
    st.error(f"⚠️ **High Risk:** The renovation costs exceed the value added. ROI: **{roi:.1f}%**.")

# --- 5. VISUAL COMPARISON ---
st.markdown("---")
st.markdown("### 📈 Scenario Comparison: Asset Appreciation")
comparison_df = pd.DataFrame({
    'Scenario': ['Current Status', 'After Sustainable Retrofit'],
    'Property Value ($)': [predicted_price, upgraded_price]
})
st.bar_chart(data=comparison_df, x='Scenario', y='Property Value ($)')

st.info("Note: The 'After Retrofit' scenario assumes an upgrade to high-efficiency heating systems and premium interior materials.")


# --- ADD THIS TO THE VERY END OF YOUR project_main.py ---

# 1. Prepare data for validation
from sklearn.metrics import mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt

# We need to define X and y again to avoid NameError
val_features = [
    'MS SubClass', 'Lot Frontage', 'Lot Area', 'Overall Qual', 'Overall Cond', 
    'Year Built', 'Year Remod/Add', 'Mas Vnr Area', 'BsmtFin SF 1', 'BsmtFin SF 2', 
    'Bsmt Unf SF', 'Total Bsmt SF', '1st Flr SF', '2nd Flr SF', 'Low Qual Fin SF', 
    'Gr Liv Area', 'Bsmt Full Bath', 'Bsmt Half Bath', 'Full Bath', 'Half Bath', 
    'Bedroom AbvGr', 'Kitchen AbvGr', 'TotRms AbvGrd', 'Fireplaces', 'Garage Yr Blt', 
    'Garage Cars', 'Garage Area', 'Wood Deck SF', 'Open Porch SF', 'Enclosed Porch', 
    '3Ssn Porch', 'Screen Porch', 'Pool Area', 'Misc Val', 'Mo Sold', 'Yr Sold',
    'Heating QC', 'Kitchen Qual'
]

# Ensure quality mapping matches the training
qual_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}
df_val = pd.read_csv('AmesHousing.csv')
df_val['Heating QC'] = df_val['Heating QC'].map(qual_map).fillna(3)
df_val['Kitchen Qual'] = df_val['Kitchen Qual'].map(qual_map).fillna(3)

X_val = df_val[val_features].fillna(0)
y_val = df_val['SalePrice']

# 2. Calculate MAE
y_pred_val = model.predict(X_val) # 'model' should be defined in your previous code
mae = mean_absolute_error(y_val, y_pred_val)
mean_price = y_val.mean()
error_percentage = (mae / mean_price) * 100

print(f"Mean Absolute Error (MAE): ${mae:,.0f}")
print(f"Average House Price: ${mean_price:,.0f}")
print(f"Error Percentage: {error_percentage:.2f}%")

# 3. Plotting the Distribution vs. MAE
plt.figure(figsize=(12, 6))
sns.histplot(y_val, kde=True, color='teal', label='Price Distribution (Actual)')
plt.axvline(mae, color='red', linestyle='--', linewidth=2, label=f'MAE: ${mae:,.0f}')

plt.title('Validation: Sales Price Distribution vs. Model MAE', fontsize=14)
plt.xlabel('Sale Price ($)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.show()

# Instead of plt.show(), save the figure to a file
plt.savefig('model_validation_plot.png', dpi=300, bbox_inches='tight')
print("Validation plot has been saved as 'model_validation_plot.png' in your project folder.")
