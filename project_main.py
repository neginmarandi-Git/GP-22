import pandas as pd
import numpy as np
import xgboost as xgb
import streamlit as st

# --- PAGE CONFIG ---
st.set_page_config(page_title="Retrofit Strategy Tool", layout="wide")

@st.cache_data
def load_and_train_model():
    df = pd.read_csv('AmesHousing.csv')
    df = df[df['Gr Liv Area'] < 4000]
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
    qual_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}
    df['Heating QC'] = df['Heating QC'].map(qual_map).fillna(3)
    df['Kitchen Qual'] = df['Kitchen Qual'].map(qual_map).fillna(3)
    X = df[features].fillna(0)
    y = df['SalePrice']
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
    model.fit(X, y)
    return model, features, df

model, feature_names, original_df = load_and_train_model()

st.title("🏢 Building Retrofit Strategy & ROI Analyzer")

# --- SIDEBAR: SETTINGS ---
st.sidebar.header("1. General Parameters")
area = st.sidebar.slider("Living Area (sq ft)", 500, 4000, 1500)
overall_q = st.sidebar.slider("Overall Quality (1-10)", 1, 10, 6)
year_built = st.sidebar.number_input("Year of Construction", 1900, 2026, 2000)

st.sidebar.markdown("---")
st.sidebar.header("2. Current Condition (Baseline)")
h_current = st.sidebar.select_slider("Current Heating Quality", options=[1,2,3,4,5], value=2)
k_current = st.sidebar.select_slider("Current Kitchen Quality", options=[1,2,3,4,5], value=2)

st.sidebar.markdown("---")
st.sidebar.header("3. Targeted Retrofit (Proposed)")
h_target = st.sidebar.select_slider("Target Heating Quality", options=[1,2,3,4,5], value=4)
k_target = st.sidebar.select_slider("Target Kitchen Quality", options=[1,2,3,4,5], value=4)

st.sidebar.markdown("---")
st.sidebar.header("4. Financials")
renovation_cost = st.sidebar.number_input("Total Renovation Budget ($)", min_value=0, value=20000)

# --- CALCULATIONS ---
base_vals = original_df[feature_names].mean().to_dict()
base_vals.update({'Gr Liv Area': area, 'Overall Qual': overall_q, 'Year Built': year_built})

# Predicted Price: CURRENT
current_data = base_vals.copy()
current_data.update({'Heating QC': h_current, 'Kitchen Qual': k_current})
price_current = model.predict(pd.DataFrame([current_data])[feature_names])[0]

# Predicted Price: AFTER RETROFIT
target_data = base_vals.copy()
target_data.update({'Heating QC': h_target, 'Kitchen Qual': k_target})
price_target = model.predict(pd.DataFrame([target_data])[feature_names])[0]

# ROI Logic
value_increase = price_target - price_current
net_profit = value_increase - renovation_cost

# --- DASHBOARD DISPLAY ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("📍 Current Asset Status")
    st.metric("Estimated Market Value", f"${price_current:,.0f}")
    st.write(f"Heating Level: {h_current} | Kitchen Level: {k_current}")

with col2:
    st.subheader("🚀 Post-Retrofit Projection")
    st.metric("Future Market Value", f"${price_target:,.0f}", delta=f"${value_increase:,.0f}")
    st.write(f"Heating Level: {h_target} | Kitchen Level: {k_target}")

st.markdown("---")
st.header("💰 ROI & Feasibility Study")
c1, c2, c3 = st.columns(3)

c1.metric("Value Appreciation", f"${value_increase:,.0f}")
c2.metric("Retrofit Cost", f"${renovation_cost:,.0f}")
c3.metric("Net Profit / Loss", f"${net_profit:,.0f}")

if net_profit > 0:
    st.success(f"✅ HIGH FEASIBILITY: This strategy generates a net profit of ${net_profit:,.0f}.")
else:
    st.error(f"❌ LOW FEASIBILITY: Retrofit costs exceed the predicted value increase by ${abs(net_profit):,.0f}.")

# Chart Comparison
st.bar_chart(pd.DataFrame({
    'Condition': ['Current', 'Proposed'],
    'Price ($)': [price_current, price_target]
}).set_index('Condition'))


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
