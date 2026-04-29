import pandas as pd
import numpy as np
import xgboost as xgb
import streamlit as st
import matplotlib.pyplot as plt

# --- PAGE CONFIG ---
st.set_page_config(page_title="Retrofit Sensitivity Tool", layout="wide")

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

# --- SIDEBAR ---
st.sidebar.header("1. Fixed Parameters")
overall_q = st.sidebar.slider("Overall Quality (1-10)", 1, 10, 6)
year_built = st.sidebar.number_input("Year of Construction", 1900, 2026, 2000)

st.sidebar.header("2. Retrofit Scenario")
h_current = st.sidebar.select_slider("Current Heating Quality", options=[1,2,3,4,5], value=2)
k_current = st.sidebar.select_slider("Current Kitchen Quality", options=[1,2,3,4,5], value=2)
h_target = st.sidebar.select_slider("Target Heating Quality", options=[1,2,3,4,5], value=4)
k_target = st.sidebar.select_slider("Target Kitchen Quality", options=[1,2,3,4,5], value=4)

st.sidebar.header("3. Financials")
renovation_cost = st.sidebar.number_input("Renovation Budget ($)", min_value=0, value=20000)
selected_area = st.sidebar.slider("Focus Area (sq ft)", 500, 4000, 1500)

# --- CALCULATION LOGIC ---
base_vals = original_df[feature_names].mean().to_dict()
base_vals.update({'Overall Qual': overall_q, 'Year Built': year_built})

def get_profit(area_input):
    temp_vals = base_vals.copy()
    temp_vals['Gr Liv Area'] = area_input
    
    # Current
    temp_vals.update({'Heating QC': h_current, 'Kitchen Qual': k_current})
    p_curr = model.predict(pd.DataFrame([temp_vals])[feature_names])[0]
    
    # Target
    temp_vals.update({'Heating QC': h_target, 'Kitchen Qual': k_target})
    p_targ = model.predict(pd.DataFrame([temp_vals])[feature_names])[0]
    
    return (p_targ - p_curr) - renovation_cost

# Main Calculation for Selected Area
current_profit = get_profit(selected_area)

# --- DASHBOARD ---
st.header(f"Results for {selected_area} sq ft")
c1, c2 = st.columns(2)
if current_profit > 0:
    c1.success(f"Profit: ${current_profit:,.0f}")
else:
    c1.error(f"Loss: ${current_profit:,.0f}")

# --- SENSITIVITY CHART: PROFIT VS AREA ---
st.markdown("---")
st.subheader("📈 Sensitivity Analysis: Profit vs. Building Size")
st.write("This chart shows how the profitability of your retrofit strategy changes as the building area increases.")

area_range = np.linspace(500, 4000, 20)
profits = [get_profit(a) for a in area_range]

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(area_range, profits, color='teal', marker='o', linewidth=2)
ax.axhline(0, color='red', linestyle='--') # Zero profit line
ax.set_xlabel("Living Area (sq ft)")
ax.set_ylabel("Net Profit ($)")
ax.grid(True, alpha=0.3)

# Highlight the selected point
ax.scatter([selected_area], [current_profit], color='orange', s=100, zorder=5, label='Your Selection')
ax.legend()

st.pyplot(fig)

st.info("💡 **Insight:** The red dashed line represents the 'Break-even Point'. If the curve falls below this line, the renovation cost is higher than the value added to the property.")


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
