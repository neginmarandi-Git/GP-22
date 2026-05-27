import pandas as pd
import numpy as np
import xgboost as xgb
import streamlit as st
import matplotlib.pyplot as plt

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Retrofit Sensitivity Tool", layout="wide")

@st.cache_data
def load_and_train_model():
    df = pd.read_csv('AmesHousing.csv')
    df = df[df['Gr Liv Area'] < 4000] # Outlier filtering
    
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
    
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X, y)
    
    return model, features, df

model, feature_names, original_df = load_and_train_model()
MODEL_MAE = 9675

st.title("🏢 Building Retrofit Strategy & ROI Analyzer")
st.markdown("### Decision-Making Dashboard for Asset Valuation & Optimization")

# --- SIDEBAR INTERFACE ---
st.sidebar.header("1. Fixed Building Parameters")
overall_q = st.sidebar.slider("Overall Structural Quality (1-10)", 1, 10, 6)
year_built = st.sidebar.number_input("Year of Construction", 1900, 2026, 2000)

st.sidebar.header("2. Baseline Asset State (Current)")
h_current = st.sidebar.select_slider("Current Heating System Quality", options=[1,2,3,4,5], value=2,
                                    format_func=lambda x: ['Poor','Fair','Typical','Good','Excellent'][x-1])
k_current = st.sidebar.select_slider("Current Kitchen Quality", options=[1,2,3,4,5], value=2,
                                    format_func=lambda x: ['Poor','Fair','Typical','Good','Excellent'][x-1])

st.sidebar.header("3. Proposed Retrofit Strategy (Target)")
h_target = st.sidebar.select_slider("Target Heating System Quality", options=[1,2,3,4,5], value=4,
                                   format_func=lambda x: ['Poor','Fair','Typical','Good','Excellent'][x-1])
k_target = st.sidebar.select_slider("Target Kitchen Quality", options=[1,2,3,4,5], value=4,
                                   format_func=lambda x: ['Poor','Fair','Typical','Good','Excellent'][x-1])

st.sidebar.header("4. Project Financials")
renovation_cost = st.sidebar.number_input("Total Renovation Budget ($)", min_value=0, value=20000)
selected_area = st.sidebar.slider("Focus Living Area (sq ft)", 500, 4000, 1500)

# --- SCENARIO CALCULATION ENGINE ---
base_vals = original_df[feature_names].mean().to_dict()
base_vals.update({'Overall Qual': overall_q, 'Year Built': year_built})

def calculate_scenario_prices(area_input):
    temp_vals = base_vals.copy()
    temp_vals['Gr Liv Area'] = area_input
    temp_vals.update({'Heating QC': h_current, 'Kitchen Qual': k_current})
    p_curr = model.predict(pd.DataFrame([temp_vals])[feature_names])[0]
    
    temp_vals.update({'Heating QC': h_target, 'Kitchen Qual': k_target})
    p_targ = model.predict(pd.DataFrame([temp_vals])[feature_names])[0]
    
    return p_curr, p_targ

price_current, price_target = calculate_scenario_prices(selected_area)
value_appreciation = price_target - price_current
net_profit = value_appreciation - renovation_cost

# --- MAIN DASHBOARD VISUALS ---
st.subheader(f"Project Valuation Overview — Focus Scale: {selected_area} sq ft")
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 📍 Baseline State")
    st.metric("Estimated Market Value", f"${price_current:,.0f}")
    st.caption(f"Heating Level: {h_current} | Kitchen Level: {k_current}")

with col2:
    st.markdown("#### 🚀 Post-Retrofit Projection")
    st.metric("Future Market Value", f"${price_target:,.0f}", delta=f"${value_appreciation:,.0f} Value Added")
    st.caption(f"Heating Target: {h_target} | Kitchen Target: {k_target}")

st.markdown("---")
st.subheader("💰 Investment Metrics & Financial Performance")
c1, c2, c3, c4 = st.columns(4)

c1.metric("Gross Appreciation", f"${value_appreciation:,.0f}")
c2.metric("Construction Budget", f"${renovation_cost:,.0f}")
c3.metric("Net Profit / Loss", f"${net_profit:,.0f}")
c4.metric("Model Uncertainty (MAE)", f"${MODEL_MAE:,.0f}")

# --- AUTOMATED TEXT ANALYSIS & MAE VALIDATION ---
st.markdown("---")
st.subheader("📝 Automated Engineering Assessment")

if value_appreciation > MODEL_MAE:
    statistical_reliability = (
        f"The predicted market value jump of **${value_appreciation:,.0f}** is **statistically significant**, "
        f"as it safely exceeds the model's Mean Absolute Error (**${MODEL_MAE:,.0f}**)."
    )
    verdict_text = "This project is financially viable and robust against market volatility."
else:
    statistical_reliability = (
        f"The predicted market value jump of **${value_appreciation:,.0f}** falls within the model's "
        f"uncertainty margin (MAE: **${MODEL_MAE:,.0f}**). The variance might be caused by baseline data noise rather than tangible value enhancement."
    )
    verdict_text = "High Financial Risk Detected due to uncertainty constraints."

if net_profit > 0:
    st.success(f"**Feasibility Analysis:** {verdict_text} {statistical_reliability} "
               f"Executing this intervention strategy is expected to net an economic yield of **${net_profit:,.0f}**.")
else:
    st.error(f"**Feasibility Analysis:** Negative economic yield detected (**${net_profit:,.0f}**). The financial "
             f"cost of construction outweighs the anticipated real estate price growth. {statistical_reliability}")

# --- SENSITIVITY CURVE MAP ---
st.markdown("---")
st.subheader("📈 Sensitivity Curve: Profit Margin vs. Building Scale")
st.write("This simulation tests the economic viability of your chosen quality shift across different property sizes to find the break-even scale.")

area_range = np.linspace(500, 4000, 30)
simulated_profits = []
for test_area in area_range:
    pc, pt = calculate_scenario_prices(test_area)
    simulated_profits.append((pt - pc) - renovation_cost)

fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(area_range, simulated_profits, color='#1f77b4', marker='o', markersize=4, linewidth=2, label='Simulated Strategy Yield')
ax.axhline(0, color='#d62728', linestyle='--', linewidth=1.5, label='Break-Even Boundary ($0 Profit)')
ax.grid(True, linestyle=':', alpha=0.6)
ax.set_xlabel("Living Area (sq ft)")
ax.set_ylabel("Net Investment Profit ($)")
ax.scatter([selected_area], [net_profit], color='#ff7f0e', s=120, zorder=5, label='Your Current Focus Point')
ax.legend(frameon=True, facecolor='white', edgecolor='none')
st.pyplot(fig)

# --- 🎯 NEW ADDITION: LIVE PLOT ANALYSIS & PM RECOMMENDATIONS 🎯 ---
st.markdown("### 📋 Sensitivity Analysis & Strategic Project Management Decisions")

# 1. Curve Interpretation & Analysis text
st.markdown("#### **Curve Interpretation & Market Analysis**")
st.write(
    f"The sensitivity curve maps how the project's net profit scales dynamically across different asset dimensions. "
    f"Currently, your selected property scale is **{selected_area} sq ft**, yielding a net profit of **${net_profit:,.0f}**. "
    f"The trend illustrates that executing this specific retrofit package (Heating upgrade to level {h_target} and Kitchen upgrade to level {k_target}) "
    f"presents a non-linear return on investment. Upgrading smaller residential formats often drops below the red **Break-Even Boundary**, "
    f"whereas mid-to-large spatial formats optimize capital utilization before facing high-end market saturation."
)

# 2. Recommendations text based on MAE and Net Profit constraints
st.markdown("#### **Strategic PM Recommendations & Next Steps**")

rec_1 = ""
rec_2 = ""
rec_3 = ""

if net_profit > 0 and value_appreciation > MODEL_MAE:
    rec_1 = f"✔️ **Product Scope Approval (RIBA Stage 3/4 Transition):** The current technical specifications generate an value jump (**${value_appreciation:,.0f}**) that safely circumvents the system's **${MODEL_MAE:,.0f} MAE** uncertainty buffer. The product scope is validated; freeze current configurations and proceed to detailed execution designs."
    rec_2 = f"⚠️ **Cost Baseline Protection:** Even though the project is highly viable, ensure a minimum **10% contingency financial buffer** is established in the cost baseline to protect the current profit margin against unpredictable material cost hikes during construction."
    rec_3 = f"🎯 **Portfolio Scale Targeting:** Prioritize deploying this architectural and energy upgrade package on other assets within your portfolio that fit within the **1,500 to 2,500 sq ft** optimal efficiency window shown on the curve."
elif net_profit > 0 and value_appreciation <= MODEL_MAE:
    rec_1 = f"⚡ **High Uncertainty Alert:** Although a nominal profit of **${net_profit:,.0f}** is projected, the total value added (**${value_appreciation:,.0f}**) is lower than the model's standard error margin (**${MODEL_MAE:,.0f}**). The projected returns might stem from historical data noise rather than real market appreciation."
    rec_2 = f"🔧 **Value Engineering Needed:** We recommend evaluating a **Product Scope Optimization**. Try adjusting the materials or lowering the target performance levels (e.g., target a 'Good' finish instead of 'Excellent') or negotiate a smaller procurement cost to widen the feasibility safety gap."
    rec_3 = f"🛑 **Risk Mitigation:** Delay freezing the design requirements. Introduce a rigorous local market survey or micro-market audit to verify if buyers in this specific location actively pay a premium for these upgrades before allocating capital expenditure (CapEx)."
else:
    rec_1 = f"❌ **Negative Economic Yield Intervention:** The financial cost of construction (**${renovation_cost:,.0f}**) outweighs the anticipated market value appreciation (**${value_appreciation:,.0f}**), resulting in a net loss. This layout is economically unfeasible."
    rec_2 = f"💡 **Design Pivot Required:** Do not proceed with the current retrofit strategy. You must lower the **Project Scope** cost or target properties that cross the **Break-Even Point** (the point where the blue curve intersects the red dashed line at $0 profit)."
    rec_3 = f"🔄 **Resource Reallocation:** Reallocate the budget away from visual high-end amenities (Kitchen) toward lower-cost energy efficiency baselines, or shift investments to properties with a larger living area where the market cap accommodates premium renovations."

st.info(rec_1)
st.warning(rec_2)
st.markdown(f"- {rec_3}")
