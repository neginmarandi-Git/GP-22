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
st.markdown("### 📝 Automated Engineering Assessment")

if value_appreciation > MODEL_MAE:
    statistical_reliability = (
        f"The predicted market value jump of **${value_appreciation:,.0f}** is **statistically significant**, "
        f"as it safely exceeds the model's Mean Absolute Error (**${MODEL_MAE:,.0f}**)."
    )
    verdict_text = "This project is financially viable and robust against market volatility."
    alert_type = "success"
else:
    statistical_reliability = (
        f"The predicted market value jump of **${value_appreciation:,.0f}** falls within the model's "
        f"uncertainty margin (MAE: **${MODEL_MAE:,.0f}**). The variance might be caused by baseline data noise rather than tangible value enhancement."
    )
    verdict_text = "High Financial Risk Detected due to uncertainty constraints."
    alert_type = "error"

if net_profit > 0:
    st.success(f"**Feasibility Analysis:** {verdict_text} {statistical_reliability} "
               f"Executing this intervention strategy is expected to net an economic yield of **...** ${net_profit:,.0f}.")
else:
    st.error(f"**Feasibility Analysis:** Negative economic yield detected (**...** ${net_profit:,.0f}). The financial "
             f"cost of construction outweighs the anticipated real estate price growth. {statistical_reliability}")

# --- SENSITIVITY CURVE MAP ---
st.markdown("---")
st.subheader("📈 Sensitivity Curve: Profit Margin vs. Building Scale")

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

# --- 📄 NEW NATIVE REPORT GENERATOR (NO WEASYPRINT REQUIRED) 📄 ---
st.sidebar.markdown("---")
st.sidebar.header("📥 Project Reports & Deliverables")

if st.sidebar.checkbox("Preview Executive Report"):
    labels = ['Poor', 'Fair', 'Typical', 'Good', 'Excellent']
    
    st.markdown("---")
    st.markdown("## 📄 Executive Feasibility Report")
    st.info("💡 **Tip for Presentation:** Press **Ctrl + P** (or **Cmd + P** on Mac) to save this clean report directly as a PDF via your browser!")
    
    # Render a clean, printable HTML report layout inside Streamlit using Markdown
    report_html = f"""
    <div style="background-color: #f8fafc; padding: 25px; border: 1px solid #cbd5e0; border-radius: 6px; font-family: sans-serif; color: #2d3748;">
        <div style="background-color: #1a365d; color: white; padding: 20px; border-radius: 4px; margin-bottom: 20px;">
            <h2 style="margin: 0; color: white; font-size: 18pt;">Building Retrofit Strategy & Feasibility Report</h2>
            <p style="margin: 5px 0 0 0; font-size: 10.5pt; color: #e2e8f0;">Politecnico di Milano - Quantitative Design Optimization Support</p>
        </div>
        
        <h3 style="color: #2b6cb0; border-bottom: 2px solid #e2e8f0; padding-bottom: 5px;">1. Product Scope Specifications (Technical Matrix)</h3>
        <table style="width: 100%; border-collapse: collapse; margin-bottom: 20px; font-size: 10pt;">
            <thead>
                <tr style="background-color: #edf2f7;">
                    <th style="border: 1px solid #cbd5e0; padding: 10px;">Structural Feature</th>
                    <th style="border: 1px solid #cbd5e0; padding: 10px;">Baseline Asset State (Current)</th>
                    <th style="border: 1px solid #cbd5e0; padding: 10px;">Proposed Retrofit Strategy (Target)</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td style="border: 1px solid #cbd5e0; padding: 10px;"><strong>Overall Structural Quality</strong></td>
                    <td style="border: 1px solid #cbd5e0; padding: 10px;">Level {overall_q} / 10</td>
                    <td style="border: 1px solid #cbd5e0; padding: 10px;">Level {overall_q} / 10 (Fixed Baseline)</td>
                </tr>
                <tr>
                    <td style="border: 1px solid #cbd5e0; padding: 10px;"><strong>Heating System Condition (QC)</strong></td>
                    <td style="border: 1px solid #cbd5e0; padding: 10px;">{labels[h_current-1]} (Level {h_current})</td>
                    <td style="border: 1px solid #cbd5e0; padding: 10px;">{labels[h_target-1]} (Level {h_target})</td>
                </tr>
                <tr>
                    <td style="border: 1px solid #cbd5e0; padding: 10px;"><strong>Kitchen Material Quality</strong></td>
                    <td style="border: 1px solid #cbd5e0; padding: 10px;">{labels[k_current-1]} (Level {k_current})</td>
                    <td style="border: 1px solid #cbd5e0; padding: 10px;">{labels[k_target-1]} (Level {k_target})</td>
                </tr>
                <tr>
                    <td style="border: 1px solid #cbd5e0; padding: 10px;"><strong>Focus Asset Dimension (Scale)</strong></td>
                    <td style="border: 1px solid #cbd5e0; padding: 10px;">{selected_area} sq ft</td>
                    <td style="border: 1px solid #cbd5e0; padding: 10px;">{selected_area} sq ft</td>
                </tr>
            </tbody>
        </table>

        <h3 style="color: #2b6cb0; border-bottom: 2px solid #e2e8f0; padding-bottom: 5px;">2. Investment Performance & Market Uncertainty Appraisal</h3>
        <div style="background-color: white; border: 1px solid #e2e8f0; padding: 15px; border-radius: 4px; margin-bottom: 20px;">
            <p><strong>Estimated Asset Baseline Value:</strong> ${price_current:,.0f}</p>
            <p><strong>Post-Retrofit Target Valuation:</strong> ${price_target:,.0f}</p>
            <p><strong>Gross Added Value (Appreciation):</strong> ${value_appreciation:,.0f}</p>
            <p><strong>Allocated Capital Expenditure (CapEx Budget):</strong> ${renovation_cost:,.0f}</p>
            <p><strong>Net Projected Profit Margin:</strong> ${net_profit:,.0f}</p>
            <p style="color: #e53e3e;"><strong>Model Uncertainty Baseline (MAE Error Buffer):</strong> ${MODEL_MAE:,.0f}</p>
        </div>

        <div style="padding: 15px; border-radius: 4px; margin-bottom: 20px; background-color: {'#f0fff4' if alert_type=='success' else '#fffaf0'}; border-left: 5px solid {'#38a169' if alert_type=='success' else '#dd6b20'}; color: {'#276749' if alert_type=='success' else '#7b341e'};">
            <strong>Project Manager Executive Summary:</strong> {verdict_text} {statistical_reliability}
        </div>

        <h3 style="color: #2b6cb0; border-bottom: 2px solid #e2e8f0; padding-bottom: 5px;">3. Strategic Decision-Support Recommendations</h3>
        <ul style="padding-left: 20px; line-height: 1.6;">
            <li><strong>Scope Lock Criterion:</strong> If value added safely exceeds the ${MODEL_MAE:,.0f} MAE benchmark, the current technical specifications are approved. Freeze configurations and transition to detailed structural blueprints.</li>
            <li><strong>Risk Contingency Protocol:</strong> Maintain a minimum 10% cash contingency allowance in the baseline cost estimation. This isolates and safeguards your net profit against material supply chain spikes.</li>
            <li><strong>Portfolio Scale Targeting:</strong> Prioritize deploying this architectural retrofit package exclusively on properties matching the optimal square footage zones highlighted on the dynamic sensitivity graph.</li>
        </ul>
    </div>
    """
    st.markdown(report_html, unsafe_allow_html=True)
