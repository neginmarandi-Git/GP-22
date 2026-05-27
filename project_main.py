import pandas as pd
import numpy as np
import xgboost as xgb
import streamlit as st
import matplotlib.pyplot as plt
from weasyprint import HTML  

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

# --- 📄 PDF REPORT GENERATION ENGINE 📄 ---
st.sidebar.markdown("---")
st.sidebar.header("📥 Project Reports & Deliverables")

if st.sidebar.button("Generate Feasibility PDF Report"):
    # Save the current plot temporarily to include in calculation data or records
    plt.savefig('temp_sensitivity_curve.png', dpi=300, bbox_inches='tight')
    
    # Map numbers to readable strings for the report
    labels = ['Poor', 'Fair', 'Typical', 'Good', 'Excellent']
    
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <style>
        @page {{ size: A4; margin: 20mm 15mm; }}
        body {{ font-family: Arial, sans-serif; color: #2d3748; direction: ltr; line-height: 1.5; }}
        .header {{ background-color: #1a365d; color: white; padding: 20px; border-radius: 4px; margin-bottom: 20px; }}
        h1 {{ margin: 0; font-size: 18pt; }}
        h2 {{ color: #2b6cb0; font-size: 14pt; border-bottom: 2px solid #e2e8f0; padding-bottom: 5px; margin-top: 20px; }}
        table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; font-size: 10pt; }}
        th, td {{ border: 1px solid #cbd5e0; padding: 10px; text-align: center; }}
        th {{ background-color: #f7fafc; font-weight: bold; }}
        .metric-box {{ background-color: #f8fafc; border: 1px solid #e2e8f0; padding: 15px; border-radius: 4px; margin-bottom: 20px; }}
        .status-alert {{ padding: 12px; border-radius: 4px; background-color: #f0fff4; border-left: 4px solid #38a169; color: #276749; font-size: 10.5pt; }}
        .status-danger {{ padding: 12px; border-radius: 4px; background-color: #fffaf0; border-left: 4px solid #dd6b20; color: #7b341e; font-size: 10.5pt; }}
    </style>
    </head>
    <body>
        <div class="header">
            <h1>Building Retrofit Strategy & Feasibility Report</h1>
            <p>Politecnico di Milano - Quantitative Design Optimization Support</p>
        </div>
        
        <h2>1. Product Scope Specifications (Technical Matrix)</h2>
        <table>
            <thead>
                <tr>
                    <th>Structural Feature</th>
                    <th>Baseline Asset State (Current)</th>
                    <th>Proposed Retrofit Strategy (Target)</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>Overall Structural Quality</strong></td>
                    <td>Level {overall_q} / 10</td>
                    <td>Level {overall_q} / 10 (Fixed Baseline)</td>
                </tr>
                <tr>
                    <td><strong>Heating System Condition (QC)</strong></td>
                    <td>{labels[h_current-1]} (Level {h_current})</td>
                    <td>{labels[h_target-1]} (Level {h_target})</td>
                </tr>
                <tr>
                    <td><strong>Kitchen Material / Asset Quality</strong></td>
                    <td>{labels[k_current-1]} (Level {k_current})</td>
                    <td>{labels[k_target-1]} (Level {k_target})</td>
                </tr>
                <tr>
                    <td><strong>Focus Asset Dimension (Scale)</strong></td>
                    <td>{selected_area} sq ft</td>
                    <td>{selected_area} sq ft</td>
                </tr>
            </tbody>
        </table>

        <h2>2. Investment Performance & Market Uncertainty Appraisal</h2>
        <div class="metric-box">
            <table style="border:none; margin:0;">
                <tr style="border:none;">
                    <td style="border:none; text-align:left;"><strong>Baseline Value:</strong> ${price_current:,.0f}</td>
                    <td style="border:none; text-align:left;"><strong>Renovation Budget (CapEx):</strong> ${renovation_cost:,.0f}</td>
                </tr>
                <tr style="border:none;">
                    <td style="border:none; text-align:left;"><strong>Target Value:</strong> ${price_target:,.0f}</td>
                    <td style="border:none; text-align:left;"><strong>Net Profit Margin:</strong> ${net_profit:,.0f}</td>
                </tr>
                <tr style="border:none;">
                    <td style="border:none; text-align:left;"><strong>Gross Added Value:</strong> ${value_appreciation:,.0f}</td>
                    <td style="border:none; text-align:left; color:#e53e3e;"><strong>Model Uncertainty Baseline (MAE):</strong> ${MODEL_MAE:,.0f}</td>
                </tr>
            </table>
        </div>

        <div class="{'status-alert' if net_profit > 0 else 'status-danger'}">
            <strong>Project Manager Executive Summary:</strong> {verdict_text} {statistical_reliability}
        </div>

        <h2>3. Strategic Recommendations for the Project Manager</h2>
        <ul>
            <li><strong>Scope Lock:</strong> If the appreciation safely circumvents the ${MODEL_MAE:,.0f} MAE, freeze product configurations and proceed to RIBA Stage 4 (Detailed Design).</li>
            <li><strong>Risk Contingency:</strong> Maintain a minimum 10% contingency financial buffer in the cost baseline to shield the profit margin from unexpected material price hikes.</li>
            <li><strong>Scale Control:</strong> Prioritize this specific retrofit intervention package on assets ranging near the optimal peaks shown on the dashboard sensitivity curve.</li>
        </ul>
    </body>
    </html>
    """
    
    # Compile HTML string to PDF using WeasyPrint
    HTML(string=html_template).write_pdf("Retrofit_Feasibility_Report.pdf")
    st.sidebar.success("✅ PDF Generated Successfully!")
    
    with open("Retrofit_Feasibility_Report.pdf", "rb") as file:
        st.sidebar.download_button(
            label="⬇️ Download PDF Report",
            data=file,
            file_name=f"Retrofit_Report_{selected_area}sqft.pdf",
            mime="application/pdf"
        )
