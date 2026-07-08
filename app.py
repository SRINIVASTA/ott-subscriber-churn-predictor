import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import streamlit as st
import plotly.express as px

# Set page configuration layout
st.set_page_config(page_title="OTT Churn Predictor", layout="wide")

# Hardcoded reference map for categorical conversions
CONTENT_MAP = {"Global": 0, "Live Sports": 1, "Regional": 2}

# =====================================================================
# WIDGET SESSION STATE TRACKER & RESET LOGIC
# =====================================================================
if "watch_hours" not in st.session_state:
    st.session_state.watch_hours = 15.0
if "monthly_cost" not in st.session_state:
    st.session_state.monthly_cost = 649
if "tenure_months" not in st.session_state:
    st.session_state.tenure_months = 3
if "content_preference" not in st.session_state:
    st.session_state.content_preference = "Global"

def reset_dashboard_layout():
    """Snaps all slider and selection inputs back to original defaults."""
    st.session_state.watch_hours = 15.0
    st.session_state.monthly_cost = 649
    st.session_state.tenure_months = 3
    st.session_state.content_preference = "Global"
    st.rerun()

# =====================================================================
# CACHED MACHINE LEARNING PIPELINE ENGINE
# =====================================================================
@st.cache_resource
def initialize_and_train_model():
    """Generates synthetic data using explicit mathematical logic
    rules to train a Random Forest model on clear telemetry behaviors.
    """
    np.random.seed(42)
    records = 3000

    watch_hours_arr = np.random.uniform(2.0, 160.0, records)
    monthly_cost_arr = np.random.choice([149, 199, 299, 499, 649], records)
    tenure_months_arr = np.random.randint(1, 48, records)
    content_pref_arr = np.random.choice(list(CONTENT_MAP.keys()), records)

    churn_labels = []
    for wh, cost, tenure, pref in zip(watch_hours_arr, monthly_cost_arr, tenure_months_arr, content_pref_arr):
        risk_score = 0.1
        if wh < 20:
            risk_score += 0.5
        elif wh < 50:
            risk_score += 0.2
        if cost > 400 and tenure < 6:
            risk_score += 0.4
        if tenure > 24:
            risk_score -= 0.3
        if pref == "Live Sports":
            risk_score += 0.15
        elif pref == "Regional":
            risk_score -= 0.10

        risk_score += np.random.normal(0, 0.05)
        churn_labels.append(1 if risk_score > 0.45 else 0)

    df = pd.DataFrame({
        "Watch_Hours_Monthly": watch_hours_arr,
        "Monthly_Cost_INR": monthly_cost_arr,
        "Tenure_Months": tenure_months_arr,
        "Content_Preference": [CONTENT_MAP[p] for p in content_pref_arr],
        "Churned": churn_labels,
    })

    df_retained = df[df["Churned"] == 0]
    df_churned = df[df["Churned"] == 1]
    df_churned_upsampled = df_churned.sample(len(df_retained), replace=True, random_state=42)
    df_balanced = pd.concat([df_retained, df_churned_upsampled]).sample(frac=1, random_state=42)

    X = df_balanced.drop("Churned", axis=1)
    y = df_balanced["Churned"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=150, max_depth=10, class_weight="balanced", random_state=42)
    model.fit(X_scaled, y)

    return model, scaler, list(X.columns)

model, scaler, feature_names = initialize_and_train_model()
# =====================================================================
# STREAMLIT USER INTERFACE DECORATION
# =====================================================================
st.title("🎬 OTT Customer Cancellation Risk Tracker")

st.info(
    "💡 **How to use this dashboard:**\n\n"
    "Use this early-warning system to check if a subscriber is happy or at risk of leaving "
    "our OTT video streaming platform. Simply adjust the customer behavior sliders on the left "
    "to match an active subscriber's profile, then click the evaluation button to view an "
    "instant mathematical risk assessment."
)

st.markdown("---")

col_inputs, col_spacer, col_outputs = st.columns([1.2, 0.1, 1.2])

with col_inputs:
    st.subheader("👤 Customer Behavior Profile")

    watch_hours = st.slider("Monthly Hours Watched", min_value=1.0, max_value=200.0, step=0.5, key="watch_hours")
    monthly_cost = st.slider("Monthly Subscription Plan Cost (INR)", min_value=50, max_value=1000, step=10, key="monthly_cost")
    tenure_months = st.slider("Account Age (Months Active)", min_value=1, max_value=60, key="tenure_months")
    content_preference = st.selectbox("Favorite Type of Shows/Movies", list(CONTENT_MAP.keys()), key="content_preference")

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        run_analysis = st.button("Check If Customer Will Cancel", type="primary", use_container_width=True)
    with btn_col2:
        st.button("🔄 Reset Profile Sliders", on_click=reset_dashboard_layout, use_container_width=True)

with col_outputs:
    st.subheader("📊 System Assessment")

    if run_analysis:
        input_data = pd.DataFrame([{
            "Watch_Hours_Monthly": watch_hours,
            "Monthly_Cost_INR": monthly_cost,
            "Tenure_Months": tenure_months,
            "Content_Preference": CONTENT_MAP[content_preference],
        }])

        input_data = input_data[feature_names]
        scaled_input = scaler.transform(input_data)

        prediction = model.predict(scaled_input)
        probability = model.predict_proba(scaled_input)
        churn_prob = float(probability[0][1])

        m_col1, m_col2 = st.columns(2)
        with m_col1:
            if prediction == 1:
                st.error("🚨 HIGH RISK OF LEAVING")
            else:
                st.success("✅ HAPPY CUSTOMER")
        with m_col2:
            st.metric(label="Calculated Leaving Probability", value=f"{churn_prob:.2%}")

        st.markdown("**📉 Projected Financial Footprint:**")
        annual_loss = monthly_cost * 12
        
        if prediction == 1:
            st.metric(
                label="At-Risk Annual Subscription Revenue Loss", 
                value=f"₹{annual_loss:,} INR", 
                delta=f"-₹{monthly_cost} / month",
                delta_color="inverse"
            )
        else:
            st.metric(
                label="Secured Annual Active Account Value", 
                value=f"₹{annual_loss:,} INR", 
                delta=f"+₹{monthly_cost} / month",
                delta_color="normal"
            )

        st.markdown("**What Our Team Should Do Next:**")
        action_text = ""
        if churn_prob > 0.75:
            action_text = "Critical Risk! Send an immediate, highly-targeted discount code or offer a free premium content bundle right now."
            st.warning(action_text)
        elif churn_prob > 0.45:
            action_text = "Moderate Risk. Send a direct mobile app notification highlighting new and trending shows in their favorite category."
            st.info(action_text)
        else:
            action_text = "No Action Needed. Keep things running normally. This customer shows healthy platform habits."
            st.success(action_text)

        report_data = f"""
        <html>
        <body style="font-family: Arial, sans-serif; padding: 20px; color: #333;">
            <div style="text-align: center; border-bottom: 2px solid #333; padding-bottom: 10px;">
                <h1 style="margin: 0; color: #111;">OTT Subscriber Churn Risk Assessment</h1>
                <p style="margin: 5px 0 0 0; color: #666;">Automated Operations Analytics Report</p>
            </div>
            
            <h3 style="color: #444; margin-top: 20px;">1. Subscriber Behavioral Profile</h3>
            <table style="width: 100%; border-collapse: collapse; margin-bottom: 20px;">
                <tr style="background-color: #f9f9f9;"><td style="padding: 8px; border: 1px solid #ddd; font-weight: bold;">Monthly Viewing Time:</td><td style="padding: 8px; border: 1px solid #ddd;">{watch_hours} Hours</td></tr>
                <tr><td style="padding: 8px; border: 1px solid #ddd; font-weight: bold;">Plan Rate Cost:</td><td style="padding: 8px; border: 1px solid #ddd;">₹{monthly_cost} INR / month</td></tr>
                <tr style="background-color: #f9f9f9;"><td style="padding: 8px; border: 1px solid #ddd; font-weight: bold;">Account Age Longevity:</td><td style="padding: 8px; border: 1px solid #ddd;">{tenure_months} Months</td></tr>
                <tr><td style="padding: 8px; border: 1px solid #ddd; font-weight: bold;">Favorite Content Type:</td><td style="padding: 8px; border: 1px solid #ddd;">{content_preference} Catalog</td></tr>
            </table>

            <h3 style="color: #444;">2. Algorithmic System Summary</h3>
            <div style="padding: 15px; background-color: {'#FADBD8' if prediction == 1 else '#D4EFDF'}; border-radius: 5px; margin-bottom: 20px;">
                <p style="margin: 0; font-size: 16px; font-weight: bold; color: {'#922B21' if prediction == 1 else '#196F3D'};">
                    STATUS ASSESSMENT: {'🚨 HIGH RISK OF CANCELLATION' if prediction == 1 else '✅ SECURED / RETAINED USER'}
                </p>
                <p style="margin: 5px 0 0 0;">Computed Leave Likelihood: <strong>{churn_prob:.2%}</strong></p>
                <p style="margin: 5px 0 0 0;">Projected Revenue Exposure: <strong>₹{annual_loss:,} INR / Year</strong></p>
            </div>

            <h3 style="color: #444;">3. Prescriptive Strategy Directives</h3>
            <p style="padding: 10px; background-color: #f5f5f5; border-left: 4px solid #333; margin: 0;">{action_text}</p>
            
            <div style="margin-top: 40px; text-align: center; font-size: 11px; color: #999; border-top: 1px solid #eee; padding-top: 10px;">
                This assessment was generated electronically using certified operational ML analytics weights.
            </div>
        </body>
        </html>
        """
        
        st.download_button(
            label="📥 Export Report As File (HTML format for printing/PDF)",
            data=report_data,
            file_name=f"OTT_Risk_Report_Cost_{monthly_cost}.html",
            mime="text/html",
            use_container_width=True
        )
        st.caption("ℹ️ *Tip: Open the downloaded file in your browser, press Ctrl+P (or Cmd+P), and click 'Save as PDF' to generate a physical document report instantly.*")
    else:
        st.info("👈 Set the sliders on the left and click the button to see the customer evaluation report.")

st.markdown("---")

# =====================================================================
# INTERACTIVE DATA GRAPHING SECTION (Layman-Friendly Risk Breakdown)
# =====================================================================
st.subheader("🔮 What is Driving this Customer's Decisions?")
st.write(
    f"This chart breaks down the customer's active habits into positive and negative risk bars. "
    f"🔴 **Red bars pointing UP** mean the customer is unhappy and more likely to quit. "
    f"🟢 **Green bars pointing DOWN** mean they are happy and want to stay. "
)

watch_risk = 0.5 if watch_hours < 20 else (0.2 if watch_hours < 50 else 0.0)
cost_tenure_risk = 0.4 if (monthly_cost > 400 and tenure_months < 6) else 0.0
loyalty_discount = -0.3 if tenure_months > 24 else 0.0

if content_preference == "Live Sports":
    content_risk = 0.15
elif content_preference == "Regional":
    content_risk = -0.10
else:
    content_risk = 0.0

mathematical_risk_df = pd.DataFrame({
    "Customer Habit": [
        "Normal Starting Baseline",
        "Low Viewing Time Penalty",
        "High Price / New Customer Penalty",
        "Long-Term Loyalty Discount",
        f"Favorite Content Choice ({content_preference})",
    ],
    "Impact Level on Final Score": [
        0.1, 
        watch_risk, 
        cost_tenure_risk, 
        loyalty_discount, 
        content_risk
    ],
})

mathematical_risk_df["Risk Status"] = np.where(
    mathematical_risk_df["Impact Level on Final Score"] > 0, "Adds Risk (Unhappy)",
    np.where(mathematical_risk_df["Impact Level on Final Score"] < 0, "Reduces Risk (Happy)", "Neutral")
)

color_map = {
    "Adds Risk (Unhappy)": "#EF553B",
    "Reduces Risk (Happy)": "#00CC96",
    "Neutral": "#636EFA"
}

fig = px.bar(
    mathematical_risk_df,
    x="Customer Habit",
    y="Impact Level on Final Score",
    color="Risk Status",
    color_discrete_map=color_map,
    text=mathematical_risk_df["Impact Level on Final Score"].apply(lambda x: f"{x:+.2f}"),
)

fig.update_layout(
    xaxis_title="",
    yaxis_title="Risk Impact Level",
    showlegend=True,
    legend_title_text="Behavior Type",
    margin=dict(l=20, r=20, t=20, b=20),
    height=450
)
fig.update_traces(textposition="outside")

st.plotly_chart(fig, use_container_width=True)

net_score = mathematical_risk_df["Impact Level on Final Score"].sum()
st.info(
    f"🧮 **Total Core Dissatisfaction Score:** **{net_score:.2f}** "
    f"*(If this final number goes over **0.45**, the system flags the customer as a high risk of leaving).* "
)
