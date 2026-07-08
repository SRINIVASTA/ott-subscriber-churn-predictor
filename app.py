import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import streamlit as st

# Set page configuration layout
st.set_page_config(page_title="OTT Churn Predictor", layout="wide")

# Hardcoded reference map for categorical conversions
CONTENT_MAP = {"Global": 0, "Live Sports": 1, "Regional": 2}


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

    # Simulate messy real-world OTT telemetry distributions
    watch_hours = np.random.uniform(2.0, 160.0, records)
    monthly_cost = np.random.choice([149, 199, 299, 499, 649], records)
    tenure_months = np.random.randint(1, 48, records)
    content_pref = np.random.choice(list(CONTENT_MAP.keys()), records)

    churn_labels = []
    for wh, cost, tenure, pref in zip(watch_hours, monthly_cost, tenure_months, content_pref):
        risk_score = 0.1  # Baseline risk
        
        # 1. Watch Time Math Rules
        if wh < 20:
            risk_score += 0.5
        elif wh < 50:
            risk_score += 0.2
            
        # 2. Cost vs Tenure Rule
        if cost > 400 and tenure < 6:
            risk_score += 0.4
            
        # 3. Loyalty Discount Rule
        if tenure > 24:
            risk_score -= 0.3
            
        # 4. Content Preference Math Rules
        if pref == "Live Sports":
            risk_score += 0.15
        elif pref == "Regional":
            risk_score -= 0.10

        # Injecting natural human variance noise
        risk_score += np.random.normal(0, 0.05)
        churn_labels.append(1 if risk_score > 0.45 else 0)

    # Convert arrays directly to processing dataframes
    df = pd.DataFrame(
        {
            "Watch_Hours_Monthly": watch_hours,
            "Monthly_Cost_INR": monthly_cost,
            "Tenure_Months": tenure_months,
            "Content_Preference": [CONTENT_MAP[p] for p in content_pref],
            "Churned": churn_labels,
        }
    )

    # NATIVE PURE-PYTHON REBALANCING STEP (Bypasses SMOTE)
    df_retained = df[df["Churned"] == 0]
    df_churned = df[df["Churned"] == 1]

    # Oversample minority class using pandas sampling logic
    df_churned_upsampled = df_churned.sample(len(df_retained), replace=True, random_state=42)
    df_balanced = pd.concat([df_retained, df_churned_upsampled]).sample(frac=1, random_state=42)

    X = df_balanced.drop("Churned", axis=1)
    y = df_balanced["Churned"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit best hyperparameters uncovered during cross-validation tuning
    model = RandomForestClassifier(
        n_estimators=150, max_depth=10, class_weight="balanced", random_state=42
    )
    model.fit(X_scaled, y)

    return model, scaler, list(X.columns)


# Execute initialization
model, scaler, feature_names = initialize_and_train_model()

# =====================================================================
# STREAMLIT USER INTERFACE DECORATION
# =====================================================================
st.title("🎬 OTT Subscriber Churn Analytics")
st.write(
    "Adjust subscriber behavior vectors using the sliders below to predict churn risks instantly."
)

st.markdown("---")

# Layout Configuration: Split Screen into Input Sidebar Elements and Center Analytics
col_inputs, col_spacer, col_outputs = st.columns([1.2, 0.1, 1.2])

with col_inputs:
    st.subheader("👤 User Behavior Inputs")

    # Interactive User Widgets
    watch_hours = st.slider(
        "Monthly Watch Time (Hours)", min_value=1.0, max_value=200.0, value=15.0, step=0.5
    )
    monthly_cost = st.slider(
        "Monthly Subscription Cost (INR)", min_value=50, max_value=1000, value=649, step=10
    )
    tenure_months = st.slider("Account Tenure (Months)", min_value=1, max_value=60, value=3)
    content_preference = st.selectbox("Preferred Catalog Content Type", list(CONTENT_MAP.keys()))

    # Trigger action button
    run_analysis = st.button("Analyze Subscriber Churn Risk", type="primary")

with col_outputs:
    st.subheader("📊 Algorithmic Evaluation")

    if run_analysis:
        # Convert active user configuration inputs to evaluation dataframe formats safely
        input_data = pd.DataFrame(
            [
                {
                    "Watch_Hours_Monthly": watch_hours,
                    "Monthly_Cost_INR": monthly_cost,
                    "Tenure_Months": tenure_months,
                    "Content_Preference": CONTENT_MAP[content_preference],
                }
            ]
        )

        # Enforce strict column alignment to match scaler criteria perfectly
        input_data = input_data[feature_names]
        scaled_input = scaler.transform(input_data)

        # Compute binary values and true mathematical probabilities
        prediction = model.predict(scaled_input)
        probability = model.predict_proba(scaled_input)
        churn_prob = float(probability[0][1])  # Fixed slicing assignment logic

        # Metrics presentation layout columns
        m_col1, m_col2 = st.columns(2)

        with m_col1:
            if prediction == 1:
                st.error("🚨 HIGH CHURN RISK")
            else:
                st.success("✅ RETAINED USER")

        with m_col2:
            st.metric(label="Churn Probability", value=f"{churn_prob:.2%}")

        # Prescriptive Action Strategy Block
        st.markdown("**Recommended Next Steps:**")
        if churn_prob > 0.75:
            st.warning(
                "Dispatch an immediate hyper-targeted discount promo code or high-value regional package offer."
            )
        elif churn_prob > 0.45:
            st.info(
                "Trigger direct mobile push notifications highlighting trending content releases inside their category choice."
            )
        else:
            st.success("Maintain automated general update streams. User shows positive health signs.")
    else:
        st.info("👈 Adjust sliders and click the button to generate prediction telemetry.")

st.markdown("---")

# =====================================================================
# INTERACTIVE DATA GRAPHING SECTION (Exact Mathematical Risk Contribution)
# =====================================================================
st.subheader("🔮 Behind the Algorithm: Mathematical Risk Breakdown")
st.write(
    f"This chart displays the exact mathematical risk vectors added or subtracted based on your slider settings and your choice of **{content_preference}**."
)

# 1. Compute exact mathematical risk components mimicking your dataset logic
watch_risk = 0.5 if watch_hours < 20 else (0.2 if watch_hours < 50 else 0.0)
cost_tenure_risk = 0.4 if (monthly_cost > 400 and tenure_months < 6) else 0.0
loyalty_discount = -0.3 if tenure_months > 24 else 0.0

# 2. Compute exact mathematical contribution of chosen content preferences
if content_preference == "Live Sports":
    content_risk = 0.15
elif content_preference == "Regional":
    content_risk = -0.10
else:
    content_risk = 0.0  # Global acts as the mathematical control baseline

# Create the mathematical dataset for the visualization chart
mathematical_risk_df = pd.DataFrame(
    {
        "Risk Vector Component": [
            "Base Subscriber Risk",
            "Low Watch Time Penalty",
            "High Cost / Low Tenure Penalty",
            "Long-Term Loyalty Discount",
            f"Catalog Focus ({content_preference})",
        ],
        "Mathematical Score Impact": [
            0.1, 
            watch_risk, 
            cost_tenure_risk, 
            loyalty_discount, 
            content_risk
        ],
    }
)

# Render the mathematically precise bar chart
st.bar_chart(
    data=mathematical_risk_df,
    x="Risk Vector Component",
    y="Mathematical Score Impact",
    use_container_width=True,
)

# Display the net calculated risk score summary matrix
net_score = mathematical_risk_df["Mathematical Score Impact"].sum()
st.info(f"🧮 **Total Core Mathematical Risk Score:** {net_score:.2f} *(Threshold for high risk flags is > 0.45)*")
