import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import streamlit as st

# Set page configuration layout
st.set_page_config(page_title="OTT Churn Predictor", layout="centered")

# Hardcoded reference map for categorical conversions
CONTENT_MAP = {"Global": 0, "Live Sports": 1, "Regional": 2}


# =====================================================================
# CACHED MACHINE LEARNING PIPELINE ENGINE
# =====================================================================
@st.cache_resource
def initialize_and_train_model():
    """Generates synthetic data and trains the pipeline once, caching it

    internally to make user input slider changes instantaneous.
    """
    np.random.seed(42)
    records = 3000

    # Simulate messy real-world OTT telemetry distributions
    watch_hours = np.random.uniform(2.0, 160.0, records)
    monthly_cost = np.random.choice([149, 199, 299, 499, 649], records)
    tenure_months = np.random.randint(1, 48, records)
    content_pref = np.random.choice(list(CONTENT_MAP.keys()), records)

    churn_labels = []
    for wh, cost, tenure in zip(watch_hours, monthly_cost, tenure_months):
        risk_score = 0.1
        if wh < 20:
            risk_score += 0.5
        elif wh < 50:
            risk_score += 0.2
        if cost > 400 and tenure < 6:
            risk_score += 0.4
        if tenure > 24:
            risk_score -= 0.3

        # Injecting natural human variance noise
        risk_score += np.random.normal(0, 0.15)
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

    X = df.drop("Churned", axis=1)
    y = df["Churned"]

    # Balance classes via SMOTE to address minority data gaps
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_resampled)

    # Fit best hyperparameters uncovered during cross-validation tuning
    model = RandomForestClassifier(
        n_estimators=150, max_depth=10, class_weight="balanced", random_state=42
    )
    model.fit(X_scaled, y_resampled)

    return model, scaler, X.columns


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
col_inputs, col_spacer, col_outputs = st.columns([1.1, 0.1, 1.1])

with col_inputs:
    st.subheader("👤 User Behavior Inputs")

    # Interactive User Widgets
    watch_hours = st.slider(
        "Monthly Watch Time (Hours)",
        min_value=1.0,
        max_value=200.0,
        value=15.0,
        step=0.5,
    )
    monthly_cost = st.slider(
        "Monthly Subscription Cost (INR)",
        min_value=50,
        max_value=1000,
        value=649,
        step=10,
    )
    tenure_months = st.slider(
        "Account Tenure (Months)", min_value=1, max_value=60, value=3
    )
    content_preference = st.selectbox(
        "Preferred Catalog Content Type", list(CONTENT_MAP.keys())
    )

    # Convert active user configuration inputs to evaluation dataframe formats
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

    # Trigger action button
    run_analysis = st.button("Analyze Subscriber Churn Risk", type="primary")

with col_outputs:
    st.subheader("📊 Algorithmic Evaluation")

    if run_analysis:
        # Apply global processing configurations
        scaled_input = scaler.transform(input_data)

        # Compute binary values and true mathematical probabilities
        prediction = model.predict(scaled_input)
        probability = model.predict_proba(scaled_input)[0][1]

        # Metrics presentation layout columns
        m_col1, m_col2 = st.columns(2)

        with m_col1:
            if prediction == 1:
                st.error("🚨 HIGH CHURN RISK")
            else:
                st.success("✅ RETAINED USER")

        with m_col2:
            st.metric(label="Churn Probability", value=f"{probability:.2%}")

        # Prescriptive Action Strategy Block
        st.markdown("**Recommended Next Steps:**")
        if probability > 0.75:
            st.warning(
                "Dispatch an immediate hyper-targeted discount promo code or high-value regional package offer."
            )
        elif probability > 0.45:
            st.info(
                "Trigger direct mobile push notifications highlighting trending content releases inside their category choice."
            )
        else:
            st.success(
                "Maintain automated general update streams. User shows positive health signs."
            )
    else:
        st.info("👈 Adjust sliders and click the button to generate prediction telemetry.")

st.markdown("---")

# =====================================================================
# INTERACTIVE DATA GRAPHING SECTION (Feature Importance Layout)
# =====================================================================
st.subheader("🔮 Behind the Algorithm: Operational Churn Signals")
st.write(
    "This interactive chart displays exactly which behavioral metrics the model values most when calculating churn scores."
)

# Extract Gini importance vectors dynamically from the Random Forest model
importance_df = pd.DataFrame(
    {"Behavioral Metric": feature_names, "Signal Importance": model.feature_importances_}
).sort_values(by="Signal Importance", ascending=False)

# Render native, clean, interactive Streamlit bar chart
st.bar_chart(
    data=importance_df,
    x="Behavioral Metric",
    y="Signal Importance",
    use_container_width=True,
)
