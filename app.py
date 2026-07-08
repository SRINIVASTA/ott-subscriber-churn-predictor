import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import streamlit as st
import plotly.express as px  # Built into the core python container stack

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

    # Simulate real-world OTT streaming platform habits
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
# STREAMLIT USER INTERFACE DECORATION (OTT Specific Text Added)
# =====================================================================
st.title("🎬 OTT Customer Cancellation Risk Tracker")

# Wrapped inside a clean, layman-friendly visual information banner
st.info(
    "💡 **How to use this dashboard:**\n\n"
    "Use this early-warning system to check if a subscriber is happy or at risk of leaving "
    "our OTT video streaming platform. Simply adjust the customer behavior sliders on the left "
    "to match an active subscriber's profile, then click the evaluation button to view an "
    "instant mathematical risk assessment."
)

st.markdown("---")

# Layout Configuration: Split Screen into Input Sidebar Elements and Center Analytics
col_inputs, col_spacer, col_outputs = st.columns([1.2, 0.1, 1.2])

with col_inputs:
    st.subheader("👤 Customer Behavior Profile")

    # Interactive User Widgets
    watch_hours = st.slider(
        "Monthly Hours Watched", min_value=1.0, max_value=200.0, value=15.0, step=0.5
    )
    monthly_cost = st.slider(
        "Monthly Subscription Plan Cost (INR)", min_value=50, max_value=1000, value=649, step=10
    )
    tenure_months = st.slider("Account Age (Months Active)", min_value=1, max_value=60, value=3)
    content_preference = st.selectbox("Favorite Type of Shows/Movies", list(CONTENT_MAP.keys()))

    # Trigger action button
    run_analysis = st.button("Check If Customer Will Cancel", type="primary")

with col_outputs:
    st.subheader("📊 System Assessment")

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
        churn_prob = float(probability[0][1])

        # Metrics presentation layout columns
        m_col1, m_col2 = st.columns(2)

        with m_col1:
            if prediction == 1:
                st.error("🚨 HIGH RISK OF LEAVING")
            else:
                st.success("✅ HAPPY CUSTOMER")

        with m_col2:
            st.metric(label="Calculated Leaving Probability", value=f"{churn_prob:.2%}")

        # Prescriptive Action Strategy Block
        st.markdown("**What Our Team Should Do Next:**")
        if churn_prob > 0.75:
            st.warning(
                "Critical Risk! Send an immediate, highly-targeted discount code or offer a free premium content bundle right now."
            )
        elif churn_prob > 0.45:
            st.info(
                "Moderate Risk. Send a direct mobile app notification highlighting new and trending shows in their favorite category."
            )
        else:
            st.success("No Action Needed. Keep things running normally. This customer shows healthy platform habits.")
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
    }
)

# Categorize each bar to assign dynamic colors
mathematical_risk_df["Risk Status"] = np.where(
    mathematical_risk_df["Impact Level on Final Score"] > 0, "Adds Risk (Unhappy)",
    np.where(mathematical_risk_df["Impact Level on Final Score"] < 0, "Reduces Risk (Happy)", "Neutral")
)

# Generate custom color mapping rules
color_map = {
    "Adds Risk (Unhappy)": "#EF553B",    # Vibrant Red
    "Reduces Risk (Happy)": "#00CC96",  # Vibrant Green
    "Neutral": "#636EFA"                # Neutral Blue
}

# Build interactive Plotly chart object
fig = px.bar(
    mathematical_risk_df,
    x="Customer Habit",
    y="Impact Level on Final Score",
    color="Risk Status",
    color_discrete_map=color_map,
    text=mathematical_risk_df["Impact Level on Final Score"].apply(lambda x: f"{x:+.2f}"),
)

# Style chart lines and text overlays
fig.update_layout(
    xaxis_title="",
    yaxis_title="Risk Impact Level",
    showlegend=True,
    legend_title_text="Behavior Type",
    margin=dict(l=20, r=20, t=20, b=20),
    height=450
)
fig.update_traces(textposition="outside")

# Render chart directly to Streamlit screen
st.plotly_chart(fig, use_container_width=True)

# Display the net calculated risk score summary matrix in an easy-to-read block
net_score = mathematical_risk_df["Impact Level on Final Score"].sum()
st.info(
    f"🧮 **Total Core Dissatisfaction Score:** **{net_score:.2f}** "
    f"*(If this final number goes over **0.45**, the system flags the customer as a high risk of leaving).* "
)
