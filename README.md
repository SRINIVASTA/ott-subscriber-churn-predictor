# 🎬 OTT Customer Cancellation Risk Tracker
**Created by Srinivasta**

An interactive early-warning business intelligence dashboard built to predict subscriber churn (cancellation) risks on an Over-The-Top (OTT) video streaming platform. This system translates advanced machine learning mechanics into clean, actionable, color-coded directives that any non-technical manager can understand instantly.

---

## 💡 What This App Does
This application serves as an operations toolkit for customer support and retention teams:
1. **Predicts Risk:** Evaluates user patterns (hours watched, subscription costs, account age, and content preferences) to determine if a customer is likely to quit.
2. **Quantifies Financial Impact:** Calculates exactly how much annual subscription revenue (in INR) is at stake if the customer cancels.
3. **Prescribes Solutions:** Generates dynamic, automated customer-service action items (e.g., promo codes or push notifications) based on specific churn scores.
4. **Exports Executive Reports:** Allows operators to instantly export a formatted performance statement as an HTML file ready to be printed or saved directly as a corporate PDF.

---

## 🧮 Behind the Dashboard: The Mathematical Rules
To remain accessible to a non-technical user, the chart uses clear **Red Flags (Red Bars)** and **Green Flags (Green Bars)** to show what drives a subscriber's choices. The math functions on a structured scoring loop:

* **Normal Starting Baseline (+0.10):** The starting point risk level for every single customer account.
* **Low Viewing Time Penalty (+0.50 or +0.20):** If a user watches less than 20 hours a month, their cancellation risk spikes by `+0.50`. If they watch between 20 and 50 hours, it increases by `+0.20`.
* **High Price / New Customer Penalty (+0.40):** If an account pays over ₹400 INR/month and has been active for less than 6 months, an extra `+0.40` risk weight is applied.
* **Long-Term Loyalty Discount (-0.30):** If a subscriber stays past 24 months, their risk score drops by `-0.30` because they have developed strong platform habits.
* **Catalog Preference Modifiers:** 
  * Choosing **Live Sports** adds `+0.15` to risk profiles due to seasonal subscription-hopping behaviors.
  * Choosing **Regional Content** subtracts `-0.10` from risk because local programming drives strong customer retention.

> ⚠️ **The Alarm Threshold:** The system sums all these positive and negative weights. If the final **Total Core Dissatisfaction Score is greater than 0.45**, the machine learning model triggers an automatic **🚨 HIGH RISK OF LEAVING** alert.

---

## 🚀 Quick Start & Installation

### 1. Project Directory Layout
Ensure your repository is organized cleanly like this:
```text
├── app.py              # Contains the 2 main Streamlit code blocks
├── requirements.txt    # Lists your external code dependencies
└── README.md           # This project guide file created by Srinivastaa
```

### 2. Configure Dependencies (`requirements.txt`)
Create a file named `requirements.txt` in your root folder and add the following core modules:
```text
scikit-learn
numpy
pandas
streamlit
plotly
```

### 3. Launching Locally
Open your command terminal inside the project directory and run the following execution scripts:
```bash
# Install the missing container libraries
pip install -r requirements.txt

# Boot the web application interface dashboard
streamlit run app.py
```

---

## ☁️ Deploying to Streamlit Cloud
1. Push your updated code files (`app.py`, `requirements.txt`, and `README.md`) to a public **GitHub repository**.
2. Go to [share.streamlit.io](https://ott-subscriber-churn-predictor-angjc7sspwxgzfkmwedmfn.streamlit.app/) and log in using your GitHub account.
3. Click **"New App"**, select your repository name, choose the `main` branch, and set the entry file to `app.py`.
4. Click **"Deploy"**! Streamlit Cloud will parse your `requirements.txt` file, install the libraries automatically, and host your live dashboard on the web.
