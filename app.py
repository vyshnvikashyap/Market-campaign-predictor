import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from datetime import datetime

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Bank Deposit Predictor",
    page_icon="🏦",
    layout="wide"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        width: 100%;
        background-color: #2c3e50;
        color: white;
        border-radius: 8px;
        padding: 10px;
        font-size: 16px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #3498db;
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .section-header {
        font-size: 20px;
        font-weight: bold;
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 8px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ── Load Models ──────────────────────────────────────────────
rf       = joblib.load('model.pkl')
scaler   = joblib.load('scaler.pkl')
features = joblib.load('features.pkl')
cat_cols = joblib.load('cat_cols.pkl')

# ── Initialize History ───────────────────────────────────────
if 'history' not in st.session_state:
    st.session_state.history = []

# ── Helper Functions ─────────────────────────────────────────
def predict(customer):
    sample     = pd.DataFrame([customer])
    sample_enc = pd.get_dummies(sample, columns=cat_cols, drop_first=True)
    sample_enc = sample_enc.reindex(columns=features, fill_value=0)
    sample_sc  = scaler.transform(sample_enc)
    pred       = rf.predict(sample_sc)[0]
    proba      = rf.predict_proba(sample_sc)[0][1]
    return pred, proba

def gauge_chart(proba):
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number+delta",
        value = proba * 100,
        title = {'text': "Subscription Probability", 'font': {'size': 18}},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar':  {'color': "#2ecc71" if proba >= 0.5 else "#e74c3c"},
            'steps': [
                {'range': [0,  40], 'color': '#ffcccc'},
                {'range': [40, 60], 'color': '#fff3cc'},
                {'range': [60, 100],'color': '#ccffcc'},
            ],
            'threshold': {
                'line':  {'color': "#2c3e50", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    fig.update_layout(height=300, margin=dict(t=50, b=0))
    return fig

def risk_profile(customer, proba):
    st.markdown('<p class="section-header">📋 Customer Risk Profile</p>',
                unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    # Risk Level
    if proba >= 0.7:
        risk, color, emoji = "Low Risk",    "#2ecc71", "🟢"
    elif proba >= 0.4:
        risk, color, emoji = "Medium Risk", "#f39c12", "🟡"
    else:
        risk, color, emoji = "High Risk",   "#e74c3c", "🔴"

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{emoji} {risk}</h3>
            <p style="color:{color}; font-size:24px; font-weight:bold;">
                {proba:.2%}
            </p>
            <p>Subscription Probability</p>
        </div>""", unsafe_allow_html=True)

    with col2:
        balance_status = "✅ Positive" if customer['balance'] > 0 else "❌ Negative"
        st.markdown(f"""
        <div class="metric-card">
            <h3>💰 Balance Status</h3>
            <p style="font-size:20px; font-weight:bold;">
                {balance_status}
            </p>
            <p>€{customer['balance']:,}</p>
        </div>""", unsafe_allow_html=True)

    with col3:
        call_status = "✅ Good" if customer['campaign'] <= 3 else "⚠️ Too Many"
        st.markdown(f"""
        <div class="metric-card">
            <h3>📞 Call Status</h3>
            <p style="font-size:20px; font-weight:bold;">
                {call_status}
            </p>
            <p>{customer['campaign']} calls made</p>
        </div>""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────
st.title("🏦 Bank Deposit Subscription Predictor")
st.write("Predict whether a customer will subscribe to a term deposit")
st.divider()

# ── Tabs ─────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🔍 Single Prediction",
    "📂 Batch Prediction",
    "📜 Prediction History"
])

# ════════════════════════════════════════════════════════════
# TAB 1 — Single Prediction
# ════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-header">👤 Customer Details</p>',
                unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        age      = st.number_input("Age", 18, 95, 40)
        balance  = st.number_input("Balance (€)", -5000, 50000, 1000)
        duration = st.number_input("Call Duration (sec)", 0, 3000, 300)
        campaign = st.number_input("Number of Calls", 1, 50, 2)

    with col2:
        job       = st.selectbox("Job", ["management","blue-collar","technician",
                                         "admin.","services","retired","student",
                                         "self-employed","entrepreneur","unemployed"])
        marital   = st.selectbox("Marital Status", ["single","married","divorced"])
        education = st.selectbox("Education", ["tertiary","secondary","primary"])
        default   = st.selectbox("Has Credit Default?", ["no","yes"])

    with col3:
        housing  = st.selectbox("Has Housing Loan?", ["no","yes"])
        loan     = st.selectbox("Has Personal Loan?", ["no","yes"])
        contact  = st.selectbox("Contact Type", ["cellular","telephone","unknown"])
        month    = st.selectbox("Month", ["jan","feb","mar","apr","may","jun",
                                          "jul","aug","sep","oct","nov","dec"])
        poutcome = st.selectbox("Previous Outcome",
                                ["success","failure","unknown","other"])

    col4, col5 = st.columns(2)
    with col4:
        day = st.slider("Day of Month", 1, 31, 15)
    with col5:
        pdays    = st.number_input("Days Since Last Contact (-1 if never)", -1, 999, -1)
        previous = st.number_input("Previous Contacts", 0, 50, 0)

    st.divider()

    if st.button("🔍 Predict"):
        customer = {
            "age": age, "job": job, "marital": marital,
            "education": education, "default": default,
            "balance": balance, "housing": housing,
            "loan": loan, "contact": contact, "day": day,
            "month": month, "duration": duration,
            "campaign": campaign, "pdays": pdays,
            "previous": previous, "poutcome": poutcome
        }

        pred, proba = predict(customer)

        # Result Banner
        if pred == 1:
            st.success(f"✅ Likely to Subscribe — Probability: {proba:.2%}")
            st.balloons()
        else:
            st.error(f"❌ Unlikely to Subscribe — Probability: {proba:.2%}")

        # Gauge Chart
        st.plotly_chart(gauge_chart(proba), use_container_width=True)

        # Risk Profile
        risk_profile(customer, proba)

        # Save to history
        st.session_state.history.append({
            "Time":        datetime.now().strftime("%H:%M:%S"),
            "Age":         age,
            "Job":         job,
            "Balance":     balance,
            "Duration":    duration,
            "Probability": f"{proba:.2%}",
            "Result":      "✅ Yes" if pred == 1 else "❌ No"
        })

# ════════════════════════════════════════════════════════════
# TAB 2 — Batch Prediction
# ════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-header">📂 Batch Prediction</p>',
                unsafe_allow_html=True)
    st.write("Upload a CSV file with customer data to predict all at once")

    # Download template
    template = pd.DataFrame(columns=[
        'age','job','marital','education','default','balance',
        'housing','loan','contact','day','month','duration',
        'campaign','pdays','previous','poutcome'
    ])
    st.download_button(
        "📥 Download CSV Template",
        template.to_csv(index=False),
        "template.csv",
        "text/csv"
    )

    uploaded = st.file_uploader("Upload CSV", type=['csv'])

    if uploaded:
        df_batch = pd.read_csv(uploaded)
        st.write(f"✅ Loaded {len(df_batch)} customers")
        st.dataframe(df_batch.head())

        if st.button("🔍 Predict All"):
            results = []
            for _, row in df_batch.iterrows():
                pred, proba = predict(row.to_dict())
                results.append({
                    **row.to_dict(),
                    'Probability': f"{proba:.2%}",
                    'Prediction':  "✅ Yes" if pred == 1 else "❌ No"
                })

            results_df = pd.DataFrame(results)
            st.dataframe(results_df)

            # Summary
            yes_count = sum(1 for r in results if r['Prediction'] == "✅ Yes")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Customers", len(results))
            col2.metric("Will Subscribe",  yes_count)
            col3.metric("Won't Subscribe", len(results) - yes_count)

            # Download results
            st.download_button(
                "📥 Download Results",
                results_df.to_csv(index=False),
                "predictions.csv",
                "text/csv"
            )

# ════════════════════════════════════════════════════════════
# TAB 3 — Prediction History
# ════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-header">📜 Prediction History</p>',
                unsafe_allow_html=True)

    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True)

        col1, col2 = st.columns(2)
        yes = sum(1 for h in st.session_state.history if h['Result'] == "✅ Yes")
        col1.metric("Total Predictions", len(st.session_state.history))
        col2.metric("Subscriptions",     yes)

        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("No predictions yet — go to Single Prediction tab to start!")
