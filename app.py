# Find this line:
tab1, tab2, tab3 = st.tabs([
    "🔍 Single Prediction",
    "📂 Batch Prediction",
    "📜 Prediction History"
])

# Replace with:
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Single Prediction",
    "📂 Batch Prediction",
    "📜 Prediction History",
    "📊 Dashboard"
])
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

# ── Sidebar — Model Metrics ──────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bank.png", width=80)
    st.title("🏦 Bank Predictor")
    st.divider()
    # ── What-if Analysis ─────────────────────────────────────────
st.divider()
st.markdown('<p class="section-header">🔄 What-if Analysis</p>',
            unsafe_allow_html=True)
st.write("See how changing values affects subscription probability")

col1, col2 = st.columns(2)

with col1:
    whatif_duration = st.slider(
        "📞 What if call duration was...",
        min_value  = 0,
        max_value  = 3000,
        value      = duration,
        step       = 30
    )
    whatif_balance = st.slider(
        "💰 What if balance was...",
        min_value  = -5000,
        max_value  = 50000,
        value      = balance,
        step       = 500
    )

with col2:
    whatif_campaign = st.slider(
        "📱 What if number of calls was...",
        min_value = 1,
        max_value = 50,
        value     = campaign,
        step      = 1
    )
    whatif_age = st.slider(
        "👤 What if age was...",
        min_value = 18,
        max_value = 95,
        value     = age,
        step      = 1
    )

# Run what-if prediction
whatif_customer = {
    "age":       whatif_age,
    "job":       job,
    "marital":   marital,
    "education": education,
    "default":   default,
    "balance":   whatif_balance,
    "housing":   housing,
    "loan":      loan,
    "contact":   contact,
    "day":       day,
    "month":     month,
    "duration":  whatif_duration,
    "campaign":  whatif_campaign,
    "pdays":     pdays,
    "previous":  previous,
    "poutcome":  poutcome
}

whatif_pred, whatif_proba = predict(whatif_customer)

# Compare original vs whatif
col1, col2, col3 = st.columns(3)

col1.metric(
    "Original Probability",
    f"{proba:.2%}"
)
col2.metric(
    "New Probability",
    f"{whatif_proba:.2%}",
    delta = f"{(whatif_proba - proba)*100:.1f}%"
)
col3.metric(
    "New Prediction",
    "✅ Yes" if whatif_pred == 1 else "❌ No"
)

# Side by side gauge
col1, col2 = st.columns(2)
with col1:
    st.write("**Original**")
    st.plotly_chart(
        gauge_chart(proba),
        use_container_width=True
    )
with col2:
    st.write("**What-if**")
    st.plotly_chart(
        gauge_chart(whatif_proba),
        use_container_width=True
    )

    # Model Info
    st.markdown("### 🤖 Model Info")
    st.info("""
    **Model:** Random Forest
    **Trees:** 100
    **Task:** Binary Classification
    """)
    st.divider()

    # Model Metrics
    st.markdown("### 📊 Model Performance")

    metrics = {
        "Accuracy":  "89.2%",
        "Precision": "82.1%",
        "Recall":    "74.3%",
        "F1 Score":  "78.0%",
        "ROC AUC":   "91.2%"
    }

    for metric, value in metrics.items():
        col1, col2 = st.columns(2)
        col1.markdown(f"**{metric}**")
        col2.markdown(f"`{value}`")

    st.divider()

    # Feature Importance Mini Chart
    st.markdown("### 🎯 Top Features")
    
    importance_data = {
        "Duration":  0.32,
        "Balance":   0.18,
        "Age":       0.14,
        "Pdays":     0.10,
        "Campaign":  0.08,
        "Month":     0.07,
        "Poutcome":  0.06,
        "Job":       0.05
    }

    for feature, score in importance_data.items():
        st.markdown(f"**{feature}**")
        st.progress(score)
        st.caption(f"Importance: {score:.2f}")

    st.divider()

    # Dataset Info
    st.markdown("### 📁 Dataset Info")
    st.success("""
    **Dataset:** Bank Marketing
    **Rows:** 45,211
    **Features:** 16
    **Target:** Deposit (Yes/No)
    """)

    st.divider()

    # Footer
    st.caption("Built with ❤️ using Streamlit")
    st.caption("Model: Random Forest Classifier")

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

# ════════════════════════════════════════════════════════════
# TAB 4 — Dashboard
# ════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<p class="section-header">📊 Prediction Dashboard</p>',
                unsafe_allow_html=True)

    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        probs_hist = history_df['Probability'].str.rstrip('%').astype(float)

        # ── Top Metrics ──────────────────────────────────────
        col1, col2, col3, col4 = st.columns(4)
        yes_count = (history_df['Result'] == '✅ Yes').sum()
        no_count  = (history_df['Result'] == '❌ No').sum()

        col1.metric("Total Predictions", len(history_df))
        col2.metric("Subscriptions",     yes_count)
        col3.metric("Rejections",        no_count)
        col4.metric("Success Rate",      f"{yes_count/len(history_df):.1%}")

        st.divider()

        col1, col2 = st.columns(2)

        # ── Pie Chart ────────────────────────────────────────
        with col1:
            fig_pie = go.Figure(go.Pie(
                labels = ['Will Subscribe ✅', 'Will Not ❌'],
                values = [yes_count, no_count],
                hole   = 0.4,
                marker = dict(colors=['#2ecc71','#e74c3c'])
            ))
            fig_pie.update_layout(
                title  = 'Overall Results',
                height = 350
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # ── Probability Trend ────────────────────────────────
        with col2:
            fig_trend = go.Figure(go.Scatter(
                x    = list(range(1, len(history_df)+1)),
                y    = probs_hist,
                mode = 'lines+markers',
                line = dict(color='#3498db', width=2),
                marker = dict(size=8)
            ))
            fig_trend.update_layout(
                title       = 'Probability Trend Over Time',
                xaxis_title = 'Prediction #',
                yaxis_title = 'Probability (%)',
                height      = 350
            )
            st.plotly_chart(fig_trend, use_container_width=True)

        col3, col4 = st.columns(2)

        # ── Age Distribution ─────────────────────────────────
        with col3:
            fig_age = go.Figure()
            fig_age.add_trace(go.Histogram(
                x    = history_df[history_df['Result']=='✅ Yes']['Age'],
                name = 'Subscribed',
                marker_color = '#2ecc71',
                opacity = 0.7
            ))
            fig_age.add_trace(go.Histogram(
                x    = history_df[history_df['Result']=='❌ No']['Age'],
                name = 'Not Subscribed',
                marker_color = '#e74c3c',
                opacity = 0.7
            ))
            fig_age.update_layout(
                title       = 'Age Distribution by Result',
                xaxis_title = 'Age',
                yaxis_title = 'Count',
                barmode     = 'overlay',
                height      = 350
            )
            st.plotly_chart(fig_age, use_container_width=True)

        # ── Job Distribution ─────────────────────────────────
        with col4:
            job_counts = history_df.groupby(['Job','Result']).size().unstack(fill_value=0)
            fig_job = go.Figure()

            if '✅ Yes' in job_counts.columns:
                fig_job.add_trace(go.Bar(
                    name = 'Subscribed',
                    x    = job_counts.index,
                    y    = job_counts['✅ Yes'],
                    marker_color = '#2ecc71'
                ))
            if '❌ No' in job_counts.columns:
                fig_job.add_trace(go.Bar(
                    name = 'Not Subscribed',
                    x    = job_counts.index,
                    y    = job_counts['❌ No'],
                    marker_color = '#e74c3c'
                ))

            fig_job.update_layout(
                title       = 'Results by Job',
                xaxis_title = 'Job',
                yaxis_title = 'Count',
                barmode     = 'group',
                height      = 350
            )
            st.plotly_chart(fig_job, use_container_width=True)

    else:
        st.info("Make some predictions first to see dashboard! 📊")
        # ════════════════════════════════════════════════════════════
# TAB 4 — Dashboard
# ════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<p class="section-header">📊 Prediction Dashboard</p>',
                unsafe_allow_html=True)

    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        probs_hist = history_df['Probability'].str.rstrip('%').astype(float)

        # ── Top Metrics ──────────────────────────────────────
        col1, col2, col3, col4 = st.columns(4)
        yes_count = (history_df['Result'] == '✅ Yes').sum()
        no_count  = (history_df['Result'] == '❌ No').sum()

        col1.metric("Total Predictions", len(history_df))
        col2.metric("Subscriptions",     yes_count)
        col3.metric("Rejections",        no_count)
        col4.metric("Success Rate",      f"{yes_count/len(history_df):.1%}")

        st.divider()

        col1, col2 = st.columns(2)

        # ── Pie Chart ────────────────────────────────────────
        with col1:
            fig_pie = go.Figure(go.Pie(
                labels = ['Will Subscribe ✅', 'Will Not ❌'],
                values = [yes_count, no_count],
                hole   = 0.4,
                marker = dict(colors=['#2ecc71','#e74c3c'])
            ))
            fig_pie.update_layout(
                title  = 'Overall Results',
                height = 350
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # ── Probability Trend ────────────────────────────────
        with col2:
            fig_trend = go.Figure(go.Scatter(
                x    = list(range(1, len(history_df)+1)),
                y    = probs_hist,
                mode = 'lines+markers',
                line = dict(color='#3498db', width=2),
                marker = dict(size=8)
            ))
            fig_trend.update_layout(
                title       = 'Probability Trend Over Time',
                xaxis_title = 'Prediction #',
                yaxis_title = 'Probability (%)',
                height      = 350
            )
            st.plotly_chart(fig_trend, use_container_width=True)

        col3, col4 = st.columns(2)

        # ── Age Distribution ─────────────────────────────────
        with col3:
            fig_age = go.Figure()
            fig_age.add_trace(go.Histogram(
                x    = history_df[history_df['Result']=='✅ Yes']['Age'],
                name = 'Subscribed',
                marker_color = '#2ecc71',
                opacity = 0.7
            ))
            fig_age.add_trace(go.Histogram(
                x    = history_df[history_df['Result']=='❌ No']['Age'],
                name = 'Not Subscribed',
                marker_color = '#e74c3c',
                opacity = 0.7
            ))
            fig_age.update_layout(
                title       = 'Age Distribution by Result',
                xaxis_title = 'Age',
                yaxis_title = 'Count',
                barmode     = 'overlay',
                height      = 350
            )
            st.plotly_chart(fig_age, use_container_width=True)

        # ── Job Distribution ─────────────────────────────────
        with col4:
            job_counts = history_df.groupby(['Job','Result']).size().unstack(fill_value=0)
            fig_job = go.Figure()

            if '✅ Yes' in job_counts.columns:
                fig_job.add_trace(go.Bar(
                    name = 'Subscribed',
                    x    = job_counts.index,
                    y    = job_counts['✅ Yes'],
                    marker_color = '#2ecc71'
                ))
            if '❌ No' in job_counts.columns:
                fig_job.add_trace(go.Bar(
                    name = 'Not Subscribed',
                    x    = job_counts.index,
                    y    = job_counts['❌ No'],
                    marker_color = '#e74c3c'
                ))

            fig_job.update_layout(
                title       = 'Results by Job',
                xaxis_title = 'Job',
                yaxis_title = 'Count',
                barmode     = 'group',
                height      = 350
            )
            st.plotly_chart(fig_job, use_container_width=True)

    else:
        st.info("Make some predictions first to see dashboard! 📊")
