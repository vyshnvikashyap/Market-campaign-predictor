
import streamlit as st
import pandas as pd
import joblib

rf       = joblib.load('model.pkl')
scaler   = joblib.load('scaler.pkl')
features = joblib.load('features.pkl')
cat_cols = joblib.load('cat_cols.pkl')

st.set_page_config(page_title="Bank Deposit Predictor", page_icon="🏦")
st.title("🏦 Bank Deposit Subscription Predictor")
st.write("Fill in customer details to predict subscription likelihood")

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
    poutcome = st.selectbox("Previous Outcome", ["success","failure","unknown","other"])

day      = st.slider("Day of Month", 1, 31, 15)
pdays    = st.number_input("Days Since Last Contact (-1 if never)", -1, 999, -1)
previous = st.number_input("Previous Contacts", 0, 50, 0)

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

    sample     = pd.DataFrame([customer])
    sample_enc = pd.get_dummies(sample, columns=cat_cols, drop_first=True)
    sample_enc = sample_enc.reindex(columns=features, fill_value=0)
    sample_sc  = scaler.transform(sample_enc)

    pred  = rf.predict(sample_sc)[0]
    proba = rf.predict_proba(sample_sc)[0][1]

    if pred == 1:
        st.success(f"✅ Likely to Subscribe — Probability: {proba:.2%}")
        st.balloons()
    else:
        st.error(f"❌ Unlikely to Subscribe — Probability: {proba:.2%}")
