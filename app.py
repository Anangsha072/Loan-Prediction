import streamlit as st
import pandas as pd
import pickle as pk

# Load model and scaler
model = pk.load(open('model.pkl', 'rb'))
scaler = pk.load(open('scaler.pkl', 'rb'))

# App title
st.markdown("<h1 style='text-align: center; color: #ff4b4b;'>Loan Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #ffffff;'>Check if your loan is likely to be approved</h4>", unsafe_allow_html=True)
st.write("---")

# Create two columns for inputs
col1, col2 = st.columns(2)

with col1:
    no_of_dep = st.slider('No of Dependents', 0, 5)
    grad = st.selectbox('Education', ['Graduated', 'Not Graduated'])
    self_emp = st.selectbox('Self Employed?', ['Yes', 'No'])
    Cibil = st.number_input('Cibil Score', min_value=0, max_value=1000, value=700, step=1)

with col2:
    Annual_Income = st.number_input('Annual Income', min_value=0, max_value=20000000, value=5000000, step=100000)
    Loan_Amount = st.number_input('Loan Amount', min_value=0, max_value=20000000, value=5000000, step=100000)
    Loan_Dur = st.slider('Loan Duration (Years)', 0, 30, 12)
    Assets = st.number_input('Assets', min_value=0, max_value=50000000, value=1000000, step=50000)

# Convert categorical inputs to numeric
grad_s = 0 if grad == 'Graduated' else 1
emp_s = 0 if self_emp == 'No' else 1

# Prediction button
if st.button("Predict"):
    # Create dataframe for prediction
    pred_data = pd.DataFrame([[no_of_dep, grad_s, emp_s, Annual_Income, Loan_Amount, Loan_Dur, Cibil, Assets]],
                             columns=['no_of_dependents','education','self_employed','income_annum',
                                      'loan_amount','loan_term','cibil_score','Assets'])
    
    pred_data_scaled = scaler.transform(pred_data)
    prediction = model.predict(pred_data_scaled)

    # Display result with color
    if prediction[0] == 1:
        st.success("✅ Congratulations! Loan is Approved")
    else:
        st.error("❌ Sorry! Loan is Rejected")



