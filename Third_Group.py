import streamlit as st
import joblib
import pandas as pd
import numpy as np

Model = joblib.load("Third_Group_Loan.pkl")
Inputs = joblib.load("Third_Group_Loan_Inputs.pkl")

def Prediction(Gender, Married, Dependents, Education, Self_Employed,LoanAmount, Loan_Amount_Term, Credit_History, Property_Area,log_Total_Income, log_Loan_Monthly_Paid, log_Income_After_Loan):
    df = pd.DataFrame(columns=Inputs)
    df.at[0,"Gender"] = Gender
    df.at[0,"Married"] = Married
    df.at[0,"Dependents"] = Dependents
    df.at[0,"Education"] = Education
    df.at[0,"Self_Employed"] = Self_Employed
    df.at[0,"LoanAmount"] = LoanAmount
    df.at[0,"Loan_Amount_Term"] = Loan_Amount_Term
    df.at[0,"Credit_History"] = Credit_History
    df.at[0,"Property_Area"] = Property_Area
    df.at[0,"log_Total_Income"] = log_Total_Income
    df.at[0,"log_Loan_Monthly_Paid"] = log_Loan_Monthly_Paid
    df.at[0,"log_Income_After_Loan"] = log_Income_After_Loan
    result = Model.predict(df)
    return result[0]

def Main():
    st.title("Loan Accepetance")
    Gender = st.selectbox("Gender" , ['Male', 'Female'])
    Married = st.selectbox("Married",['No', 'Yes'])
    Dependents = st.selectbox("Dependents",[ '0',  '1',  '2',  '3+'])
    Education = st.selectbox("Education",['Graduate', 'Not Graduate'])
    Self_Employed = st.selectbox("Self_Employed",['No', 'Yes'])
    LoanAmount = st.slider("LoanAmount in Thousands",min_value=9.0 , max_value=700.0 , step=1.0,value = 10.0)
    Loan_Amount_Term = st.slider("Loan_Amount_Term",min_value=12.0 , max_value=480.0 , step=1.0,value = 10.0)
    Credit_History = st.selectbox("Credit_History",[1, 0])
    Property_Area = st.selectbox("Property_Area",['Urban', 'Rural', 'Semiurban'])
    ApplicantIncome = st.slider("ApplicantIncome",min_value=0.0 , max_value=81000.0 , step=5.0,value = 500.0)
    CoapplicantIncome = st.slider("CoapplicantIncome",min_value=0.0 , max_value= 41667.0 , step=5.0,value = 500.0)
    
    
    
    total_income = ApplicantIncome + CoapplicantIncome
    log_Total_Income = np.log(total_income)
    Monthly_paid = (LoanAmount * 1000) / Loan_Amount_Term 
    log_Loan_Monthly_Paid= np.log(Monthly_paid)
    log_Income_After_Loan = np.log(ApplicantIncome - Monthly_paid)
    if st.button("Predict"):
        result = Prediction(Gender, Married, Dependents, Education, Self_Employed,LoanAmount, Loan_Amount_Term, Credit_History, Property_Area,log_Total_Income, log_Loan_Monthly_Paid, log_Income_After_Loan)
        list_result = ["Rejected" , "Accepted"]
        st.text(f"Your loan is {list_result[result]}")

    
    
Main()   
