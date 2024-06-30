import streamlit as st
import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, recall_score, confusion_matrix,
    roc_auc_score, precision_score, f1_score, roc_curve, auc
)
from imblearn.over_sampling import SMOTE 
import joblib
import shap

# model and data path
MODEL_PATH = "Add model path"
DATA_PATH = "../Data/churn_dataset_cleaned.csv"
OE_PATH = "../Model/encoder.joblib"
scaler_path = "../Model/scaler.joblib"
X_train_path = "../Data/x_train.csv"
X_test_path = "../Data/x_test.csv"
Y_train_path = "../Data/y_train.csv"
Y_test_path = "../Data/y_test.csv"

st.set_page_config(page_title="DTH Customer Churn Project")

@st.cache_resource
def load_data():
    data = pd.read_csv(DATA_PATH)
    return data

def load_x_y(data_path):
    data = joblib.load(data_path)
    return data
# load model
def load_model(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    return model

# ordinal encoder function
def encoder(data, OE_PATH):
    cols = data.columns
    OE = joblib.load(OE_PATH)
    encode_data = OE.transform(data)
    return pd.DataFrame(encode_data, columns=cols)

# std scaler function
def scaler(data, scaler_path):
    scaler = joblib.load(scaler_path)
    X = scaler.transform(data)
    return X

# prediction function
def model_prediction(X, MODEL_PATH):
    rf = joblib.load(MODEL_PATH)
    pred = rf.predict(X)
    return pred

# calculate shap
def calculate_shap(model, X_train, X_test):
    explainer = shap.TreeExplainer(model)
    shap_values_cat_train = explainer.shap_values(X_train)
    shap_values_cat_test = explainer.shap_values(X_test)
    return explainer, shap_values_cat_train, shap_values_cat_test

# display shap summary
def display_shap_summary(shap_values_cat_train, X_train):
    shap.summay_plot(shap_values_cat_train, X_train, plot_type="bar", plot_size=(12,12)) 
    summary_fig, _ = plt.gcf(), plt.gca()
    st.pyplot(summary_fig)
    plt.close()

# summerize the shap and display shap in one function
def summary(model, X_train, X_test):
    explainer, shap_values_cat_train, shap_values_cat_test = calculate_shap(model, X_train, X_test)
    display_shap_summary(shap_values_cat_train,X_train)

# main function:
def main():
    model = load_model(MODEL_PATH)

    X_train = load_x_y(X_train_path)
    X_test = load_x_y(X_test_path)

    selection = st.radio("Select any of your choice:",("Predict the customer churn", "Feature importance"))

    if selection=="Predict the customer churn":
        AccountID  = st.number_input("Enter the account id:")
        Tenure = st.number_input("Enter the Tenure:")
        City_Tier = st.number_input("Enter the city tier of customer:")
        CC_contected_LY = st.number_input("Enter the CC contacted in last year:")
        Payment = st.selectbox("Enter the payment method:", ['Debit Card','Credit Card','E Wallet','Cash on Delivery','UPI'])
        Gender = st.selectbox ("Enter the Gender:", ['Male','Female'])
        service_score = st.number_input("Enter the service rating:")
        account_users = st.number_input("Enter the user count:")
        account_seg = st.selectbox("Enter the acccount segment:", ['Super','Regular Plus','HNI','Super Plus','Regular'])
        CC_agent_score = st.number_input("Enter the CC agent score:")
        Marital_status = st.selectbox("Enter the merital status:",['Married','Single','Divorce'])
        Rev_per_month = st.number_input("Enter the revenue per month:")
        Complain_LY = st.number_input("Enter the complaint raised last year:")
        Rev_growth_yoy = st.number_input("Enter the revenue growth yoy:")
        Coupon__used_for_payment = st.number_input("Enter the coupon used for payment:")
        Days_since_CC_connect = st.number_input("Enter the days since CC contacted:")
        cashback = st.number_input("Enter the cashback amount:")
        Login_device = st.selectbox("Enter the login device",['Mobile','Computer','Mobile & Computer'])

        confirm = st.button("Predict")

        # create predict option
        if confirm:
            new_data = pd.DataFrame({"AccountID": [AccountID],
                "Tenure": [Tenure],
                "City_Tier": [City_Tier],
                "CC_contected_LY": [CC_contected_LY],
                "Payment": [Payment],
                "Gender": [Gender],
                "service_score": [service_score],
                "account_users": [account_users],
                "account_seg": [account_seg],
                "CC_agent_score": [CC_agent_score],
                "Marital_status": [Marital_status],
                "Rev_per_month": [Rev_per_month],
                "Complain_LY": [Complain_LY],
                "Rev_growth_yoy": [Rev_growth_yoy],
                "Coupon__used_for_payment": [Coupon__used_for_payment],
                "Days_since_CC_connect": [Days_since_CC_connect],
                "cashback": [cashback],
                "Login_device":[Login_device]
                })
            
            encoder_d = encoder(new_data, OE_PATH)
            scaler_d = scaler(encoder_d, scaler_path)

            Churn_pred = model_prediction(scaler_d, MODEL_PATH)
            st.markdown(f"Final prediction is: {Churn_pred}", unsafe_allow_html=True)
            st.write(new_data.to_dict())

    elif selection=="Feature importance":
        summary(model, X_train, X_test)

if __name__ == "__main__":
    main()