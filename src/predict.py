import pandas as pd
import joblib

# model path
MODEL_PATH = "add model path"
OE_PATH = "../Model/encoder.joblib"
scaler_path = "../Model/scaler.joblib"

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

if __name__ == "__main__":
    AccountID  = input("Enter the account id:")
    Tenure = input("Enter the Tenure:")
    City_Tier = input("Enter the city tier of customer:")
    CC_contected_LY = input("Enter the CC contacted in last year:")
    Payment = input("Enter the payment method: ['Debit Card','Credit Card','E Wallet','Cash on Delivery','UPI']")
    Gender = input("Enter the Gender:['Male','Female']")
    service_score = input("Enter the service rating:")
    account_users = input("Enter the user count:")
    account_seg = input("Enter the acccount segment:['Super','Regular Plus','HNI','Super Plus','Regular']")
    CC_agent_score = input("Enter the CC agent score:")
    Marital_status = input("Enter the merital status:['Married','Single','Divorce']")
    Rev_per_month = input("Enter the revenue per month:")
    Complain_LY = input("Enter the complaint raised last year:")
    Rev_growth_yoy = input("Enter the revenue growth yoy:")
    Coupon__used_for_payment = input("Enter the coupon used for payment:")
    Days_since_CC_connect = input("Enter the days since CC contacted:")
    cashback = input("Enter the cashback amount:")
    Login_device = input("Enter the login device: ['Mobile','Computer','Mobile & Computer']")

    new_custom_data = pd.DataFrame({"AccountID": [AccountID],
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
    
    # encode the data
    encoded_data = encoder(new_custom_data,OE_PATH)

    # scale the data
    scaled_data = scaler(encoded_data, scaler_path)

    # predict the value
    pred = model_prediction(scaled_data, MODEL_PATH)

    print(f"customer churn prediction is: {pred}")