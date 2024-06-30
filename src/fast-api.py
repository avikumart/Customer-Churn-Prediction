import uvicorn
import pandas as pd
from fastapi import FastAPI
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

# creat fastapi app
app = FastAPI(title="Customer Churn Prediction API", version="1.0")

@app.get('/')
def index():
    return {'message':'Churn Prediction API'}

# define api endpoint for prediction api
@app.post('/predict')
def predict_churn(data: dict):
    ndata = encoder(data, OE_PATH)
    scaled_data = scaler(ndata, scaler_path)
    churn_pred = model_prediction(scaled_data, MODEL_PATH)
    return {'Churn prediction': churn_pred}

if __name__ == '__main__':
    uvicorn.run("fast-api:app", host='127.0.0.1', port=3838)