import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, recall_score, confusion_matrix,
    roc_auc_score, precision_score, f1_score, roc_curve, auc
)
from imblearn.over_sampling import SMOTE 
import joblib

# load the dataset
datapath = "../Data/churn_dataset_cleaned.csv"
df = pd.read_csv(datapath)

# data pre-processing pipeline 
cols = df.columns

X = df.drop('Churn', axis=1)
Y = df['Churn']

# encoder instance
OE = OrdinalEncoder()
OE.fit(X)
ndf = OE.transform(X)
X = pd.DataFrame(ndf, columns=X.columns)

# SMOTE over samplling
over_sampling = SMOTE(sampling_strategy='minority', random_state=42)
X_n, y_n = over_sampling.fit_resample(X,Y)

# train and test split 
X_train, X_test, y_train, y_test = train_test_split(X_n, y_n, 
                                                    train_size=0.75,
                                                    random_state=42)

# standard scaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# random forest modeling
def random_forest(X_train, y_train, X_test):
    model = RandomForestClassifier(n_estimators=500, max_depth=15, random_state=42)
    model.fit(X_train,y_train)
    return model 

model_rf = random_forest(X_train, y_train, X_test)
# save the model
model_path = "../Model/rf_model.joblib"
joblib.dump(model_rf, model_path)

# save the encoder and scaler
joblib.dump(OE, "../Model/encoder.joblib")
joblib.dump(scaler, "../Model/scaler.joblib")

# save the train and test files 
joblib.dump(X_train, "../Data/x_train.csv")
joblib.dump(X_test, "../Data/x_test.csv")
joblib.dump(y_train, "../Data/y_train.csv")
joblib.dump(y_test, "../Data/y_test.csv")





