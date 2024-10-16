#For data processing
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score
from imblearn.over_sampling import SMOTE
import sklearn.metrics as metrics
from sklearn.utils import resample
import streamlit as st

#for data visualization
import seaborn as sns
import matplotlib.pyplot as plt


#For Machine Learning
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


@st.cache_data
def load_data(file):
    data = pd.read_csv(file)
    return data

def preprocess_data(data):

    #For Transaction_Amount, here the data is first being grouped by the category as the amount being spent may vary
# based on the categoty of the merchant similar will be done for Customer_Age and Customer_Income_Bracket.

    data['Transaction_Amount'] = data['Transaction_Amount'].fillna(
        data.groupby('Merchant_Category')['Transaction_Amount'].transform('median')
    )

    data['Customer_Age'] = data['Customer_Age'].fillna(
        data.groupby('Merchant_Category')['Customer_Age'].transform('median')
    )

    data['Customer_Income_Bracket'] = data['Customer_Income_Bracket'].fillna(
        data.groupby('Merchant_Category')['Customer_Income_Bracket'].transform(lambda x: x.mode()[0] if not x.mode().empty else np.nan)
    )

    #There are a total of 9778 duplicate datas in the given dataset. We need to remove them before proceeding.

    data = data.drop_duplicates(keep='last')

    total_duplicates = data.duplicated().sum()
    print(f'Total duplicate rows: {total_duplicates}')

    #Separateing date and time

    # Step 1: Convert to datetime if it's not already
    data['Transaction_Date'] = pd.to_datetime(data['Transaction_Date'], errors='coerce')

    # Step 2: Split into separate Date and Time columns
    data['Date'] = data['Transaction_Date'].dt.date
    data['Time'] = data['Transaction_Date'].dt.time

    data.drop(['Transaction_Time', 'Transaction_Date'], axis=1, inplace=True)

    Fraud = data[data['Fraud_Flag']==1].shape[0]

    Not_Fraud = data[data['Fraud_Flag']==0].shape[0]

    # separate categorical and numeric datas

    cat_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
    num_columns = data.select_dtypes(include=['int64', 'float64']).columns.tolist()

    le = LabelEncoder()
    # Applying label encoding to each of the categorical columns
    for col in cat_columns:
        data[col] = le.fit_transform(data[col])
    # Now, the dataset should be preprocessed and ready for feature 
    #selection and modeling

    return data

# Step 3: Sidebar for user input
st.sidebar.title("Credit Card fraud Detection")

# Step 4: File uploader for CSV input
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    data = load_data(uploaded_file)
    data = preprocess_data(data)
    
        # Step 6: Allow the user to display the dataset
    if st.sidebar.checkbox("Show Dataset"):
        st.subheader("Credit card dataset")
        st.write(data)
    
    fraud_txn = data[data['Fraud_Flag']==1]
    not_fraud_txn = data[data['Fraud_Flag']==0]
        
    st.sidebar.subheader("Select Features")
    selected_features = st.sidebar.multiselect(
        "Select features to include in the model", 
        options=fraud_txn
    )

    test_size = st.sidebar.slider(
    "Select test size", 
    min_value=0.1, max_value=0.5, step=0.05, value=0.2
    )