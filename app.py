# app.py
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn import metrics

# ----------------- Load and Preprocess Data -----------------
@st.cache_data
def load_data():
    # Load dataset
    car_dataset = pd.read_csv("car data.csv")
    
    # Encode categorical columns
    car_dataset.replace({"Fuel_Type": {'Petrol': 0, 'Diesel': 1, 'CNG': 2}}, inplace=True)
    car_dataset.replace({"Seller_Type": {'Dealer': 0, 'Individual': 1}}, inplace=True)
    car_dataset.replace({"Transmission": {'Manual': 0, 'Automatic': 1}}, inplace=True)
    
    return car_dataset

car_dataset = load_data()

# Features and target
X = car_dataset.drop(['Car_Name', 'Selling_Price'], axis=1)
Y = car_dataset['Selling_Price']

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=2)

# ----------------- Train Models -----------------
@st.cache_resource
def train_models():
    # Linear Regression
    lin_reg_model = LinearRegression()
    lin_reg_model.fit(X_train, Y_train)
    
    # Lasso Regression
    lasso_reg_model = Lasso()
    lasso_reg_model.fit(X_train, Y_train)
    
    return lin_reg_model, lasso_reg_model

lin_reg_model, lasso_reg_model = train_models()

# ----------------- Streamlit App -----------------
st.title("Car Price Prediction App")
st.write("""
This app predicts the **Selling Price of a car** based on various features.
""")

# Sidebar for user input
st.sidebar.header("Enter Car Details")

def user_input_features():
    Year = st.sidebar.number_input("Year of Manufacture", min_value=1990, max_value=2025, value=2015)
    Present_Price = st.sidebar.number_input("Present Price (in Lakhs)", min_value=0.0, value=5.0)
    Kms_Driven = st.sidebar.number_input("Kilometers Driven", min_value=0, value=5000)
    Fuel_Type = st.sidebar.selectbox("Fuel Type", ("Petrol", "Diesel", "CNG"))
    Seller_Type = st.sidebar.selectbox("Seller Type", ("Dealer", "Individual"))
    Transmission = st.sidebar.selectbox("Transmission Type", ("Manual", "Automatic"))
    Owner = st.sidebar.selectbox("Number of Previous Owners", (0,1,2,3))
    
    # Encode categorical inputs
    fuel_map = {'Petrol':0, 'Diesel':1, 'CNG':2}
    seller_map = {'Dealer':0, 'Individual':1}
    trans_map = {'Manual':0, 'Automatic':1}
    
    data = {
        'Year':[Year],
        'Present_Price':[Present_Price],
        'Kms_Driven':[Kms_Driven],
        'Fuel_Type':[fuel_map[Fuel_Type]],
        'Seller_Type':[seller_map[Seller_Type]],
        'Transmission':[trans_map[Transmission]],
        'Owner':[Owner]
    }
    
    features = pd.DataFrame(data)
    return features

input_df = user_input_features()

# Choose model
model_choice = st.selectbox("Select Model for Prediction", ["Linear Regression", "Lasso Regression"])

if st.button("Predict Selling Price"):
    if model_choice == "Linear Regression":
        prediction = lin_reg_model.predict(input_df)
    else:
        prediction = lasso_reg_model.predict(input_df)
    
    st.success(f"The predicted selling price of the car is: ₹ {round(prediction[0],2)} Lakhs")
