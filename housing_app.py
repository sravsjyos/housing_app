import streamlit as st
import numpy as np
import pandas as pd
import joblib


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


df = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv")


X = df.drop('medv', axis=1)
y = df['medv']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


joblib.dump(model, "house_model.pkl")
print("Model saved as house_model.pkl")
model = joblib.load("house_model.pkl")

st.title("🏠 House Price Prediction")
st.write("Enter the details:")

features = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 
            'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat']

input_data = [st.number_input(f"{feat}", value=0.0) for feat in features]

if st.button("Predict"):
    prediction = model.predict([input_data])
    st.success(f"Estimated House Price: ${prediction[0]*1000:.2f}")
    import joblib
joblib.dump(rf_model, "house_model.pkl")

