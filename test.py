import pandas as pd
import seaborn as sns
import numpy as np
import streamlit as st 
import joblib
from sklearn.linear_model import LogisticRegression
from pickle import dump
from pickle import load

st.title('Model Deployment: Logistic Regression')

   
#df = sns.load_dataset("iris")
#x=df.iloc[:,:4]
#y=df.iloc[:,4:]

encoder=joblib.load('encoder.pkl')
scaler=joblib.load('scaler.pkl')
model=joblib.load('model.pkl')

st.subheader('Please enter your input')

num1=st.number_input("Enter the SL", value=0.0)
num2=st.number_input("Enter the SW" ,value=0.0)
num3=st.number_input("Enter the PL", value=0.0)
num4=st.number_input("Enter the PW" ,value=0.0)

if st.button("Predict"):
    
    features=np.array([[num1,num2,num3,num4]])



    scaled_features=scaler.transform(features)



    prediction = model.predict(scaled_features)

#st.subheader('Actual Classes')
#st.write(y)

    final_predictions=encoder.inverse_transform(list(prediction))
    st.subheader('Predicted Class')
    st.write(final_predictions)