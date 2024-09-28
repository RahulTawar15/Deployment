import pandas as pd
import seaborn as sns
import streamlit as st 
import joblib
from sklearn.linear_model import LogisticRegression
from pickle import dump
from pickle import load

st.title('Model Deployment: Logistic Regression')

   
df = sns.load_dataset("iris")
x=df.iloc[:,:4]
y=df.iloc[:,4:]

encoder=joblib.load('encoder.pkl')
scaler=joblib.load('scaler.pkl')
model=joblib.load('model.pkl')

st.subheader('Dataset')
st.write(df)

scaled_features=scaler.transform(x)



prediction = model.predict(scaled_features)

st.subheader('Actual Classes')
st.write(y)

final_predictions=encoder.inverse_transform(list(prediction))
st.subheader('Predicted Class')
st.write(final_predictions)