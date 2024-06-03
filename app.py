import streamlit as st
import joblib # type: ignore
import numpy as np


loaded_pipe = joblib.load('model1.pkl')
load_data=joblib.load('data')
st.title("Laptop price prediction ")

company=st.selectbox('Brand',load_data['Company'].unique())

Type=st.selectbox('Type',load_data['TypeName'].unique())

Ram=st.selectbox('Ram[GB]',load_data['Ram'].unique())

weight=st.selectbox('weight',load_data['Weight'].unique())

Touchscreen=st.selectbox('Touchscreen',load_data['Touchscreen'].unique())

IPS=st.selectbox('IPS',load_data['IPS'].unique())


PPI=st.selectbox('PPI',load_data['ppi'].unique())


CpuBrand=st.selectbox('CpuBrand',load_data['Cpu Brand'].unique())


OS=st.selectbox('OS',load_data['os'].unique())

GpuBrand=st.selectbox('GpuBrand',load_data['GPU brand'].unique())

if st.button('Predict Price'):
    query=np.array([company,Type,Ram,weight,Touchscreen,IPS,PPI,CpuBrand,OS,GpuBrand])
    query=query.reshape(1,10)
    
    st.title("The predicted price of this configuration is " + str(int(np.exp(loaded_pipe.predict(query)[0]))))


