import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA


@st.cache_data
def get_data():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
    df = pd.read_csv(url, parse_dates=['Month'], index_col='Month')
    #df1 = df.reset_index()
    return df
df1 = get_data()


uploaded_file = st.sidebar.file_uploader(
    "Upload any CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    df2 = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")

b6 = st.sidebar.radio("Select Data",["Default data",'Uploaded data'])
if b6=="Uploaded data":
    date1 = st.sidebar.selectbox("Select Time Column",df2.columns)
    value1 = st.sidebar.selectbox("Select Value Column",df2.columns,index=1)
    df = df2[[date1,value1]]
    df0 =df.rename(columns={date1:'DT',value1:'Value'})
    df = df0.set_index('DT')
else:
    df1 = df1.reset_index()
    df0 = df1.rename(columns={'Month': 'DT', 'Passengers': 'Value'})
    df = df0.set_index('DT')



b1 = st.sidebar.checkbox("View Data")
if b1:
    st.dataframe(df)

b2 = st.sidebar.checkbox("Plot")
if b2:
    fig, ax = plt.subplots()
    ax.plot(df.index, df["Value"])
    #ax.set_title("Monthly Airline Passengers")
    ax.set_xlabel("DT")
    ax.set_ylabel("Value")
    st.pyplot(fig)

AR1 = st.sidebar.number_input("Enter AR no.",0,5,2,1)
I1 = st.sidebar.number_input("Enter I no.",0,5,0,1)
MA1 = st.sidebar.number_input("Enter MA no.",0,5,2,1)
model = ARIMA(df, order=(AR1,I1,MA1))
arima_fit = model.fit()

b3 = st.sidebar.button("Summary")
if b3:
    st.write(arima_fit.summary())

b4 = st.sidebar.checkbox('Forcast')
if b4:
    pn = st.number_input("Enter Number",2,50,10)
    b5 = st.button("Predict")
    if b5:
        forecast = arima_fit.forecast(steps=pn)
        st.write(forecast)
        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(df.index, df['Value'], label='Actual')
        ax.plot(forecast.index, forecast, label='Forecast')

        #ax.set_title("Actual vs Forecast - Airline Passengers")
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)