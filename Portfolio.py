import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
np.float_ = np.float64
np.complex_ = np.complex128
from pandas_datareader import data as pdr
from datetime import datetime, timedelta
import pyomo.environ as pyo
import math

st.sidebar.header("Being a Sustainable Investor")
st.sidebar.markdown("""
Imagine a world where a sustainability score is attached to every publicly traded company. 
Any portfolio of stocks that you own has a weighted average sustainability score depending on what and how much you buy. 
Pair that with your investment capital, and the overall volatility you are willing to accept for your investment, and you have a linear optimization problem.

I have created a web app that takes on data on 50 random stocks from the S&P500 Index and attaches 50 “random” sustainability scores to them. 
It then maximizes returns on your portfolio based on your desired Sustainability Score, Volatility and Investment Capital. 
You can view the app and interact with the constraints and view the changes in your profits on the go! 
It also displays non-zero shadow prices and automatically interprets them.
""")
st.sidebar.markdown("Payam Saeedi")

# Read S&P500 symbols
tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
symbols = tickers.Symbol.to_list()

yf.pdr_override()

import warnings
warnings.filterwarnings("ignore", message="The 'unit' keyword in TimedeltaIndex construction is deprecated", category=FutureWarning, module="yfinance.utils")

startdate = datetime(2023,1,1)
enddate = datetime.today().date() - timedelta(days=1)

@st.cache_data
def capture_data(symbols, startdate, enddate):
    """
    Capture stock data. Tries live Yahoo Finance first. 
    Falls back to stockdata.csv if live data cannot be captured.
    Always returns 50 stocks (or fewer if not enough available).
    """
    data_list = []
    try:
        for symbol in symbols:
            df = pdr.get_data_yahoo(symbol, start=startdate, end=enddate)
            if df.empty:
                continue
            df['Return'] = df['Adj Close'].pct_change().fillna(0)
            df['Volatility'] = df['Return'].rolling(window=21).std().fillna(0)
            df['ESG score'] = np.random.randint(0, 101)
            data_list.append({
                'Ticker': symbol,
                'Closing Price': df['Adj Close'].iloc[-1],
                'Return': df['Return'].mean(),
                'Volatility': df['Volatility'].mean(),
                'ESG score': df['ESG score'].iloc[-1]
            })
        if len(data_list) < 10:  # Arbitrary threshold to decide if live data is unreliable
            raise ValueError("Insufficient live data")
        new_df = pd.DataFrame(data_list).set_index('Ticker')
        new_df = new_df.sample(n=min(50, len(new_df)))  # Ensure 50 stocks
    except Exception as e:
        st.warning(f"Live data could not be captured ({e}). Using local CSV instead.")
        new_df = pd.read_csv('stockdata.csv')
        if 'ESG Score' in new_df.columns:
            new_df.rename(columns={"ESG Score": "ESG score"}, inplace=True)
        new_df = new_df.set_index("Ticker")
        new_df = new_df.dropna()
        new_df = new_df.sample(n=min(50, len(new_df)))  # Ensure 50 stocks

    return new_df

# Capture the data
data = capture_data(symbols, startdate, enddate)

# User Inputs
st.write("Set minimum ESG score:")
esg_score = st.slider("ESG Score", min_value=0, max_value=100, value=80, step=1)

st.write("Set maximum portfolio volatility:")
volatility_score = st.slider("Volatility", min_value=0.0, max_value=1.0, value=0.5, step=0.1)

st.write("Set your investment capital:")
investment_capital = st.slider("Investment Capital", min_value=2000, max_value=1000000, value=100000, step=100)

# Pyomo Model
model = pyo.ConcreteModel()
num_stocks = len(data)
model.x = pyo.Var(range(num_stocks), within=pyo.NonNegativeReals)

model.P = {i: data['Closing Price'].iloc[i] for i in range(num_stocks)}
model.R = {i: data['Return'].iloc[i] for i in range(num_stocks)}
model.ESG = {i: data['ESG score'].iloc[i] for i in range(num_stocks)}
model.Volatility = {i: data['Volatility'].iloc[i] for i in range(num_stocks)}

model.IC = pyo.Param(initialize=investment_capital)
model.Threshold = pyo.Param(initialize=esg_score)
model.VolatilityConstraint = pyo.Param(initialize=volatility_score)

model.obj = pyo.Objective(expr=sum(model.x[i] * model.P[i] * model.R[i] for i in range(num_stocks)), sense=pyo.maximize)

# Constraints
model.budget_constraint = pyo.Constraint(expr=sum(model.x[i] * model.P[i] for i in range(num_stocks)) == model.IC)
model.esg_constraint = pyo.Constraint(expr=sum(model.ESG[i] * model.x[i]* model.P[i] for i in range(num_stocks)) >= model.Threshold * sum(model.x[i]* model.P[i] for i in range(num_stocks)))
model.volatility_constraint = pyo.Constraint(expr=sum(model.Volatility[i] * model.x[i] for i in range(num_stocks)) <= model.VolatilityConstraint * sum(model.x[i] for i in range(num_stocks)))

def budget_allocation(model, i):
    return (model.x[i] * model.P[i] <= 0.5 * model.IC)
model.budgetallocation = pyo.Constraint(range(num_stocks), rule=budget_allocation)

model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

opt = pyo.SolverFactory('glpk')
results = opt.solve(model)

def custom_floor(x):
    return 0 if 0 < x < 1 else math.floor(x)

# Display Results
st.write("Return on Portfolio: ", model.obj())  
st.write("Selected Stocks and Investment Amounts:")
for i in range(num_stocks):
    if model.x[i].value != 0:
        st.write(data.index[i], custom_floor(model.x[i].value))

st.subheader("Non-zero Shadow Prices (Dual Values) with Interpretations:")
for c in model.component_objects(pyo.Constraint, active=True):
    for index in c:
        shadow_price = model.dual[c[index]]
        if shadow_price != 0:
            if c == model.budget_constraint:
                st.write(f"Budget Constraint - Shadow Price: {shadow_price}")
            elif c == model.esg_constraint:
                st.write(f"ESG Constraint - Shadow Price: {shadow_price}")
            elif c == model.volatility_constraint:
                st.write(f"Volatility Constraint - Shadow Price: {shadow_price}")
