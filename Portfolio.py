import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_datareader as web
from pandas_datareader import data as pdr
from datetime import datetime, timedelta
import pyomo.environ as pyo
import math

st.sidebar.header("Being a Sustainable Investor")
st.sidebar.markdown("""Multicriteria Sustainable Systems Final Project""")
st.sidebar.markdown("""Payam Saeedi""")

# Read and print the stock tickers that make up S&P500
tickers = pd.read_html(
    'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
symbols = tickers.Symbol
symbols = symbols.to_list()

yf.pdr_override()

import warnings
warnings.filterwarnings("ignore", message="The 'unit' keyword in TimedeltaIndex construction is deprecated and will be removed in a future version. Use pd.to_timedelta instead.", category=FutureWarning, module="yfinance.utils")


startdate = datetime(2023,1,1)
enddate = datetime(2024,3,31)
#enddate = datetime.today() - timedelta(days = 1)


@st.cache_data #
def capture_data(symbols, startdate, enddate):
    data = pdr.get_data_yahoo(symbols, start=startdate, end=enddate)
    close = data.Close

    TRADING_DAYS = 252
    returns = np.log(data['Close']/data['Close'].shift(1))
    returns.fillna(0, inplace=True)
    volatility = returns.rolling(window=TRADING_DAYS).std()*np.sqrt(TRADING_DAYS)

# Capturing the latest information on Price, Returns and Volatility and generating a new dataframe

    last_row_df1 = close.iloc[[-1]].T
    last_row_df2 = returns.iloc[[-1]].T
    last_row_df3 = volatility.iloc[[-1]].T

    new_df = pd.concat([last_row_df1, last_row_df2, last_row_df3], axis=1, ignore_index=True)

# Renaming the columns
    new_df.columns = ['Closing Price', 'Return', 'Volatility']
# Adding a new column "ESG score" with random numbers between 0 and 100
    new_df['ESG score'] = np.random.randint(0, 101, size=len(new_df))

    new_df = new_df.dropna()
    data = new_df.sample(n=50)
    return data

data = capture_data(symbols, startdate, enddate)

st.write("This is the ESG score of your portfolio, calculated on a weighted average basis. Use the slider to set it to the mininum ESG score of your choice. ")

esg_score = st.slider("ESG Score", min_value=0, max_value=100, value=80, step=1, format=None, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")

st.write("This is the Volatility of your portfolio, calculated on a weighted average basis. Use the slider to set it to the maximum volatility score of your choice. ")

volatility_score = st.slider("Valotaility", min_value=0.0, max_value=1.0, value=0.5, step=0.1, format=None, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")

st.write("This is your investment capital. Use the slider to determine the overal cost of your investment portfolio.")

investment_capital = st.slider("Investment Capital", min_value=2000, max_value=1000000, value=100000, step=100, format=None, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")


# Create a Pyomo model
model = pyo.ConcreteModel()

# Defining 50 Decision Variables - These would be the amount of each stock to be purchased
model.x = pyo.Var(range(50), within=pyo.NonNegativeReals)

# Defining Inputs - These will be read from the dataframe imported above
model.P = {i: data['Closing Price'].iloc[i] for i in range(50)}  # Price of each stock
model.R = {i: data['Return'].iloc[i] for i in range(50)}  # Return on each stock
model.ESG = {i: data['ESG score'].iloc[i] for i in range(50)}  # ESG score of each stock
model.Volatility = {i: data['Volatility'].iloc[i] for i in range(50)}  # Volatility of each stock

# Here, I am initializing values for investment capital, the ESG score of the portfolio and the volatility
# The main idea is for the user to be able to play with these constraints and calculate optimal allocations accordingly
model.IC = pyo.Param(initialize=investment_capital)  # Initial investment capital (assuming 100,000)
model.Threshold = pyo.Param(initialize=esg_score)  # Minimum weighted average ESG score (assuming 80)
model.VolatilityConstraint = pyo.Param(initialize=volatility_score)  # Maximum weighted average volatility (assuming 0.5)

# Define the objective function to maximise return on portfolio
model.obj = pyo.Objective(expr=sum(model.x[i] * model.P[i] * model.R[i] for i in range(50)), sense=pyo.maximize)

# Defining constraints:

# Budget constraint
model.budget_constraint = pyo.Constraint(expr=sum(model.x[i] * model.P[i] for i in range(50)) == model.IC)
# Constraint on the Weighted average ESG score of the portfolio
# Calculated based on the overall value of each stock in the portfolio
model.esg_constraint = pyo.Constraint(expr=sum(model.ESG[i] * model.x[i]* model.P[i] for i in range(50)) >= model.Threshold * sum(model.x[i]* model.P[i] for i in range(50)))
# Constraint on the Weighted average volatility of the portfolio
# Calculated based on the number of each stock in the portfolio
model.volatility_constraint = pyo.Constraint(expr=sum(model.Volatility[i] * model.x[i] for i in range(50)) <= model.VolatilityConstraint * sum(model.x[i]for i in range(50)))

# No single stock can exhaust more than 50% of the investment capital
def budget_allocation(model,i):
    return (model.x[i] * model.P[i] <= 0.5 * model.IC)

model.budgetallocation =pyo.Constraint(model.P,rule = budget_allocation)

model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

model.pprint()
opt = pyo.SolverFactory('glpk')

# run the solver opt on the model to obtain optimization results
results = opt.solve(model)

# print optimization summary
model.display()

# print shadow prices
model.dual.display()

def custom_floor(x):
    if 0 < x < 1:
        return 0
    else:
        return math.floor(x)

# Print names of selected stocks
st.write("Return on Portfolio: ", model.obj())  
st.write("Selected Stocks and Investment Amounts:")
for i in range(50):
    if model.x[i].value != 0:
        st.write(data.index[i], custom_floor(model.x[i].value))
