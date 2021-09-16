import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize as sco

st.title('Efficient Frontier')
st.set_option('deprecation.showPyplotGlobalUse', False)

user_tickers = st.text_input("Stock Symbols separate by space", "COM7.BK CPALL.BK")
user_tickers_count = len(user_tickers.split(" "))
st.text(user_tickers)

data = yf.download(tickers = user_tickers, period="1y", group_by = 'Ticker')

# data.head()

data = data.stack(level=0).rename_axis(['Date', 'Ticker']).reset_index(level=1)
df = pd.DataFrame(data)
# df.head()
df = df.loc[:,'Ticker':'Adj Close']
# df.head()

table = df.pivot(columns='Ticker')
table.columns = [col[1] for col in table.columns]
table.head()

plt.style.use('fivethirtyeight')
plt.figure(figsize=(14,7))
for c in table.columns.values:
  plt.plot(table.index, table[c], lw=3, alpha=0.8, label=c)
plt.legend(loc='upper left', fontsize=12)
plt.ylabel('price in USD')


returns = table.pct_change()
plt.figure(figsize=(14, 7))
for c in returns.columns.values:
    plt.plot(returns.index, returns[c], lw=3, alpha=0.8,label=c)
plt.legend(loc='upper right', fontsize=12)
plt.ylabel('daily returns')

def portfolio_annualised_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns*weights ) *252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return std, returns

def random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate):
    results = np.zeros((3,num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.random(user_tickers_count)
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_std_dev, portfolio_return = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
        results[0,i] = portfolio_std_dev
        results[1,i] = portfolio_return
        results[2,i] = (portfolio_return - risk_free_rate) / portfolio_std_dev
    return results, weights_record

def display_simulated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate):
    results, weights = random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate)
    
    max_sharpe_idx = np.argmax(results[2])
    sdp, rp = results[0,max_sharpe_idx], results[1,max_sharpe_idx]
    max_sharpe_allocation = pd.DataFrame(weights[max_sharpe_idx],index=table.columns,columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i*100,2)for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T
    
    min_vol_idx = np.argmin(results[0])
    sdp_min, rp_min = results[0,min_vol_idx], results[1,min_vol_idx]
    min_vol_allocation = pd.DataFrame(weights[min_vol_idx],index=table.columns,columns=['allocation'])
    min_vol_allocation.allocation = [round(i*100,2)for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T
    
    print("-"*80)
    print("Maximum Sharpe Ratio Portfolio Allocation\n")
    print("Annualised Return:", round(rp,2))
    print("Annualised Volatility:", round(sdp,2))
    print("\n")
    print(max_sharpe_allocation)
    print("-"*80)
    print("Minimum Volatility Portfolio Allocation\n")
    print("Annualised Return:", round(rp_min,2))
    print("Annualised Volatility:", round(sdp_min,2))
    print("\n")
    print(min_vol_allocation)
    plt.figure(figsize=(10, 7))
    plt.scatter(results[0,:],results[1,:],c=results[2,:],cmap='BuPu', marker='o', s=10, alpha=0.3)
    plt.colorbar()
    plt.scatter(sdp,rp,marker='*',color='r',s=500, label='Maximum Sharpe ratio')
    plt.scatter(sdp_min,rp_min,marker='*',color='g',s=500, label='Minimum volatility')
    plt.title('Simulated Portfolio Optimization based on Efficient Frontier')
    plt.xlabel('annualised volatility')
    plt.ylabel('annualised returns')
    plt.legend(labelspacing=0.8)

    # plt.figure(figsize=(10, 7))
    # fig, ax = plt.subplots()
    # ax.scatter(results[0,:],results[1,:],c=results[2,:],cmap='BuPu', marker='o', s=10, alpha=0.3)
    # plt.colorbar()
    # ax.scatter(sdp,rp,marker='*',color='r',s=500, label='Maximum Sharpe ratio')
    # ax.scatter(sdp_min,rp_min,marker='*',color='g',s=500, label='Minimum volatility')
    # ax.title('Simulated Portfolio Optimization based on Efficient Frontier')
    # ax.xlabel('annualised volatility')
    # ax.ylabel('annualised returns')
    # ax.legend(labelspacing=0.8)
    st.pyplot()

mean_returns = returns.mean()
cov_matrix = returns.cov()
num_portfolios = 25000
risk_free_rate = 0.0178

# display_simulated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate)
# mean_returns.head()

display_simulated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate)