import pandas as pd
import numpy as np
import sys
sys.path.append("..")
import Pair_Trade_Backtester as ptb_
import useful.backtester as bt
import Kalman as kl
from useful.plot_settings import *
from statsmodels.tsa.stattools import coint
# import empyrical as em
# import pyfolio as pf

##########
########## Fetch data on the pairs
tickers = ['BTC-USD', 'ETH-USD']
# tickers = ['SOL-USD', 'ETH-USD']
# tickers = ['BTC-USD', 'TSLA']
start = '2010-01-01'
# df = bt.fetch_prices(tickers=tickers, start_date=start).dropna()
data = bt.get_data(tickers=tickers, start=start)
# df = bt.get_data(tickers=tickers, start=start).ffill().dropna()
# data = pd.read_pickle('../../data/raw/pricesSOLETH.pkl')
# data = pd.read_pickle('../../data/raw/pricesBTCETH.pkl')
backup = data.ffill().dropna().copy()
df = backup.copy()
df

########## Back up to pickle
# file = 'prices' + tickers[0].split('-')[0] + tickers[1].split('-')[0] + '.pkl'
# backup.to_pickle(f'../../data/raw/{file}')

ptb = ptb_.Pair_Trade_Backtester(price_data=backup)
df = ptb.prepare_data()
########## Plot Prices
ptb.plot_prices(log=True, normalised=True, plot=True)

########## Test for cointegration
ptb.get_johansen_test(log=False)
coint(df.BTC, df.ETH)
coint(np.log(df.BTC), np.log(df.ETH))

########## Build the model
halflife, thres = 120, 1
df = ptb.build(halflife=halflife, thres=thres, plot=True, halflife_func=False)

########## Augmented Dickey-Fuller
ptb.get_adf_spread()

########## Backtest model
df, sharpe, cagr = ptb.backtest(thres=thres, tc=0.001)
print(f'Sharpe ratio: {round(sharpe,2)}, CAGR: {cagr}%.')

########## Evaluate Performance
ptb.portfolio_eval()
plt.figure()
np.log(df[['cstrategy','cTicker1','cTicker2']]).plot()
plt.figure()
df[['cstrategy','cTicker1','cTicker2']].plot()
### Pyfolio
# pf.create_simple_tear_sheet(returns=df['strategy'], benchmark_rets = df.iloc[:,2])

########## Current day
# aa = kl.current_df(df, start='2022-07-15')
# aa.iloc[:20]

aa = kl.current_df(df, start='2023-01-01', principal=100)
aa.head(40)
aa.zscore.plot()
np.log(aa[['cstrategy','cTicker1','cTicker2']]).plot()

aa = kl.current_df(df, start='2023-01-01', principal=100)
aa.head(40)
aa[['cstrategy','cTicker1','cTicker2']].plot()

aa = kl.current_df(df, start='2023-01-23', principal=1000)
aa.head(15)
aa.spread.plot()

################################################################
########## Optimise Parameters
df = backup.copy()
def optimise_params(df, stoploss=20, tc=0):
    # half_life = np.arange(10, 300, 5)
    # threshold = np.arange(0.5, 3.5, 0.5)
    half_life = np.arange(5, 205+25, 25)
    threshold = np.arange(0.5, 3, 0.5)
    delta = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    delta = [1e-3]
    ret_list = []
    for halflife, thres, delta_ in [(x,y,z) for x in half_life for y in threshold for z in delta]:
        build = kl.build(df, halflife=halflife, thres=thres, opt=True, delta=delta_)
        backtest, sharpe, cagr = kl.backtest(build, thres=thres, stop_loss=stoploss, tc=tc)
        ret_list.append((halflife, thres, sharpe, delta_, cagr, backtest['cstrategy'][-1]))
    opt = pd.DataFrame(ret_list, columns=['Halflife', 'Threshold', 'Sharpe', 'Delta', 'CAGR', 'Cumret'])
    return opt

x = optimise_params(df)
x.sort_values(by='Cumret', ascending=False).head(10)



# l = []
# for i in [0.5, 1, 1.5, 2, 2.5]:
#     l.append(x[x['Threshold'] == i].sort_values(by='Cumret', ascending=False).max())
# l

ret_list = x
# ret_list = optimise_params(df)
ret_list.sort_values(by='Cumret', ascending=False).head(10)
ret_list[ret_list['Threshold'] == 1.5].sort_values(by='Cumret', ascending=False).head(10)

### return the best parameter set
params = max(ret_list.values, key=lambda x: x[2])
params

###################### Metrics
# import empyrical as em
# ret = df['strategy']
# em.aggregate_returns(df['strategy'], convert_to='yearly').mean()
# em.annual_return(df['strategy'])
# em.cagr(ret)
# em.conditional_value_at_risk(ret)
# em.max_drawdown??
# em.simple_returns??
# em.alpha(ret, df['BTCret'], risk_free=0)

########## Intra-day data 
# start = '2023-01-01'
# end = None
# recent = yf.download(tickers=tickers, start=start, interval='90m')['Close']
# backup2 = recent.copy()
# recent = backup2.copy()

########## Optimisations
########## Optimised parameters 
# BTC vs TSLA - no stoploss
halflife, thres = 80, 0.5 # 1e-3 delta
halflife, thres = 205, 1 ## 60, 1.5 ## 90, 2 ## 60, 2.5 ## 105, 3

# BTC vs ETH - no stoploss
halflife, thres = 115, 0.5 # 0.00010 == 1e-4 delta (ultimately does not change much as all 5 delta's in top 6)
halflife, thres = 120, 1 ## 60, 1.5 ## 90, 2 ## 60, 2.5 ## 105, 3

# SOL
halflife, thres = 80, 0.5 # 30, 1.5 # 30, 1 # 205, 1 # 80, 2 # 30, 2.5
halflife, thres = 30, 1.5
halflife, thres = 30, 1
halflife, thres = 205, 1
halflife, thres = 80, 2
halflife, thres = 30, 2.5

# BTC vs TSLA - no stoploss
halflife, thres = 80, 0.5 # 1e-3 delta
halflife, thres = 205, 1 ## 60, 1.5 ## 90, 2 ## 60, 2.5 ## 105, 3

############ To-do
# Add in a dollar based backtester system
# calculate win ratio, skewness, kurtosis, CVAR, VAR, and others
# is it okay to apply exponentail to a arthmetic return

# Data just comes in every 10 minutes and re-calculates the 
# zscore and trades based on that. Limit 1 trade per day. 
# Could position size with 50% in first touch of threshold and then 
# other 50% on the second touch.


result = coint(np.log(df['BTC']), np.log(df['ETH']))
pvalue = result[1]
pvalue

np.log(df['BTC']).plot()
np.log(df['ETH']).plot()