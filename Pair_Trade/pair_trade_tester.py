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

########## Fetch data on the pairs
tickers = ['BTC-USD', 'ETH-USD']
start = '2010-01-01'
data = bt.get_data(tickers=tickers, start=start, end=None)
backup = data.ffill().dropna().copy()
df = backup.copy()

########## Back up prices to pickle
# file = 'prices' + tickers[0].split('-')[0] + tickers[1].split('-')[0] + '.pkl'
# backup.to_pickle(f'../../data/raw/{file}')

ptb = ptb_.Pair_Trade_Backtester(price_data=backup)
df = ptb.prepare_data()

########## Plot Prices
ptb.plot_prices(log=True, normalised=True, plot=True)

########## Test for cointegration multiple ways
ptb.get_johansen_test(log=False)
coint(df.BTC, df.ETH)
coint(np.log(df.BTC), np.log(df.ETH))
# Johansen test - cointegration found
# Engle-Granger test p-value ~ 0.07

########## Build the model
halflife, thres = 120, 1
df = ptb.build(halflife=halflife, thres=thres, plot=True, halflife_func=False)

########## Augmented Dickey-Fuller for stationarity of spread
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
