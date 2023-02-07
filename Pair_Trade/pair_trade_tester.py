import pandas as pd
import numpy as np
import sys
sys.path.append("..")
import Pair_Trade_Backtester
import useful.backtester as bt
from useful.plot_settings import *
import seaborn as sns
from sklearn.model_selection import train_test_split

################################################################ 
########## Fetch data on the pairs
# tickers = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'TSLA', 'MSTR']
tickers = ['GOOG', 'BTC-USD', 'ETH-USD', 'TSLA', 'MSTR', 'ADA-USD', 
           'SHOP', 'AMZN', 'AAPL']
start = '2010-01-01'
data = bt.get_data(tickers=tickers, start=start)
backup = data.drop(['ADA', 'SHOP'], axis=1).ffill().dropna().copy()
df = backup.copy()

### Set instance
ptb = Pair_Trade_Backtester.Pair_Trade_Backtester()

### Create train test split and extract cointegrated pairs
X_train, X_test, _, _ = train_test_split(df, df, test_size=0.33, shuffle=False)
critical_value = 0.2
pvalue_matrix, pairs = ptb.find_cointegrated_pairs(X_train, critical_value=critical_value)
for pair in pairs:
    print("Asset {} and Asset {} has a co-integration score of {}".format(pair[0],pair[1],round(pair[2],4)))

### Convert our matrix of stored results into a DataFrame
pvalue_matrix_df = pd.DataFrame(pvalue_matrix)

### Use Seaborn to plot a heatmap of our results matrix
sns.clustermap(pvalue_matrix_df, xticklabels=df.columns,yticklabels=df.columns, figsize=(12, 12))
plt.title('Stock P-value Matrix')
plt.tight_layout()
plt.show()
    
################################################################ 
########## Build strategy
pair_num = 0
pair = X_test[list(pairs[pair_num][0:2])]
ptb.prepare_data(pair)
ptb.plot_prices(log=True, normalised=True, plot=True)
halflife, thres = 115, 1.5
df = ptb.build(halflife=halflife, halflife_func=False, thres=thres, plot=True)

##### Check for stationarity
bt.get_adf_spread(df['spread']) ## must be significant to at least 10%

##### Run Backtest
df, sharpe, cagr = ptb.backtest(thres=thres)
print(f'Sharpe ratio: {round(sharpe,2)}, CAGR: {cagr}%.')

##### Performance 
df[['cstrategy','cTicker1','cTicker2']].plot()
ptb.portfolio_eval()

################################################################
########## Optimise Parameters
df = ptb.prepare_data(pair)
def optimise_params(stoploss=20, tc=0.001):
    # half_life = np.arange(10, 300, 5)
    # threshold = np.arange(0.5, 3.5, 0.5)
    half_life = np.arange(5, 160, 10)
    threshold = np.arange(1, 2.5, 0.5)
    delta = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    delta = [1e-3]
    ret_list = []
    for halflife, thres, delta_ in [(x,y,z) for x in half_life for y in threshold for z in delta]:
        ptb.build(halflife=halflife, thres=thres, opt=True, delta=delta_)
        backtest, sharpe, cagr = ptb.backtest(thres=thres, stop_loss=stoploss, tc=tc)
        ret_list.append((halflife, thres, sharpe, delta_, cagr, backtest['cstrategy'][-1]))
    opt = pd.DataFrame(ret_list, columns=['Halflife', 'Threshold', 'Sharpe', 'Delta', 'CAGR', 'Cumret'])
    return opt

ret_list = optimise_params()
ret_list.sort_values(by='Cumret', ascending=False).head(10)
[ret_list[ret_list['Threshold'] == i].sort_values(by='Cumret').tail(1)[['Halflife', 'Threshold', 'Sharpe', 'Cumret']] for i in ret_list['Threshold'].unique()]

    
