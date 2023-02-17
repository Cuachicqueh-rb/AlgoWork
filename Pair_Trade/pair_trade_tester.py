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
pairs_list
pair_num = 1
chosen_pair = X_test[list(pairs_list[pair_num-1][0:2])]
# chosen_pair = backup[list(pairs_list[pair_num-1][0:2])] # for whole series

ptb.prepare_data(chosen_pair)
ptb.plot_prices(log=True, normalised=True, plot=True)
halflife, thres = 90, 1.5
ptb.build(halflife=halflife, halflife_func=False, thres=thres, plot=True)

##### Check for stationarity
ptb.get_adf_spread() ## must be significant to at least 10%

##### Run Backtest
backtest, sharpe, cagr = ptb.backtest(thres=thres)
print(f'Cumret: {round(backtest.cstrategy.iloc[-1],2)}, Sharpe ratio: {round(sharpe,2)}, CAGR: {cagr}%, Max Drawdown: {round(backtest.drawdown.max()*100,2)}%.')

##### Performance 
backtest[['cstrategy','cTicker1','cTicker2']].plot()
ptb.portfolio_eval()

################################################################
########## Optimise Parameters
ptb.prepare_data(chosen_pair)
def optimise_params(pair_num, stoploss=20, tc=0.001):
    half_life = np.arange(50, 150, 10)
    threshold = np.arange(1, 2.5, 0.5)
    # delta = [1e-3, 1e-7] # 
    delta = [1e-7]
    opt_list = []
    for halflife, thres, D in [(x,y,z) for x in half_life for y in threshold for z in delta]:
        ptb.prepare_data(X_test[list(pairs_list[pair_num-1][0:2])])
        ptb.build(halflife=halflife, thres=thres, opt=True, delta=D)
        backtest, sharpe, cagr = ptb.backtest(thres=thres, stop_loss=stoploss, tc=tc)
        print(f'Col1-2: {backtest.columns[0]}-{backtest.columns[1]}, Cumret: {round(backtest.cstrategy[-1],2)},  Sharpe ratio: {round(sharpe,2)}, CAGR: {cagr}%, Max Drawdown: {round(backtest.drawdown.max()*100,2)}%\n')
        opt_list.append((halflife, thres, sharpe, D, cagr, backtest['cstrategy'][-1]))
    opt = pd.DataFrame(opt_list, columns=['Halflife', 'Threshold', 'Sharpe', 'Delta', 'CAGR', 'Cumret'])
    return opt

pairs_list
opt_list = optimise_params(pair_num=pair_num, stoploss=20)
opt_list.sort_values(by='Cumret', ascending=False).head(10)
[opt_list[opt_list['Threshold'] == i].sort_values(by='Cumret').tail(1)[['Halflife', 'Threshold', 'Sharpe', 'Cumret', 'Delta']] for i in opt_list['Threshold'].unique()]

################################################################
########## Looping through all pairs

def loop_through_all_pairs(pairs_list, X_test, halflife=90, thres=1.5, delta=1e-7):
    ret_list = []
    in_position = []
    for pair in range(len(pairs_list)):
        cols = pairs_list[pair][0:2]
        chosen_pair_loop = X_test[list(cols)]
        ptb.prepare_data(chosen_pair_loop)
        ptb.plot_prices(log=True, normalised=True, plot=False)
        halflife, thres = halflife, thres
        print(f"Chosen pair {list(cols)}.")
        ptb.build(delta=delta, halflife=halflife, halflife_func=False, thres=thres, plot=False)
        ptb.get_adf_spread()
        backtest, sharpe, cagr = ptb.backtest(long_pairs=False, alloc_strategy=1, thres=thres, stop_loss=20)
        if backtest['position'][-1] != 0:
            in_position.append(list(cols))      
        print(f'Cumret: {round(backtest.cstrategy.iloc[-1],2)}, Sharpe ratio: {round(sharpe,2)}, CAGR: {cagr}%, Max Drawdown: {round(backtest.drawdown.max()*100,2)}%.\n')
    return ret_list, in_position, recent, trade

halflife, thres = 90, 1.5
ret_list, in_position_pairs, in_position = loop_through_all_pairs(pairs_list, X_test, halflife, thres)


