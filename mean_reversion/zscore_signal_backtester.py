import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from useful.config import get_adf
# import pyfolio as pf

####
# Signals data set of zcores
# index_prices of indices 
# col, index are the particular signal / index that is being tested.

###################################################################
class Economic_Indicator_Signals(object):
    def __init__(self, signals, index_prices, col, index, contrarian=1, start=None, end=None):
        self.signals = signals
        self.prices = index_prices
        self.col = col
        self.index = index
        self.contrarian = contrarian
        self.start = start
        self.end = end
   
    def initialise_data(self):
        self.df = self.signals[[self.col]].join(self.prices[self.index])
        valid_row = self.df[self.col].first_valid_index()
        self.df = self.df.loc[valid_row:].astype(float)
        self.df.loc[:, self.index] = self.df[self.index].ffill()
        zscore = self.df[self.col]+100
        self.df['ret_signal'] = np.log(zscore/zscore.shift(1))
   
        valid_row = self.df['ret_signal'].first_valid_index()
        self.df = self.df.loc[valid_row:].loc[self.start:self.end]
        return self.df
   
    def get_signal_plot(self, remove_blackswans=False, thres=1):
        if remove_blackswans:
            df = self.reduce_lookahead_bias(remove_covid=True, remove_gfc=True, remove_outliers=True)
        else:
            df = self.df.copy()
        sig_ = df.loc[self.start:self.end]
        mean = sig_[self.col].mean()
        std = sig_[self.col].std()
        sig_[[self.col]].dropna().plot(legend=True)       
        plt.axhline(std*thres+mean)
        plt.axhline(-std*thres+mean)
        plt.axhline(mean)
           
    def get_adfuller_test(self):
        # Test of stationarity. It tests the regression of the residuals.
        return get_adf(self.df[self.col])
   
    def reduce_lookahead_bias(self, remove_covid=False, remove_gfc=False, remove_outliers=False):
        df = self.initialise_data()
        if remove_outliers:
            ret = self.df['ret_signal']
            NOSD = 3
            mu = ret.mean()
            std = ret.std()
            k = 1.4826
            median = ret.median()
            MAD = (ret - ret.mean()).abs().mean()
            sig_robust = k * MAD
            df = df[np.abs(ret-median) < NOSD*sig_robust]
           
        if remove_covid:
            from_ts = '2020-03-01'
            to_ts = '2020-05-01'
            df = df[(df.index < from_ts) | (df.index > to_ts)]
           
        if remove_gfc:
            from_ts = '2007-12-01' #'2007-03-01'
            to_ts = '2009-04-01'
            df = df[(df.index < from_ts) | (df.index > to_ts)]
        self.df = df
        return df
 
    def mean_reversion(self, thres=1, half_IR=False):
        df = self.df.copy()
        threshold = df[self.col].std()*thres
        mean = df[self.col].mean()
        # short when signal goes above threshold or swap with contrarian
        df['position'] = np.where(df[self.col] > threshold,
                            -1*self.contrarian, np.nan)
        # long when signal goes below threshold or swap with contrarian
        df['position'] = np.where(df[self.col] < -threshold,
                                1*self.contrarian, df['position'])
        # close out position when it crosses mean
        df['position'] = df['position'].ffill()
        df['position'] = np.where((df['position'] == -1) & (df[self.col] < mean), 0, df['position'])
        df['position'] = np.where((df['position'] == 1) & (df[self.col] > mean), 0, df['position'])
        
        # if half_IR:
        #     volatility = (df['strategy'] - df['ret_index']).std()
        #     mean_return = (df['strategy'] - df['ret_index']).mean()
        #     information_ratio = mean_return / volatility
        #     # half the IR ratio
        #     half_IR_ratio = information_ratio / 2
        #     # use the halfed IR as benchmark
        #     df['excess_ret'] = df['strategy'] - half_IR_ratio
        #     # df['cstrategy'] = excess_ret.cumsum().apply(np.exp)
       
        # # determine returns and trading logic
        df['strategy'] = (df['ret_signal'] * df['position'].shift(1)).fillna(0)
        ## cumulative returns
        df['cstrategy'] = df['strategy'].cumsum().apply(np.exp)
        # df['cIndex'] = df['ret_index'].cumsum().apply(np.exp)
        df['drawdown'] = df['cstrategy'].cummax() - df['cstrategy']
        self.df = df   
        return df
   
    def bollinger_bands(self, window=12, std=2, plot=False):
        df = self.df.copy()
        df['mid'] = df[self.col].rolling(window=window).mean()
        std_rolling = df[self.col].rolling(window=window).std()
        df['upbnd'] = df['mid'] + std_rolling * std
        df['lwbnd'] = df['mid'] - std_rolling * std
        if plot:
            plt.title('Bollinger Bands')
            plt.plot(df[self.col], label='Signal Line')
            plt.plot(df['upbnd'], label='Bollinger Up', c='g')
            plt.plot(df['lwbnd'], label='Bollinger Down', c='r')
            plt.legend()
            plt.show()
       
        # short when signal goes above threshold or swap with contrarian
        df['position'] = np.where(df[self.col] > df['upbnd'],
                            -1*self.contrarian, np.nan)
        # long when signal goes below threshold or swap with contrarian
        df['position'] = np.where(df[self.col] < -df['lwbnd'],
                                1*self.contrarian, df['position'])
       
        # close out position when it crosses middle sma
        df['position'] = df['position'].ffill().fillna(0)
        df['position'] = np.where((df['position'] == -1) & (df[self.col] < df['mid']), 0, df['position'])
        df['position'] = np.where((df['position'] == 1) & (df[self.col] > df['mid']), 0, df['position'])
 
        # # determine returns and trading logic
        df['strategy'] = (df['ret_signal'] * df['position'].shift(1)).fillna(0)
        df['cstrategy'] = df['strategy'].cumsum().apply(np.exp)
        df['drawdown'] = df['cstrategy'].cummax() - df['cstrategy']   
        
        self.df = df
       return df
                                      
    def portfolio_eval(self):
        df = self.df.copy()
        #####################################
        # Prepare Metrics
        metrics = [
            'Annual Return',
            'Cumulative Returns',
            'Cumulative Returns Index',
            'Annual Volatility',
            'Sharpe Ratio',
            'Sortino Ratio',
            # 'Maximum Drawdown',
        ]
        columns = ['Backtest']
        portfolio_eval = pd.DataFrame(index=metrics, columns=columns)
        portfolio_eval.loc['Cumulative Returns'] = (
            df['cstrategy'][-1]
        )
        ret_index = np.log(self.df[self.index]/self.df[self.index].shift(1))
        cIndex = ret_index.cumsum().apply(np.exp)
        portfolio_eval.loc['Cumulative Returns Index'] = (
            cIndex[-1]
        )
        # portfolio_eval.loc['Maximum Drawdown'] = (
        #     df['cstrategy'].cummax()
        # )
        portfolio_eval.loc['Annual Return'] = (
            df['strategy'].mean() * 252
        )
        portfolio_eval.loc['Annual Volatility'] = (
            df['strategy'].std() * np.sqrt(252)
        )
        portfolio_eval.loc['Sharpe Ratio'] = (
            portfolio_eval.loc['Annual Return'] / portfolio_eval.loc['Annual Volatility']
        )
        try:
            sortino_ratio = df[['strategy']].copy()
            sortino_ratio.loc[:,'Downside Returns'] = 0
            target = 0
            mask = sortino_ratio['strategy'] < target
            sortino_ratio.loc[mask, 'Downside Returns'] = (
                sortino_ratio['strategy']**2
                # to make non-negative
            )
            down_stdev = (np.sqrt(sortino_ratio['Downside Returns'].mean())
                        * np.sqrt(252)
            )
            expected_returns = (
                sortino_ratio['strategy'].mean() * 252
            )
            portfolio_eval.loc['Sortino Ratio'] = (
                expected_returns/down_stdev
            )
        except:
            portfolio_eval.loc['Sortino Ratio'] = 0
        return portfolio_eval
 
    def optimise_mean_reversion_threshold(self, sort_by: str):
        string_list = ['Annual Return', 'Cumulative Returns', 'Cumulative Returns Index','Annual Volatility', 'Sharpe Ratio', 'Threshold']
        assert sort_by in string_list, f"{sort_by} not found in list: {string_list}"
        threshold = np.arange(0.5, 3.5, 0.5)
        ret_list = []
        for thres in [x for x in threshold]:
            self.mean_reversion(thres)
            metrics = self.portfolio_eval().T
            ret_list.append((metrics.iloc[:,0][-1], metrics.iloc[:,1][-1], metrics.iloc[:,2][-1], metrics.iloc[:,3][-1], metrics.iloc[:,4][-1], thres))
        opt = pd.DataFrame(ret_list, columns=string_list)
        return opt
   
    def optimise_bollinger_bands(self, sort_by: str):
        string_list = ['Annual Return', 'Cumulative Returns', 'Cumulative Returns Index','Annual Volatility', 'Sharpe Ratio', 'Window', 'Std']
        assert sort_by in string_list, f"{sort_by} not found in list: {string_list}"
        window_length = np.arange(2, 24, 2)
        threshold = np.arange(0.5, 3.5, 0.5)
        ret_list = []
        for window, thres in [(x,y) for x in window_length for y in threshold]:
            self.bollinger_bands(window=window, std=thres)
            metrics = self.portfolio_eval().T
            ret_list.append((metrics.iloc[:,0][-1], metrics.iloc[:,1][-1], metrics.iloc[:,2][-1], metrics.iloc[:,3][-1], metrics.iloc[:,4][-1], window, thres))
        opt = pd.DataFrame(ret_list, columns=string_list)
        return opt.sort_values(by=sort_by, ascending=False)
    
###################################################################  
def tester(signals, indexes, signal, index_specific, thres=0.5, window=12,
           contrarion=1, bollinger_bands=False, remove_covid=False, remove_gfc=False,
           remove_outliers=False, remove_blackswans=False):
    temp = pd.DataFrame()
    temp.index = ['Annual Return', 'Cumulative Returns', 'Cumulative Returns Index',
                'Annual Volatility', 'Sharpe Ratio', 'Sortino Ratio']
       
    for ind in index_specific:
        eis = Economic_Indicator_Signals(signals, indexes, col=signal, index=ind,
                                        contrarian=contrarion)
        eis.initialise_data()
        if remove_covid:
            eis.reduce_lookahead_bias(remove_covid=True, remove_gfc=False, remove_outliers=False)
        if remove_gfc:
            eis.reduce_lookahead_bias(remove_covid=False, remove_gfc=True, remove_outliers=False)
        if remove_outliers:
            eis.reduce_lookahead_bias(remove_covid=False, remove_gfc=False, remove_outliers=True)
        if remove_blackswans:
            eis.reduce_lookahead_bias(remove_covid=True, remove_gfc=True, remove_outliers=True)
       
        if bollinger_bands:
            eis.bollinger_bands(window=window, std=thres)
        else:
            eis.mean_reversion(thres=thres)
           
        temp[f'{ind}'] = eis.portfolio_eval().iloc[:,0]
    return temp
 
###################################################################
###################################################################
############ Indivudal signal and index tester
start, end =  None, None
signal_list = ['US', 'Australia', 'China', 'Eurozone', 'World proxy', 'Combined Australia + China']
index_list = ['S&P 500', 'S&P/ASX 300', 'Eurostoxx 50', 'Shanghai Composite',
           'Global Equities - Large Cap (h)', 'Australian Sovereigns',
           'Global Sovereigns', 'US 3 month LIBOR']  
 
signal_to_test = 'US'
index_to_test = 'S&P 500'
contrarian = 1 # swap with -1
eis = Economic_Indicator_Signals(signals, indexes, col=signal_to_test, index=index_to_test, contrarian=contrarian, start=start, end=end)
eis.initialise_data()
eis.reduce_lookahead_bias(remove_covid=True, remove_gfc=True, remove_outliers=True)
eis.get_adfuller_test()
eis.get_signal_plot(thres=1, remove_blackswans=False)
df = eis.mean_reversion(thres=1)
eis.portfolio_eval()
df = eis.bollinger_bands(window=6, std=1, plot=True)
eis.portfolio_eval()
   
##### Optimise parameters
eis.optimise_mean_reversion_threshold(sort_by='Cumulative Returns')
eis.optimise_bollinger_bands(sort_by='Annual Return')
 
##### Other metrics
# returns = df['strategy']
# bmk_ret = np.log(df[index_to_test]/df[index_to_test].shift(1))
# pf.create_simple_tear_sheet(returns=returns, benchmark_rets=bmk_ret)
   
##### Even more metrics
# import empyrical as em
# em.calmar_ratio(returns)
# em.max_drawdown(returns)
# em.cagr(returns)
# em.conditional_value_at_risk(returns)
 
###################################################################   
##### Group Tester - Highlight all and run, then individually run the
# countries below

threshold = 2
window = 6
contrarian = 1
bollinger_bands = True
   
remove_blackswans = True
remove_covid = False
remove_gfc = False
remove_outliers = False
   
us_indexes = ['S&P 500', 'Global Equities - Large Cap (h)', 'Global Sovereigns']
US = tester(signals, indexes, 'US', us_indexes, thres=threshold, bollinger_bands=bollinger_bands, window=window, contrarion=contrarian, remove_blackswans=remove_blackswans, remove_covid=remove_covid, remove_gfc=remove_gfc, remove_outliers=remove_outliers)
aus_indexes = ['S&P/ASX 300', 'Australian Sovereigns']
Aus = tester(signals, indexes, 'Australia', aus_indexes, thres=threshold, bollinger_bands=bollinger_bands, window=window, contrarion=contrarian, remove_blackswans=remove_blackswans, remove_covid=remove_covid, remove_gfc=remove_gfc, remove_outliers=remove_outliers)
china_indexes = ['S&P/ASX 300', 'Shanghai Composite','Australian Sovereigns']
China = tester(signals, indexes, 'China', china_indexes, thres=threshold, bollinger_bands=bollinger_bands, window=window, contrarion=contrarian, remove_blackswans=remove_blackswans, remove_covid=remove_covid, remove_gfc=remove_gfc, remove_outliers=remove_outliers)
euro_indexes = ['Eurostoxx 50']
Eurozone = tester(signals, indexes, 'Eurozone', euro_indexes, thres=threshold, bollinger_bands=bollinger_bands, window=window, contrarion=contrarian, remove_blackswans=remove_blackswans, remove_covid=remove_covid, remove_gfc=remove_gfc, remove_outliers=remove_outliers)
world_proxy_indexes = ['S&P 500', 'Global Equities - Large Cap (h)', 'Global Sovereigns']
World_proxy = tester(signals, indexes, 'World proxy', world_proxy_indexes, thres=threshold, bollinger_bands=bollinger_bands, window=window, contrarion=contrarian, remove_blackswans=remove_blackswans, remove_covid=remove_covid, remove_gfc=remove_gfc, remove_outliers=remove_outliers)
combined_indexes = ['S&P/ASX 300', 'Australian Sovereigns']
Combined_Aus_China = tester(signals, indexes, 'Combined Australia + China', combined_indexes, thres=threshold, bollinger_bands=bollinger_bands, window=window, contrarion=contrarian, remove_blackswans=remove_blackswans, remove_covid=remove_covid, remove_gfc=remove_gfc, remove_outliers=remove_outliers)
   
US
Aus
China
Eurozone
World_proxy
Combined_Aus_China
###################################################################
