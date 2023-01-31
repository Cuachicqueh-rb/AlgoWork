import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from pykalman import KalmanFilter
import matplotlib.pyplot as plt
# import empyrical as em
# import pyfolio as pf


class Pair_Trade_Backtester(object):
    def __init__(self, price_data):
        self.price_data = price_data
        self.df = None
        
    def prepare_data(self):
        df = self.price_data.ffill().dropna()
        for t in df.columns:
            df['{}ret'.format(t)] = np.log(df[t]/df[t].shift(1))
        df.dropna(inplace=True)
        return df 
    
    def plot_prices(self, log=False, normalised=False, plot=True):
        df = self.price_data.copy()
        if plot != True:
            return print()
        else:
            if log:
                if normalised:
                    return (np.log(df/df.iloc[0]) + 100).plot()
                else:
                    return np.log(df).plot()
            else:
                if normalised:
                    return (df/df.iloc[0] * 100).plot()
                else:
                    return df.plot()
    
    def get_johansen_test(self, log=False):
        #(Alexander, 2002), “Since it is normally the case that 
        # log prices will be cointegrated when the actual prices 
        # are cointegrated, it is standard, but not necessary, 
        # to perform the cointegration analysis on log prices.”
        df = self.price_data.copy()
        df.dropna(inplace=True)
        if log:
            df = np.log(df)
        jres = coint_johansen(df, det_order=0, k_ar_diff=1)
        result = sum(jres.lr2 > jres.cvm[:,-2]) == 2
        return print('Johansen test of cointegrated series w/ significant (<0.05) max eigen statistic:', result)
   
    def get_adf_spread(self):
        # Augmented Dickey-Fuller test
        # Extract the time series data from the DataFrame
        ts = self.df['spread'].dropna().values
        adf_results = adfuller(ts)
        # Print the test statistic and p-value
        print("Test statistic:", adf_results[0])
        print("p-value:", adf_results[1])
    
        # Check if the test statistic is less than the critical values
        # at the 1%, 5%, and 10% significance levels
        if adf_results[0] < adf_results[4]['1%']:
            print("The time series is stationary at the 1% significance level.")
        if adf_results[0] < adf_results[4]['5%']:
            print("The time series is stationary at the 5% significance level.")
        if adf_results[0] < adf_results[4]['10%']:
            print("The time series is stationary at the 10% significance level.")
         
    def KalmanFilterAverage(self, x):
        # Construct a Kalman filter
        kf = KalmanFilter(transition_matrices = [1],
            observation_matrices = [1],
            initial_state_mean = 0,
            initial_state_covariance = 1,
            observation_covariance=1,
            transition_covariance=.01
            )
        # Use the observed values of the price to get a rolling mean
        state_means, _ = kf.filter(x.values)
        state_means = pd.Series(state_means.flatten(), index=x.index)
        return state_means

    def KalmanFilterRegression(self, x, y, delta=1e-3):
        delta = delta
        trans_cov = delta / (1 - delta) * np.eye(2) # How much random walk wiggles
        obs_mat = np.expand_dims(np.vstack([[x], 
                                            [np.ones(len(x))]]).T, axis=1)
        kf = KalmanFilter(n_dim_obs=1, 
                        n_dim_state=2, 
                        # y is 1-dimensional, (alpha, beta) is 2-dimensional
                        initial_state_mean=[0,0],
                        initial_state_covariance=np.ones((2, 2)),
                        transition_matrices=np.eye(2),
                        observation_matrices=obs_mat,
                        observation_covariance=2,
                        transition_covariance=trans_cov)
        # Use the observations y to get running estimates and errors for the state parameters
        state_means, state_covs = kf.filter(y.values)
        return state_means

    def half_life(self, spread):
        spread_lag = spread.shift(1)
        spread_lag.iloc[0] = spread_lag.iloc[1]
        spread_ret = spread - spread_lag
        spread_ret.iloc[0] = spread_ret.iloc[1]
        spread_lag2 = sm.add_constant(spread_lag)
        model = sm.OLS(spread_ret,spread_lag2)
        res = model.fit()
        halflife = int(round(-np.log(2) / res.params[1],0))
        if halflife <= 0:
            halflife = 1
        return halflife

    def build(self, halflife=90, plot=False, thres=2, delta=1e-3, opt=False, export=False, halflife_func=False):
        df = self.prepare_data()
        X = df.columns[1]
        Y = df.columns[0]
        x = df[X]
        y = df[Y]
        
        if opt:
            print(f'Testing \n halflife: {halflife}, thres: {thres}, delta: {delta}')
        else:
            print(f'In our reg Y is {Y} and X is {X}.')
        state_means = self.KalmanFilterRegression(self.KalmanFilterAverage(x),
                                            self.KalmanFilterAverage(y), delta=delta)

        # state_means = self.KalmanFilterRegression(x,y, delta=delta)

        df['hr'] = -state_means[:,0]
        df['spread'] = y + (df['hr'] * x)

        if halflife_func:
            halflife = self.half_life(df['spread'])
        else:
            halflife = halflife
        # calculate rolling zscore with window = halflife period
        meanSpread = df['spread'].rolling(window=halflife).mean()
        stdSpread = df['spread'].rolling(window=halflife).std()
        df['zscore'] = (df['spread'] - meanSpread) / stdSpread   
        if plot:
            plt.figure()
            plt.plot(df['zscore'].iloc[:])
            # plt.axhline(df['zscore'].mean(), color='b')
            plt.axhline(thres, color='r')
            plt.axhline(-thres, color='g')
            if export:
                export_figure(f'{df.columns[0]}{df.columns[1]}zscore_spread')
        self.df = df
        return df

    def backtest(self, thres=2, tc=0.001, stop_loss=20):
        df = self.df
        X = df.columns[1]
        Y = df.columns[0]
        x = df[X] ## ETH
        y = df[Y] ## BTC
        ###### determine signals
        ### if zscore > thres sell N of BTC and buy beta*N of ETH
        df['position'] = np.where(df['zscore'] > thres, -1, np.nan)
        df['position'] = np.where(df['zscore'] > thres*stop_loss, 0, df['position'])
        
        ### if zscore < -thres buy N of BTC and sell beta*N of ETH
        df['position'] = np.where(df['zscore'] < -thres, 1, df['position'])
        df['position'] = np.where(df['zscore'] < -thres*stop_loss, 0, df['position'])

        ### convert to cash when zscore goes to zero and forward fill
        df['position'] = np.where(df['zscore']*df['zscore'].shift(1) < 0, 
                                    0, df['position'])
        df['position'] = df['position'].ffill().fillna(0)
        
        ##### trading logic
        spread_pct_ch = (df['spread'] - df['spread'].shift(1)) / (y + (x * abs(df['hr'])))
        df['strategy'] = spread_pct_ch * df['position'].shift(1)
        
        # # determine when trades take place
        trades = df['position'].diff().fillna(0) != 0
        df['strategy'][trades] -= tc
        
        # calculate cumlative returns
        df['cstrategy'] = df['strategy'].cumsum().apply(np.exp)
        df['cTicker1'] = df.iloc[:,2].cumsum().apply(np.exp)
        df['cTicker2'] = df.iloc[:,3].cumsum().apply(np.exp)
        df['drawdown'] = df['cstrategy'].cummax() - df['cstrategy']
        
        try:
            sharpe = ((df['strategy'].mean() / df['strategy'].std()) * np.sqrt(252))
        except ZeroDivisionError:
            sharpe = 0.0
        # ##############################################################
        start_val = 1
        end_val = df['cstrategy'].iat[-1]
        start_date = df.iloc[0].name
        end_date = df.iloc[-1].name
        days = (end_date - start_date).days
        CAGR = round(((float(end_val) / float(start_val)) ** (252.0/days)) - 1,4)*100
        return df, sharpe, CAGR

    def portfolio_eval(self):
        df = self.df
        #####################################
        # Prepare Metrics
        ticker_1 = 'Cumulative Returns ' + df.columns[0]
        ticker_2 = 'Cumulative Returns ' + df.columns[1]
        metrics = [
            'Annual Return',
            'Cumulative Returns',
            ticker_1,
            ticker_2,
            'Annual Volatility',
            'Sharpe Ratio',
            'Sortino Ratio'
        ]
        columns = ['Backtest']
        portfolio_eval = pd.DataFrame(index=metrics, columns=columns)
        portfolio_eval.loc['Cumulative Returns'] = (
            df['cstrategy'][-1]
        )
        portfolio_eval.loc[ticker_1] = (
            df['cTicker1'][-1]
        )
        portfolio_eval.loc[ticker_2] = (
            df['cTicker2'][-1]
        )
        portfolio_eval.loc['Annual Return'] = (
            df['strategy'].mean() * 252
        )
        portfolio_eval.loc['Annual Volatility'] = (
            df['strategy'].std() * np.sqrt(252)
        )
        portfolio_eval.loc['Sharpe Ratio'] = (
            portfolio_eval.loc['Annual Return'] / portfolio_eval.loc['Annual Volatility']
        )
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
        # sortino_ratio.plot()
        return portfolio_eval.reset_index().hvplot.table()
