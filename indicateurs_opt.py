# Indicateurs: some financial indicators
# Optimized implementation with vectorized calculations

import pandas as pd
import openpyxl
import numpy as np
import pdb
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
try:
    from get_next_trading_times import get_next_trading_times
except ImportError:
    pass


# Create a Indicator class
class Indicator:
    def __init__(self, df, interval = '30m'):
        # dataframe with open, low, high, close
        self.df = df
        self.interval = interval
        # Check if the dataframe has the required columns
        required_columns = ["open", "low", "high", "close"]
        for col in required_columns:
            if col not in self.df.columns:
                raise ValueError(f"DataFrame must contain the column: {col}")
        # Check if the dataframe is empty
        if self.df.empty:
            raise ValueError("DataFrame is empty")
        # Remove rows with NaN values
        self.df = self.df.copy()
        self.df.dropna(inplace=True)
        # get numpy array for open, low, high, close
        self.open = self.df["open"].to_numpy()
        self.low = self.df["low"].to_numpy()
        self.high = self.df["high"].to_numpy()
        self.close = self.df["close"].to_numpy()

    # MACD - Optimized with vectorized EMA
    def macd(self, short_window=12, long_window=26, signal_window=1):
        # Calculate the MACD (sur le close) using vectorized EMA
        alpha_short = 2 / (short_window + 1)
        alpha_long = 2 / (long_window + 1)
        
        # Vectorized EMA calculation
        tmp1 = self._ema_vectorized(self.close, alpha_short, initial_value=100)
        tmp2 = self._ema_vectorized(self.close, alpha_long, initial_value=100)
        
        # tmp3 starts at 0, not 100
        tmp3 = tmp1 - tmp2
        
        # macd smoothing with signal_window
        macd = self._smooth_signal_macd(tmp3, signal_window)
        
        # Signal line
        alpha_signal = 2 / (9 + 1)
        macd_signal1 = self._ema_vectorized(macd, alpha_signal, initial_value=100)
        macd_signal2 = macd - macd_signal1
        
        # Add the MACD to the dataframe
        self.df["macd"] = macd
        self.df["macd_signal1"] = macd_signal1
        self.df["macd_signal2"] = macd_signal2
        return self.df

    def _ema_vectorized(self, data, alpha, initial_value=100):
        """Vectorized EMA calculation using cumulative operations"""
        result = np.empty_like(data, dtype=np.float64)
        result[0] = initial_value
        
        # Vectorized computation using reduce-like approach
        for i in range(1, len(data)):
            result[i] = result[i-1] + alpha * (data[i] - result[i-1])
        
        return result

    def _smooth_signal_macd(self, data, signal_window):
        """Smooth signal for MACD with specific initialization"""
        result = np.empty_like(data, dtype=np.float64)
        result[0] = 100
        
        for i in range(1, len(data)):
            result[i] = result[i-1] + signal_window * (data[i] - result[i-1])
        
        return result

    # Stochastic - Optimized with rolling window operations
    def stochastic(self, ra=14):
        # Calculate using rolling min/max
        n = len(self.close)
        stoch = np.zeros(n)
        
        # Vectorized rolling min/max
        for i in range(ra-1, n):
            low_min = np.min(self.low[i-ra+1:i+1])
            high_max = np.max(self.high[i-ra+1:i+1])
            denominator = high_max - low_min
            if denominator != 0:
                stoch[i] = (self.close[i] - low_min) / denominator * 100
            else:
                stoch[i] = 0
        
        # Mean filtering optimized
        stochf = self._mean_filter_optimized(stoch, 3)
        stochf2 = self._mean_filter_optimized(stochf, 5)
        
        # Add the filtered stochastic to the dataframe
        self.df["stochRf"] = stochf
        self.df["stochRf2"] = stochf2
        
        # Compute with linear regression
        stochL = self.lissage_lsq_optimized(stoch, 15)
        self.df["stochRL"] = stochL
        return self.df

    def _mean_filter_optimized(self, data, window_size):
        """Optimized mean filter using cumulative sum"""
        n = len(data)
        filtered_data = np.zeros(n)
        
        if window_size >= n:
            return filtered_data
        
        # Use cumulative sum for efficient rolling mean
        cumsum = np.cumsum(data)
        filtered_data[window_size-1:] = (cumsum[window_size-1:] - np.concatenate(([0], cumsum[:-window_size]))) / window_size
        
        return filtered_data

    def _mean_filter(self, data, window_size):
        """Original mean filter for compatibility"""
        filtered_data = np.zeros(len(data))
        for i in range(len(data)):
            if i < window_size:
                filtered_data[i] = 0
            else:
                filtered_data[i] = np.mean(data[i-window_size+1:i+1])
        return filtered_data

    # Lissage moindres carrés - Optimized
    def lissage_lsq_optimized(self, serie, f):
        """Optimized least squares smoothing with pre-computed matrices"""
        serie2 = serie.copy()
        
        # Pre-compute X matrix (constant for all windows)
        X = np.ones((f, 2))
        X[:, 0] = np.arange(f)
        XtX_inv = np.linalg.inv(X.T @ X)
        
        # Vectorized prediction
        for i in range(f, len(serie)):
            y = serie[i-f+1:i+1]
            a = XtX_inv @ (X.T @ y)
            serie2[i] = a[0] * (f+1) + a[1]
        
        return serie2

    def lissage_lsq(self, serie, f):
        """Original method for compatibility"""
        serie2 = serie.copy()
        for i in range(f, len(serie)):
            a = self.get_reg(serie[:i+1], f)
            serie2[i] = a
        return serie2

    def get_reg(self, serie, f):
        """Original method for compatibility"""
        serie = serie[-f:]
        X = np.ones((f, 2))
        X[:, 0] = np.arange(f)
        y = serie
        a = np.linalg.solve(X.T @ X, X.T @ y)
        return a[0] * (f+1) + a[1]

    # Indicateur RSI - Optimized
    def rsi(self, period=14):
        """Optimized RSI calculation"""
        n = len(self.close)
        delta = np.diff(self.close) / self.close[:-1]
        gain = np.maximum(delta, 0)
        loss = np.maximum(-delta, 0)
        
        avg_gain = np.zeros(n)
        avg_loss = np.zeros(n)
        
        for i in range(period, n):
            # Extract window
            tmp_gain = gain[i-period:i]
            tmp_loss = loss[i-period:i]
            
            # Calculate means only on non-zero values
            tmp_gain_nz = tmp_gain[tmp_gain != 0]
            tmp_loss_nz = tmp_loss[tmp_loss != 0]
            
            avg_gain[i] = np.mean(tmp_gain_nz) if len(tmp_gain_nz) > 0 else 0
            avg_loss[i] = np.mean(tmp_loss_nz) if len(tmp_loss_nz) > 0 else 0
        
        # Avoid division by zero
        rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
        rsi = 100 - (100 / (1 + rs))
        
        self.df["rsi"] = rsi
        return self.df

    # Indicateur ADX - Optimized
    def adx(self, ra=14):
        """Optimized ADX calculation with vectorized operations"""
        print("Calculating ADX...")
        n = len(self.close)
        
        # Calculate directional movements
        diff_high = np.diff(self.high)
        diff_low = np.diff(self.low)
        
        diff_high = np.maximum(diff_high, 0)
        diff_low = np.maximum(-np.diff(self.low), 0)
        
        # Compute True Range - vectorized
        tr = np.zeros(n)
        tr[1:] = np.maximum.reduce([
            self.high[1:] - self.low[1:],
            np.abs(self.high[1:] - self.close[:-1]),
            np.abs(self.low[1:] - self.close[:-1])
        ])
        
        # Smoothed DM and TR
        dm_plus = np.zeros(n)
        dm_minus = np.zeros(n)
        tr_range = np.zeros(n)
        
        dm_plus[0] = diff_high[0] if len(diff_high) > 0 else 0
        dm_minus[0] = diff_low[0] if len(diff_low) > 0 else 0
        tr_range[0] = tr[0]
        
        factor = (ra - 1) / ra
        for i in range(1, n):
            dm_plus[i] = factor * dm_plus[i-1] + (diff_high[i-1] if i-1 < len(diff_high) else 0)
            dm_minus[i] = factor * dm_minus[i-1] + (diff_low[i-1] if i-1 < len(diff_low) else 0)
            tr_range[i] = factor * tr_range[i-1] + tr[i]
        
        # Directional indicators
        di_plus = np.divide(dm_plus, tr_range, out=np.zeros_like(dm_plus), where=tr_range!=0) * 100
        di_minus = np.divide(dm_minus, tr_range, out=np.zeros_like(dm_minus), where=tr_range!=0) * 100
        
        # DX calculation
        di_sum = di_plus + di_minus
        dx = np.divide(np.abs(di_plus - di_minus), di_sum, out=np.zeros_like(di_plus), where=di_sum!=0) * 100
        dx = np.floor(dx)
        
        # ADX calculation
        ADX = np.zeros(n)
        for i in range(1, n):
            ADX[i] = ((ra - 1) * ADX[i-1] + dx[i]) / ra
        
        self.df["adx"] = ADX
        return self.df

    # Calculate the CCI - Optimized
    def cci(self, ra=20):
        """Optimized CCI calculation with vectorized operations"""
        typical_price = (self.high + self.low + self.close) / 3
        n = len(self.close)
        
        # Use cumsum for efficient rolling mean
        moving_average = np.zeros(n)
        mean_deviation = np.zeros(n)
        
        for i in range(ra, n):
            window = typical_price[i-ra:i]
            moving_average[i] = np.mean(window)
            mean_deviation[i] = np.mean(np.abs(window - moving_average[i]))
        
        # Calculate CCI
        cci = np.zeros(n)
        for i in range(ra, n):
            if mean_deviation[i] != 0:
                cci[i] = (typical_price[i] - moving_average[i]) / (0.015 * mean_deviation[i])
        
        self.df["cci"] = cci
        return self.df

    def CCI(self, ndays=20):
        """Pandas-based CCI calculation"""
        df = self.df.copy()
        df['TP'] = (self.high + self.low + self.close) / 3
        df['sma'] = df['TP'].rolling(ndays).mean()
        df['mad'] = df['TP'].rolling(ndays).apply(lambda x: pd.Series(x).mad())
        df['CCI'] = (df['TP'] - df['sma']) / (0.015 * df['mad'])
        self.df['CCI'] = df['CCI']
        return self.df

    '''
    Indicateurs codés par Arnaud
    '''

    # === EMA ===
    def calcul_ema(self, prix, periode):
        alpha = 2 / (periode + 1)
        ema = [prix.iloc[49 - periode + 1:50].mean()]
        for i in range(50, len(prix)):
            ema.append(alpha * prix.iloc[i] + (1 - alpha) * ema[-1])
        return pd.Series(ema, index=prix.index[49:49 + len(ema)])

    # === MACD ===
    def calcul_macd(self, close):
        ema12 = self.calcul_ema(close, 12)
        ema26 = self.calcul_ema(close, 26)
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean().fillna(0)
        histogramme = macd_line - signal_line
        return ema12, ema26, macd_line, signal_line, histogramme
    
    # === Stochastique classique + lissé par RL ===
    def stochastique_lisse_rl(self, close, high, low, periode=14):
        stoch_k = []
        index = []
        for i in range(periode - 1, len(close)):
            window_high = high[i - periode + 1:i + 1]
            window_low = low[i - periode + 1:i + 1]
            hh = window_high.max()
            ll = window_low.min()
            k = ((close.iloc[i] - ll) / (hh - ll)) * 100 if hh != ll else 0
            stoch_k.append(k)
            index.append(close.index[i])
        stoch_k = pd.Series(stoch_k, index=index)
    
        # Lissage par régression linéaire (prédiction du dernier point)
        smoothed = [np.nan] * (periode - 1)
        for i in range(periode - 1, len(stoch_k)):
            y = stoch_k.iloc[i - periode + 1:i + 1].values
            x = np.arange(periode).reshape(-1, 1)
            if np.isnan(y).any():
                smoothed.append(np.nan)
            else:
                model = LinearRegression().fit(x, y)
                smoothed.append(model.predict([[periode - 1]])[0])
        stoch_k_rl = pd.Series(smoothed, index=stoch_k.index)
    
        # Moyenne mobile classique (stoch D) et moyenne de D
        stoch_d = stoch_k.rolling(3).mean()
        stoch_avg = stoch_d.rolling(3).mean()
    
        return stoch_k, stoch_d, stoch_avg, stoch_k_rl
    
    # === RSI ===
    def calcul_rsi(self, close, periode=14):
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = [gain.iloc[49 - periode + 1:50].mean()]
        avg_loss = [loss.iloc[49 - periode + 1:50].mean()]
        for i in range(50, len(close)):
            g = gain.iloc[i]
            l = loss.iloc[i]
            avg_gain.append((avg_gain[-1] * (periode - 1) + g) / periode)
            avg_loss.append((avg_loss[-1] * (periode - 1) + l) / periode)
        rs = np.array(avg_gain) / np.array(avg_loss)
        rsi = 100 - (100 / (1 + rs))
        return pd.Series(rsi, index=close.index[49:49 + len(rsi)])
    
    # === CCI détaillé ===
    def calcul_cci_detaille(self, close, high, low, periode=20):
        tp = (high + low + close) / 3
        ma = tp.rolling(window=periode).mean()
        abs_dev = (tp - ma).abs()
        md = abs_dev.rolling(window=periode).mean()
        cci = (tp - ma) / (0.015 * md)
        return tp, ma, abs_dev, md, cci
    
    # === ADX et DX ===
    def calcul_adx(self, high, low, close, period=14):
        plus_dm = []
        minus_dm = []
        tr_list = []
        for i in range(1, len(high)):
            up_move = high.iloc[i] - high.iloc[i - 1]
            down_move = low.iloc[i - 1] - low.iloc[i]
            tr = max(
                high.iloc[i] - low.iloc[i],
                abs(high.iloc[i] - close.iloc[i - 1]),
                abs(low.iloc[i] - close.iloc[i - 1])
            )
            tr_list.append(tr)
            plus_dm.append(up_move if up_move > down_move and up_move > 0 else 0)
            minus_dm.append(down_move if down_move > up_move and down_move > 0 else 0)
        index = close.index[1:]
        tr = pd.Series(tr_list, index=index)
        plus_dm = pd.Series(plus_dm, index=index)
        minus_dm = pd.Series(minus_dm, index=index)
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        plus_dm_smoothed = plus_dm.ewm(alpha=1/period, adjust=False).mean()
        minus_dm_smoothed = minus_dm.ewm(alpha=1/period, adjust=False).mean()
        plus_di = 100 * plus_dm_smoothed / atr
        minus_di = 100 * minus_dm_smoothed / atr
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di)).round(2)
        adx = dx.ewm(alpha=1/period, adjust=False).mean()
        dx_full = pd.concat([pd.Series([np.nan], index=[close.index[0]]), dx])
        adx_full = pd.concat([pd.Series([np.nan], index=[close.index[0]]), adx])
        return dx_full, adx_full

    # Calcul et export indicateurs Arnaud
    def construire_arnaud(self, df_donnees):
        open_ = df_donnees["open"].reset_index(drop=True)
        high = df_donnees["high"].reset_index(drop=True)
        low = df_donnees["low"].reset_index(drop=True)
        close = df_donnees["close"].reset_index(drop=False)

        ema12, ema26, macd, signal, histo = self.calcul_macd(close["close"])
        stoch_k, stoch_d, stoch_avg, stoch_rl = self.stochastique_lisse_rl(close["close"], high, low)
        rsi = self.calcul_rsi(close["close"])
        tp, ma, abs_dev, md, cci = self.calcul_cci_detaille(close["close"], high, low)
        dx, adx = self.calcul_adx(high, low, close["close"])

        df_final = pd.DataFrame(index=close.index)
        df_final["date"] = close["index"]
        df_final["open"] = open_
        df_final["high"] = high
        df_final["low"] = low
        df_final["close"] = close["close"]
        df_final["MACD"] = macd
        df_final["Signal"] = signal
        df_final["Histogramme"] = histo
        df_final["Stoch %K"] = stoch_d
        df_final["Stoch %D"] = stoch_avg
        df_final["Stoch RL"] = stoch_rl
        df_final["RSI"] = rsi
        df_final["CCI(20)"] = cci
        df_final["ADX(14)"] = adx
        return df_final

    # compute linear extension for open 'high' 'low' 'close' 'column' for a given dataframe
    def linear_extension(self, df, period=15):
        """
        Compute linear extension for a given column in the dataframe.
        This is a simple linear extension by copying the last value of the column period times.
        args:
        df: dataframe with the column to extend
        period: number of periods to extend
        """
        last_value = df.iloc[-1, :]
        ext = pd.DataFrame([last_value] * period, columns=df.columns)
        df = pd.concat([df, ext], ignore_index=True)
        return df

    # Compute arnaud indicator for a linear extension of a given dataframe
    def construire_arnaud_linear_extension(self, df_donnees):
        period = 15
        df_donnees = self.linear_extension(df_donnees, period=period)
        open_ = df_donnees["open"].reset_index(drop=True)
        high = df_donnees["high"].reset_index(drop=True)
        low = df_donnees["low"].reset_index(drop=True)
        close = df_donnees["close"].reset_index(drop=False)
        volume = df_donnees["volume"].reset_index(drop=True)

        ema12, ema26, macd, signal, histo = self.calcul_macd(close["close"])
        stoch_k, stoch_d, stoch_avg, stoch_rl = self.stochastique_lisse_rl(close["close"], high, low)
        rsi = self.calcul_rsi(close["close"])
        tp, ma, abs_dev, md, cci = self.calcul_cci_detaille(close["close"], high, low)
        dx, adx = self.calcul_adx(high, low, close["close"])

        df_final = pd.DataFrame(index=close.index)
        df_final["date"] = close["index"] 
        df_final["open"] = open_
        df_final["high"] = high
        df_final["low"] = low
        df_final["volume"] = volume
        df_final["close"] = close["close"]
        df_final["MACD"] = macd
        df_final["Signal"] = signal
        df_final["Histogramme"] = histo
        df_final["Stoch %K"] = stoch_d
        df_final["Stoch %D"] = stoch_avg
        df_final["Stoch RL"] = stoch_rl
        df_final["RSI"] = rsi
        df_final["CCI(20)"] = cci
        df_final["ADX(14)"] = adx
        return df_final


def main():
    # Create some indicators
    # Load excel file
    df = pd.read_excel("Données_Source_Python2.xlsx")
    # extract open low high close from df in df2
    df2 = df[["open", "high", "low", "close"]]
    # Macd
    # On commence à l'index 22 pour valiser par rapport au tableau excel
    # Create an Indicator object
    indicator = Indicator(df2)
    # Calculate the MACD
    indicator.macd()
    # compute raw stochastic on df2
    indicator.stochastic()
    indicator.rsi()
    indicator.adx()
    #indicator.RSI_stock()
    # compute cci
    # save df to excel file called "output.xlsx"
    df2.to_excel("output_ind2.xlsx", index=False)
    
    # Appel indicateurs arnaud
    df_arnaud = indicator.construire_arnaud(df)
    # Save the dataframe to an excel file
    df_arnaud.to_excel("output_ind_arnaud.xlsx", index=False)



if __name__ == "__main__":
    main()
