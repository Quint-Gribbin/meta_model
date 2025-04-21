def main(rolling_train_length=2100,
        rolling_test_length=21,
        n_k=8,
        holdout_start="2022-01-01",
        batch_size=512,
        learning_rate=8e-4,
        optuna_runs=50,
        random_seed=42,
        gcs_bucket="test_model_plots",
        include_fundamentals=0,
        include_will_features=0,
        notes="no notes",
        epochs=3000,
        addition_technical=0,
        additional_factors=2,
        act_func="swish",
        table_suffix="V023_test",
        use_correlations=0,
        will_portfolios=0,
        calculate_shap=1,
        calculate_ww=1,
        clamp_gradients=1,
        custom_tie_breaks=0,
        will_predictions=0,
        include_coint_regimes=0,
        include_cluster_data=0,
        include_hmm_regimes=0,
        include_skew_kurt=0,
        include_time_features=1,
        new_portfolios="cluster_portfolio_returns",
        factor_lag=1,
        live_next_day=0,
        is_test=0,
        cluster_df=0,
        return_lag=1
):
    args = locals().copy()
    args.pop("cluster_df")
    print("Here are all the input arguments to the model:")
    print(args)

    # add popping logic here
    keys_to_pop = [x for x in list(args.keys()) if x[0] == "_"] + ['In', 'Out', 'get_ipython', 'exit', 'quit']
    for key in keys_to_pop:
        if key in args:
            args.pop(key)

    print("The current type of cluster_df",  type(cluster_df))
    if (live_next_day == 1) and (type(cluster_df) == int):
        print("\n\n\n!!!!!ERROR!!!!!!!!!: \n The incoming portfolio returns in the `cluster_data` variable is not a DataFrame. \nPlease check the input data and try again.\n\n\n")
        raise ValueError(f"Incoming portfolio returns in the `cluster_data` variable is not a DataFrame. Please check the input data and try again.")
        return(None)

    YIELDS_TABLE = "issachar-feature-library.core_raw.factor_yields"
    INDEX_RETURNS_TABLE = "josh-risk.IssacharReporting.Index_Returns"
    PORTFOLIO_FRESHNESS = 5
    FACTOR_FRESHNESS = 60

    # Import key libraries
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    import seaborn as sns
    import matplotlib.pyplot as plt
    import torch
    from sklearn.preprocessing import MinMaxScaler
    import copy
    from datetime import datetime
    import os
    import io
    import pandas_gbq
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    from sklearn.preprocessing import MinMaxScaler
    import time
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.amp import autocast, GradScaler
    import shap
    import weightwatcher as ww
    from scipy.stats import skew, kurtosis
    import random
    import subprocess
    import pytz
    from datetime import datetime, timedelta
    from tabulate import tabulate

    holdout_start = pd.to_datetime(holdout_start)

    # Google Cloud imports
    from google.cloud import bigquery, storage
    # from google.api_core.exceptions import NotFound

    # Basic logging because the HP Tune does not log print statements until the end?
    import logging

    import subprocess

    # Run the command and capture output
    result = subprocess.run(
        ['curl', '-H', 'Metadata-Flavor: Google', 
        'http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/email'],
        capture_output=True,
        text=True
    )

    # Print the result
    logging.error(f"Job has started on this service account: {result.stdout.strip()}")

    # Set random seeds for reproducibility
    def set_random_seeds(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    set_random_seeds(random_seed)

    # Initialize the BigQuery client
    client = bigquery.Client(project="issachar-feature-library")

    # Function to fetch data from the specified table
    def read_table_to_dataframe(client, table_id, query=None):
        if query is None:
            query = f"SELECT * FROM `{table_id}`"
        else:
            query = query
        query_job = client.query(query)  # Make an API request to execute the query
        df = query_job.to_dataframe()   # Convert the results to a Pandas DataFrame
        return df

    # Logic to determine which portfolios to use
    if type(cluster_df) == type(0):
        if will_portfolios == 1:
            table_id = "issachar-feature-library.qjg.2025-02-13 Will Top 50 Returns"
            df_port = read_table_to_dataframe(client, table_id).fillna(0)
        else:
                # Define the table ID for the saved table
            table_id = "issachar-feature-library.qjg." + new_portfolios
            df_port = read_table_to_dataframe(client, table_id).fillna(0)

        if 'repo' in new_portfolios:
            df_port = df_port.set_index('date')
            df_port = df_port / 100 # Standardize the returns
            df_port = df_port[[x for x in df_port.columns if "inverse" not in x]] # Drop inverse portfolios (x*-1)
        else:
            df_port = df_port.pivot_table(index='date', columns='cluster', values='total_return')
            # df_port = df_port.set_index('date')
            df_port = df_port / 100 # Standardize the returns
            df_port = df_port[[x for x in df_port.columns if "inverse" not in x]] # Drop inverse portfolios (x*-1)
            subset = ['growth', 'profitability', 'volume_price_action', 'st_vol', 'accumulation_distribution', 'skew', 'rsi_2', 'short_momentum', 'technical_momentum', 'technical_hma', 'technical_candle', 'forward_estimates', 'alpha_005', 'alpha_020', 'alpha_034', 'alpha_041', 'cagr_stack', 'cagr_stack_s3', 'fql_1', 'fql_2', 'fql_3', 'fql_4', 'fql_5', 'fql_6', 'fql_7', 'fql_8', 'fql_9']
            df_port = df_port[subset]
            df_port.index = df_port.index.tz_localize(None)
    else:
        cluster_df = cluster_df.fillna(0)
        cluster_dates = cluster_df['upload_date'].copy(deep=True)
        df_port = cluster_df.pivot_table(index='date', columns='cluster', values='total_return')
        # df_port = df_port.set_index('date')
        df_port = df_port / 100 # Standardize the returns
        df_port = df_port[[x for x in df_port.columns if "inverse" not in x]] # Drop inverse portfolios (x*-1)
        subset = ['growth', 'profitability', 'volume_price_action', 'st_vol', 'accumulation_distribution', 'skew', 'rsi_2', 'short_momentum', 'technical_momentum', 'technical_hma', 'technical_candle', 'forward_estimates', 'alpha_005', 'alpha_020', 'alpha_034', 'alpha_041', 'cagr_stack', 'cagr_stack_s3', 'fql_1', 'fql_2', 'fql_3', 'fql_4', 'fql_5', 'fql_6', 'fql_7', 'fql_8', 'fql_9']
        df_port = df_port[subset]
        df_port.index = df_port.index.tz_localize(None)

    # Identifies if the table has been updated in the last 20 minutes
    def check_table_freshness(table_name, freshness_minutes=20):
        # Get the table reference
        table_ref = client.get_table(table_name)
        last_modified = table_ref.modified
        
        # Convert to datetime if it's not already
        if not isinstance(last_modified, datetime):
            last_modified = datetime.fromisoformat(last_modified.isoformat())
        
        # Check if the table was modified within the freshness window
        current_time = datetime.now(pytz.UTC)
        freshness_threshold = current_time - timedelta(minutes=freshness_minutes)
        
        # Optional: Log the comparison values for debugging
        print(f"Last modified {table_name}: {last_modified}")
        print(f"Freshness threshold: {freshness_threshold}")
        print(f"Staleness: {((current_time - last_modified).seconds / 60):.2f} minutes")
        
        if last_modified < freshness_threshold:
            raise ValueError(f"Table {table_name} was last updated at {last_modified}, which is more than {freshness_minutes} minutes ago.")
        
        # If no error is raised, return True to indicate the table is fresh
        return True
    
    # Read in the factor returns
    factor_query = f'''
                    SELECT 
                        mfr.date,
                        mfr.* EXCEPT (date), 
                        mfa.* EXCEPT (date)  
                    FROM 
                        `{INDEX_RETURNS_TABLE}` mfr
                    LEFT OUTER JOIN 
                        (SELECT 
                            PARSE_DATE('%m/%d/%Y', date) AS date,
                            * EXCEPT (date, `20yr_sp_fairvalue`)
                        FROM 
                            `{YIELDS_TABLE}`) mfa
                    ON 
                        mfr.date = mfa.date
                    ORDER BY 
                        mfr.date DESC
            '''
    
    if live_next_day == 1:
        if is_test == 0:
            check_table_freshness(INDEX_RETURNS_TABLE, freshness_minutes=FACTOR_FRESHNESS)
            check_table_freshness(YIELDS_TABLE, freshness_minutes=FACTOR_FRESHNESS)
            # TODO: Add date check here
            if len(cluster_dates.apply(lambda x: x.tz_convert('US/Eastern').date()).unique()) > 1:
                raise ValueError(f"Multiple dates detected in the portfolio data. Please ensure only one date is present.")
            if ((pd.Timestamp.today().tz_localize('UTC') - cluster_dates.max()).total_seconds() / 60) > PORTFOLIO_FRESHNESS:
                raise ValueError(f"Portfolio data is stale. Please ensure the data is updated.")

    df_factors = read_table_to_dataframe(bigquery.Client(project="issachar-feature-library"), "issachar-feature-library.qjg.macro_factor_returns", factor_query).fillna(0)
    
    # Subset to core set of 19 factors or no factors if configured to do so
    if additional_factors == 1:
        subset = ['date',
                'short_momentum',
                'thrm_momo_long',
                'long_value',
                'us_small_caps',
                'inflation',
                'xle',
                'tlt',
                'vxn_index',
                'high_beta_cyclicals',
                'qqq',
                'high_vol',
                'two_vs_5_spread',
                'crude_levered_equities',
                'low_vol',
                'two_yr',
                'secular_growth',
                'reflationary_cyclicals',
                'inflation_levered',
                'high_short_interest'
                    ]
        df_factors = df_factors[subset]
    elif additional_factors == 0:
        subset = ['date']
        df_factors = df_factors[subset]
    df_factors = df_factors.set_index('date')
    df_factors.index = df_factors.index.tz_localize(None)
    df_factors = df_factors.sort_values("date")

    # Start only with the portfolios
    df = df_port
    df = df.sort_values(by="date")
    df = df[[x for x in df_port.columns]] # Just in case something crept in
    df = df.reset_index()
    portfolio_cols = df_port.columns
    id_vars = ['date'] # Set up for melt

    # GPU check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("CUDA available:", torch.cuda.is_available())
    print("Device count:", torch.cuda.device_count())
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(0))

    # TODO: Update to Dask to use all 16 cores
    feature_gen_start = time.time()
    #####################################################################
    #  3. FEATURE ENGINEERING
    #####################################################################

    # No leakage
    def reconstruct_lag1_price(df_long, initial_price=100):
        """
        Reconstruct lag1 price series for each portfolio group
        """
        return df_long.groupby('portfolio_id')['lag1_returns'].transform(
            lambda x: initial_price * (1 + x).cumprod()
        )

    # No leakage
    def calculate_bollinger_bands(df_long, window=20, k=2):
        """
        Calculate Bollinger Bands using lag1 price for each portfolio group
        """
        grouped = df_long.groupby('portfolio_id')['lag1_price']
        rolling_mean = grouped.transform(lambda x: x.rolling(window, min_periods=window).mean())
        rolling_std = grouped.transform(lambda x: x.rolling(window, min_periods=window).std())

        upper_band = rolling_mean + (k * rolling_std)
        lower_band = rolling_mean - (k * rolling_std)

        return pd.DataFrame({
            'lag1_bollinger_upper': upper_band,
            'lag1_bollinger_lower': lower_band
        })

    # No leakage
    def calculate_channel_breakouts(df_long, window=20):
        """
        Calculate channel breakouts using lag1 price for each portfolio group
        """
        grouped = df_long.groupby('portfolio_id')['lag1_price']
        highest_high = grouped.transform(lambda x: x.rolling(window, min_periods=window).max())
        lowest_low = grouped.transform(lambda x: x.rolling(window, min_periods=window).min())

        return pd.DataFrame({
            'lag1_channel_high': highest_high,
            'lag1_channel_low': lowest_low
        })

    # No leakage
    def calculate_momentum(df_long, period=1):
        """
        Calculate momentum using lag1 returns for each portfolio group
        """
        momentum = df_long.groupby('portfolio_id')['lag1_returns'].transform(
            lambda x: x.pct_change(periods=period)
        )
        return pd.DataFrame({'lag1_momentum': momentum})

    # No leakage
    def calculate_moving_average(df_long, window=20):
        """
        Calculate moving average using lag1 price for each portfolio group
        """
        ma = df_long.groupby('portfolio_id')['lag1_price'].transform(
            lambda x: x.rolling(window, min_periods=window).mean()
        )
        return pd.DataFrame({'lag1_moving_avg': ma})

    # No leakage
    # TODO: Update to use adjust=True
    def calculate_macd(df_long, short_window=12, long_window=26, signal_window=9):
        """
        Calculate MACD using lag1 price for each portfolio group
        """
        grouped = df_long.groupby('portfolio_id')['lag1_price']

        short_ema = grouped.transform(
            lambda x: x.ewm(span=short_window, adjust=False).mean()
        )
        long_ema = grouped.transform(
            lambda x: x.ewm(span=long_window, adjust=False).mean()
        )

        macd = short_ema - long_ema
        signal_line = grouped.transform(
            lambda x: macd.ewm(span=signal_window, adjust=False).mean()
        )

        return pd.DataFrame({
            'lag1_macd': macd,
            'lag1_macd_signal': signal_line
        })

    # No leakage
    def calculate_rsi(df_long, window=14):
        """
        Calculate RSI using lag1 price for each portfolio group
        """
        grouped = df_long.groupby('portfolio_id')['lag1_price']

        delta = grouped.transform(lambda x: x.diff())
        gain = grouped.transform(
            lambda x: (delta.where(delta > 0, 0)).rolling(window, min_periods=window).mean()
        )
        loss = grouped.transform(
            lambda x: (-delta.where(delta < 0, 0)).rolling(window, min_periods=window).mean()
        )

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return pd.DataFrame({'lag1_rsi': rsi})

    # No leakage
    def calculate_support_resistance(df_long, window=20):
        """
        Calculate support and resistance using lag1 price for each portfolio group
        """
        grouped = df_long.groupby('portfolio_id')['lag1_price']
        support = grouped.transform(lambda x: x.rolling(window, min_periods=window).min())
        resistance = grouped.transform(lambda x: x.rolling(window, min_periods=window).max())

        return pd.DataFrame({
            'lag1_support': support,
            'lag1_resistance': resistance
        })

    # No leakage
    def calculate_volatility(df_long, window=20):
        """
        Calculate volatility using lag1 returns for each portfolio group
        """
        vol = df_long.groupby('portfolio_id')['lag1_returns'].transform(
            lambda x: x.rolling(window, min_periods=window).std()
        )
        return pd.DataFrame({'lag1_volatility': vol})


    # No leakage
    def calculate_sharpe_ratio(df_long, risk_free_rate=0.0, window=20):
        """
        Calculate Sharpe ratio using lag1 returns for each portfolio group
        """
        grouped = df_long.groupby('portfolio_id')['lag1_returns']
        excess_return = df_long['lag1_returns'] - risk_free_rate

        rolling_mean = grouped.transform(
            lambda x: excess_return.rolling(window, min_periods=window).mean()
        )
        rolling_std = grouped.transform(
            lambda x: excess_return.rolling(window, min_periods=window).std()
        )

        sharpe = rolling_mean / rolling_std
        return pd.DataFrame({'lag1_sharpe_ratio': sharpe})

    # No leakage
    def calculate_z_score(df_long, window=20):
        """
        Calculate z-score using lag1 returns for each portfolio group
        """
        grouped = df_long.groupby('portfolio_id')['lag1_returns']
        rolling_mean = grouped.transform(lambda x: x.rolling(window, min_periods=window).mean())
        rolling_std = grouped.transform(lambda x: x.rolling(window, min_periods=window).std())

        z_score = (df_long['lag1_returns'] - rolling_mean) / rolling_std
        return pd.DataFrame({'lag1_z_score': z_score})

    # No leakage
    def calculate_cumulative_return(df_long):
        """
        Calculate cumulative return using lag1 returns for each portfolio group
        """
        cum_ret = df_long.groupby('portfolio_id')['lag1_returns'].transform(
            lambda x: (1 + x).cumprod() - 1
        )
        return pd.DataFrame({'lag1_cumulative_return': cum_ret})
    
    # No leakage
    # TODO: Update to use adjust=True 
    def calculate_ema(df_long, span=20):
        """
        Calculate EMA using lag1 price for each portfolio group
        """
        ema = df_long.groupby('portfolio_id')['lag1_price'].transform(
            lambda x: x.ewm(span=span, adjust=False).mean()
        )
        return pd.DataFrame({'lag1_ema': ema})

    # No leakage
    def calculate_drawdown(df_long):
        """
        Calculate drawdown using lag1 returns for each portfolio group
        """
        grouped = df_long.groupby('portfolio_id')['lag1_returns']
        cumulative = grouped.transform(lambda x: (1 + x).cumprod())
        rolling_max = grouped.transform(lambda x: cumulative.cummax())

        drawdown = (cumulative - rolling_max) / rolling_max
        return pd.DataFrame({'lag1_drawdown': drawdown})

    # Potential leakage if lag_periods=0
    def add_lagged_features(df_long, lag_periods, feature):
        """
        Add additional lags beyond lag1 for each portfolio group
        """
        lagged_features = {}
        for lag in lag_periods:
            lagged = df_long.groupby('portfolio_id')[feature].transform(
                lambda x: x.shift(lag)  # -1 because we're already using lag1 data
            )
            lagged_features[f'lag{lag}_{feature}'] = lagged

        return pd.DataFrame(lagged_features)

    # No leakage
    def reconstruct_lag1_price(df_long, initial_price=100):
        return df_long.groupby('portfolio_id')['lag1_returns'].transform(
            lambda x: initial_price * (1 + x).cumprod()
        )

    # No leakage
    # NOTE: Manually calculates returns off price
    def calculate_n_day_return(df_long, n_days):
        grouped = df_long.groupby('portfolio_id')['lag1_price']
        return grouped.transform(lambda x: (x / x.shift(n_days)) - 1)

    # No leakage
    def calculate_rolling_volatility(df_long, window):
        grouped = df_long.groupby('portfolio_id')['lag1_returns']
        return grouped.transform(lambda x: x.rolling(window, min_periods=window).std())

    # No leakage
    def calculate_12m_momentum_skip_1m(df_long, skip_days=21, total_days=252):
        grouped = df_long.groupby('portfolio_id')['lag1_price']
        return grouped.transform(lambda x: (x.shift(skip_days) / x.shift(total_days)) - 1)

    # No leakage
    # NOTE: Manually calculates returns off price
    def calculate_return_skewness(df_long, ret_window, skew_window):
        grouped_price = df_long.groupby('portfolio_id')['lag1_price']
        n_day_returns = grouped_price.transform(lambda x: (x / x.shift(ret_window)) - 1)
        return n_day_returns.groupby(df_long['portfolio_id']).transform(
            lambda x: x.rolling(skew_window, min_periods=skew_window).skew()
        )

    # No leakage
    def calculate_moving_average_crossover(df_long, short_window=15, long_window=36):
        grouped = df_long.groupby('portfolio_id')['lag1_price']
        short_ma = grouped.transform(lambda x: x.rolling(short_window, min_periods=short_window).mean())
        long_ma = grouped.transform(lambda x: x.rolling(long_window, min_periods=long_window).mean())
        return np.sign(short_ma - long_ma)

    # TODO: Update function name
    def calculate_all_possible_features(df_long, initial_price=100):
        """
        Calculate initial basic set of functions
        """
        df_long['lag1_price'] = reconstruct_lag1_price(df_long, initial_price)
        feat = pd.DataFrame(index=df_long.index)

        # NOTE: These are all calculated using lag1 data
        feat['5d_price_return']   = calculate_n_day_return(df_long, 5)
        feat['10d_price_return']  = calculate_n_day_return(df_long, 10)
        feat['15d_price_return']  = calculate_n_day_return(df_long, 15)
        feat['21d_price_return']  = calculate_n_day_return(df_long, 21)
        feat['63d_price_return']  = calculate_n_day_return(df_long, 63)
        feat['126d_price_return'] = calculate_n_day_return(df_long, 126)
        feat['252d_price_return'] = calculate_n_day_return(df_long, 252)

        feat['3d_vol']  = calculate_rolling_volatility(df_long, 3)
        feat['6d_vol']  = calculate_rolling_volatility(df_long, 6)
        feat['12d_vol'] = calculate_rolling_volatility(df_long, 12)
        feat['100w_std_return'] = calculate_rolling_volatility(df_long, 500)

        feat['12m_momentum_skip_1m'] = calculate_12m_momentum_skip_1m(df_long)

        # TODO: Explore shifting different windows (ret_window != 5)
        feat['5d_ret_skew_5d']  = calculate_return_skewness(df_long, 5, 5)
        feat['5d_ret_skew_10d'] = calculate_return_skewness(df_long, 5, 10)
        feat['5d_ret_skew_15d'] = calculate_return_skewness(df_long, 5, 15)
        feat['5d_ret_skew_20d'] = calculate_return_skewness(df_long, 5, 20)

        feat['ma_crossover_signal'] = calculate_moving_average_crossover(df_long, 15, 36)
        return feat

    # No leakage
    # TODO: Add non-abs momentum calculations
    def calculate_momentum_abs_roll(df_long, windows=[5, 10, 21, 63]):
        """
        Calculate absolute momentum returns over multiple rolling windows for each portfolio group
        """
        features = {}
        grouped = df_long.groupby('portfolio_id')['lag1_returns']

        for window in windows:
            abs_mom = grouped.transform(
                lambda x: x.rolling(window, min_periods=window).apply(
                    lambda y: np.abs((1 + y).prod() - 1)
                )
            )
            features[f'lag1_abs_momentum_{window}d'] = abs_mom

        return pd.DataFrame(features)

    # No leakage
    # NOTE: Add overbought and oversold thresholds as input parameters
    def calculate_rsi_multi(df_long, windows=[14, 21, 30, 50]):
        """
        Calculate RSI, RSI overbought/oversold signals, and RSI differences using lag1_price for each portfolio group.
        """
        features = {}

        # Helper: Compute RSI for a single group, for a given window
        def _compute_rsi_for_window(x, window):
            """
            x is a Series of lag1_price for a single portfolio, indexed by date (or some time/order).
            """
            delta = x.diff()
            gain = delta.where(delta > 0, 0).rolling(window, min_periods=window).mean()
            loss = -delta.where(delta < 0, 0).rolling(window, min_periods=window).mean()

            rs = gain / loss
            # Avoid division-by-zero or NaN if loss=0 in some edge cases
            rsi = 100 - (100.0 / (1.0 + rs))
            return rsi

        # Store final RSI series for cross-window differences
        rsi_store = {}

        for w in windows:
            # Compute RSI per portfolio_id
            rsi_series = df_long.groupby('portfolio_id')['lag1_price'] \
                                .transform(lambda x: _compute_rsi_for_window(x, w))

            # Store the basic RSI
            rsi_name = f'lag1_rsi_{w}d'
            features[rsi_name] = rsi_series

            # Overbought / Oversold signals
            features[f'{rsi_name}_overbought'] = (rsi_series > 70).astype(int)
            features[f'{rsi_name}_oversold']   = (rsi_series < 30).astype(int)

            # Day-to-day RSI difference within each portfolio
            features[f'{rsi_name}_diff'] = rsi_series.groupby(df_long['portfolio_id']).diff()

            # Keep RSI for cross-window comparisons
            rsi_store[w] = rsi_series

        # Cross-window differences: RSI(window1) - RSI(window2)
        for i, w1 in enumerate(windows):
            for w2 in windows[i+1:]:
                diff_name = f'lag1_rsi_diff_{w1}_{w2}d'
                features[diff_name] = rsi_store[w1] - rsi_store[w2]

        return pd.DataFrame(features)

    # No leakage
    # TODO: Update to use adjust=True
    def calculate_macd_multi(df_long, periods=[(10, 63, 9), (15, 100, 9), (21, 200, 9)]):
        """
        Calculate MACD for multiple period combinations using lag1 price for each portfolio group
        """
        features = {}
        grouped = df_long.groupby('portfolio_id')['lag1_price']

        for short_win, long_win, sig_win in periods:
            short_ema = grouped.transform(
                lambda x: x.ewm(span=short_win, adjust=False).mean()
            )
            long_ema = grouped.transform(
                lambda x: x.ewm(span=long_win, adjust=False).mean()
            )

            macd = short_ema - long_ema
            signal = macd.groupby(df_long['portfolio_id']).transform(
                lambda x: x.ewm(span=sig_win, adjust=False).mean()
            )

            features[f'lag1_macd_{short_win}_{long_win}'] = macd
            features[f'lag1_macd_signal_{short_win}_{long_win}_{sig_win}'] = signal
            features[f'lag1_macd_hist_{short_win}_{long_win}'] = macd - signal

        return pd.DataFrame(features)

    # No leakage
    # NOTE: I don't like how the `k_pct` is calculated here because it relies on the pandas global index
    def calculate_stochastic_multi(df_long, k_windows=[14, 21, 30], d_windows=[3, 5, 7]):
        """
        Calculate Stochastic Oscillator for multiple window combinations using lag1 price
        """
        features = {}
        grouped = df_long.groupby('portfolio_id')['lag1_price']

        for k_window in k_windows:
            low_min = grouped.transform(lambda x: x.rolling(k_window, min_periods=k_window).min())
            high_max = grouped.transform(lambda x: x.rolling(k_window, min_periods=k_window).max())

            # Calculate %K
            k_pct = 100 * ((df_long['lag1_price'] - low_min) / (high_max - low_min)) # Don't like this because it relies on the pandas global index
            features[f'lag1_stoch_k_{k_window}d'] = k_pct

            # Calculate various %D (SMA of %K)
            for d_window in d_windows:
                d_pct = k_pct.groupby(df_long['portfolio_id']).transform(
                    lambda x: x.rolling(d_window, min_periods=d_window).mean()
                )
                features[f'lag1_stoch_d_{k_window}_{d_window}d'] = d_pct

                #  # Add overbought/oversold indicators
                #  features[f'lag1_stoch_{k_window}_{d_window}d_overbought'] = (d_pct > 80).astype(int)
                #  features[f'lag1_stoch_{k_window}_{d_window}d_oversold'] = (d_pct < 20).astype(int)

        return pd.DataFrame(features)

    # No leakage
    # NOTE: I don't like how the `k_pct` is calculated here because it relies on the pandas global index
    def calculate_bollinger_distances(df_long, windows=[20, 30, 50], k=2):
        """
        Calculate Bollinger Band metrics including distances, width, and period differences
        """
        features = {}
        grouped = df_long.groupby('portfolio_id')['lag1_price']

        # Store calculations for differences
        band_metrics = {}

        for window in windows:
            rolling_mean = grouped.transform(
                lambda x: x.rolling(window, min_periods=window).mean()
            )
            rolling_std = grouped.transform(
                lambda x: x.rolling(window, min_periods=window).std()
            )

            upper_band = rolling_mean + (k * rolling_std)
            lower_band = rolling_mean - (k * rolling_std)

            # Calculate distances (normalized by standard deviation for comparability)
            dist_to_upper = (upper_band - df_long['lag1_price']) / rolling_std
            dist_to_lower = (df_long['lag1_price'] - lower_band) / rolling_std

            # Calculate band width (normalized by mean price)
            band_width = (upper_band - lower_band) / rolling_mean

            # Store metrics for this window
            band_metrics[window] = {
                'upper_dist': dist_to_upper,
                'lower_dist': dist_to_lower,
                'width': band_width
            }

            # Store basic metrics
            features[f'lag1_bband_upper_dist_{window}d'] = dist_to_upper
            features[f'lag1_bband_lower_dist_{window}d'] = dist_to_lower
            features[f'lag1_bband_width_{window}d'] = band_width

            # Calculate rate of change for each metric
            features[f'lag1_bband_upper_dist_{window}d_diff'] = dist_to_upper.groupby(df_long['portfolio_id']).diff()
            features[f'lag1_bband_lower_dist_{window}d_diff'] = dist_to_lower.groupby(df_long['portfolio_id']).diff()
            features[f'lag1_bband_width_{window}d_diff'] = band_width.groupby(df_long['portfolio_id']).diff()

        # Calculate differences between periods
        for i, window1 in enumerate(windows):
            for window2 in windows[i+1:]:
                features[f'lag1_bband_upper_dist_diff_{window1}_{window2}d'] = (
                    band_metrics[window1]['upper_dist'] - band_metrics[window2]['upper_dist']
                )
                features[f'lag1_bband_lower_dist_diff_{window1}_{window2}d'] = (
                    band_metrics[window1]['lower_dist'] - band_metrics[window2]['lower_dist']
                )
                features[f'lag1_bband_width_diff_{window1}_{window2}d'] = (
                    band_metrics[window1]['width'] - band_metrics[window2]['width']
                )

        return pd.DataFrame(features)

    # No leakage
    # NOTE: Global index. Do not like how ewm stands for exponential moving window. The acronym is wrong. 
    def calculate_ema_distances(df_long, spans=[10,15,21,63,100,200]):
        """
        Calculate distances from EMAs and EMA crossovers using lag1 price for each portfolio group
        """
        features = {}
        grouped = df_long.groupby('portfolio_id')['lag1_price']

        # Calculate EMAs for all spans
        emas = {}
        for span in spans:
            emas[span] = grouped.transform(
                lambda x: x.ewm(span=span, adjust=False).mean()
            )
            # Calculate distance from price
            dist = (df_long['lag1_price'] - emas[span]) / df_long['lag1_price']
            features[f'lag1_ema_dist_{span}d'] = dist

        # Calculate differences between EMAs
        for i, span1 in enumerate(spans):
            for span2 in spans[i+1:]:
                # Calculate difference between EMAs normalized by price
                ema_diff = (emas[span1] - emas[span2]) / df_long['lag1_price']
                features[f'lag1_ema_diff_{span1}_{span2}d'] = ema_diff

        return pd.DataFrame(features)

    # No leakage
    # NOTE: Global index
    def calculate_sharpe_multi_diffs(df_long, windows=[20, 30, 60, 90], risk_free_rate=0.0):
        """
        Calculate Sharpe ratios and differences between windows using lag1_returns, grouped by portfolio.
        """
        features = {}

        # Helper: Compute rolling Sharpe ratio for a single group's returns
        def _sharpe_for_window(x, window):
            """
            x is a Series of lag1_returns for a single portfolio.
            """
            excess = x - risk_free_rate
            rolling_mean = excess.rolling(window, min_periods=window).mean()
            rolling_std  = excess.rolling(window, min_periods=window).std()
            # Sharpe = mean / std
            sharpe = rolling_mean / rolling_std
            return sharpe

        sharpe_store = {}

        for w in windows:
            # Compute Sharpe per portfolio
            sharpe_series = df_long.groupby('portfolio_id')['lag1_returns'] \
                                .transform(lambda x: _sharpe_for_window(x, w))

            sharpe_name = f'lag1_sharpe_{w}d'
            features[sharpe_name] = sharpe_series

            # Day-to-day difference of the Sharpe ratio
            features[f'{sharpe_name}_diff'] = sharpe_series.groupby(df_long['portfolio_id']).diff()

            sharpe_store[w] = sharpe_series

        # Cross-window Sharpe differences: Sharpe(w1) - Sharpe(w2)
        for i, w1 in enumerate(windows):
            for w2 in windows[i+1:]:
                diff_name = f'lag1_sharpe_diff_{w1}_{w2}d'
                features[diff_name] = sharpe_store[w1] - sharpe_store[w2]

        return pd.DataFrame(features)

    # No leakage
    # NOTE: Global index
    # TODO: Review this closely as construction of numpy arrays and re-indexing raises suspicion
    def calculate_egarch_signals(df_long,
                                windows=[20, 50, 100],
                                alpha=0.1,
                                beta=0.9,
                                time_col='date'):
        """
        Calculate E-GARCH volatility metrics for each portfolio using lag1_returns.
        """

        # Ensure df_long is sorted by (portfolio_id, time_col)
        # so that rolling/window operations happen in time order.
        df_sorted = df_long.sort_values(['portfolio_id', time_col]).copy() # Sort again - leaving it in

        features = {}

        # We'll do everything on df_sorted and preserve the same index
        # so we can reassign final columns in the same alignment.

        # For each window:
        for window in windows:
            # 1) Rolling mean (with partial windows allowed)
            #    We do an apply approach to ensure no cross-portfolio bleed in rolling.
            rolling_mean_series = (
                df_sorted
                .groupby('portfolio_id', group_keys=False)['lag1_returns']
                .apply(lambda s: s.rolling(window, min_periods=1).mean())
            )
            # Reindex to original
            rolling_mean_series = rolling_mean_series.reindex(df_sorted.index)

            # 2) Residuals = returns - rolling mean
            residuals_series = df_sorted['lag1_returns'] - rolling_mean_series

            # Prepare empty arrays (aligned to df_sorted.index)
            log_var_arr = np.full(len(df_sorted), np.nan, dtype=float)
            std_resid_arr = np.full(len(df_sorted), np.nan, dtype=float)

            # 3) eGARCH recursion per portfolio
            for pid, group_df in df_sorted.groupby('portfolio_id'):
                # Sort by date (already sorted, but just to be explicit)
                # group_df = group_df.sort_values(time_col)  # not needed if we used df_sorted
                group_index = group_df.index

                # Extract the group residuals in time order
                grp_resid = residuals_series.loc[group_index].values

                n = len(grp_resid)
                if n == 0:
                    continue  # no data for this portfolio

                group_var = np.zeros(n, dtype=float)
                group_log_var = np.zeros(n, dtype=float)

                # Initialize with variance of first 'window' points or entire group if smaller
                # We'll take the actual non-NaN portion for the init variance
                valid_init = grp_resid[:window][~np.isnan(grp_resid[:window])]
                if len(valid_init) > 1:
                    init_var = np.var(valid_init)
                else:
                    # If we have 0 or 1 valid data points, set var to something small
                    init_var = 1e-9

                group_var[0] = init_var
                group_log_var[0] = np.log(init_var + 1e-9)

                # eGARCH recursion in time order
                for t in range(1, n):
                    # z_{t-1} = resid_{t-1} / sqrt(var_{t-1})
                    prev_var = group_var[t-1] + 1e-9  # prevent zero-division
                    z = grp_resid[t-1] / np.sqrt(prev_var)

                    if t >= window:
                        # log(sigma^2_t) = (1 - alpha - beta)*log(sigma^2_0)
                        #                 + alpha*(|z| - E|z|)
                        #                 + beta*log(sigma^2_{t-1})
                        log_var_t = ((1 - alpha - beta) * group_log_var[0]
                                    + alpha * (abs(z) - np.sqrt(2/np.pi))
                                    + beta * group_log_var[t-1])
                        group_log_var[t] = log_var_t
                        group_var[t] = np.exp(log_var_t)
                    else:
                        # If we haven't hit 'window' yet, we keep it constant from t=0
                        group_log_var[t] = group_log_var[0]
                        group_var[t] = group_var[0]

                # Assign back to arrays
                log_var_arr[group_index] = group_log_var
                std_resid_arr[group_index] = grp_resid / np.sqrt(group_var + 1e-9)

            # 4) E-GARCH metrics
            egarch_vol_arr = np.sqrt(np.exp(log_var_arr))  # sqrt( exp(log variance) )
            vol_name = f'lag1_egarch_vol_{window}d'

            # Store in features dict with the SAME index as df_sorted
            features[vol_name] = pd.Series(egarch_vol_arr, index=df_sorted.index)
            features[f'lag1_egarch_log_var_{window}d'] = pd.Series(log_var_arr, index=df_sorted.index)
            features[f'lag1_egarch_std_resid_{window}d'] = pd.Series(std_resid_arr, index=df_sorted.index)

            # 5) Volatility day-to-day difference, grouped by portfolio
            diff_series = (
                features[vol_name]
                .groupby(df_sorted['portfolio_id'], group_keys=False)
                .diff()
            )
            features[f'{vol_name}_diff'] = diff_series

        # Convert the features to a DataFrame, aligned to df_sorted.index
        features_df = pd.DataFrame(features, index=df_sorted.index)

        # Finally, reindex to the original df_long's index order if it was different
        # (If df_long has the same ordering, this is no-op; otherwise we align back.)
        features_df = features_df.reindex(df_long.index)

        return features_df

    # No leakage
    def calculate_variance_changes(df_long, windows=[20, 50, 100]):
        """
        Calculate variance and variance changes over multiple windows
        """
        features = {}
        grouped = df_long.groupby('portfolio_id')['lag1_returns']

        # Store variances for differences
        var_values = {}

        for window in windows:
            # Calculate rolling variance
            rolling_var = grouped.transform(
                lambda x: x.rolling(window, min_periods=window).var()
            )
            var_values[window] = rolling_var

            # Store basic variance
            features[f'lag1_variance_{window}d'] = rolling_var

            # Calculate log variance
            features[f'lag1_log_variance_{window}d'] = np.log(rolling_var)

            # Calculate rate of change in variance
            features[f'lag1_variance_diff_{window}d'] = rolling_var.groupby(df_long['portfolio_id']).diff()
            features[f'lag1_variance_pct_change_{window}d'] = rolling_var.groupby(df_long['portfolio_id']).pct_change()

        # Calculate differences between periods
        for i, window1 in enumerate(windows):
            for window2 in windows[i+1:]:
                features[f'lag1_variance_diff_{window1}_{window2}d'] = (
                    var_values[window1] - var_values[window2]
                )
                # Add ratio between variances
                features[f'lag1_variance_ratio_{window1}_{window2}d'] = (
                    var_values[window1] / var_values[window2]
                )

        return pd.DataFrame(features)

    def calculate_all_features(df_long, initial_price=100):
        """
        Calculate all features using lag1 data for the long-format DataFrame
        """
        # First calculate lag1 price
        df_long['lag1_price'] = reconstruct_lag1_price(df_long, initial_price)


        if addition_technical == 1:

            # Calculate all features
            features = pd.concat([
                calculate_bollinger_bands(df_long),
                calculate_channel_breakouts(df_long),
                calculate_momentum(df_long),
                calculate_moving_average(df_long),
                calculate_macd(df_long),
                calculate_rsi(df_long),
                calculate_support_resistance(df_long),
                calculate_volatility(df_long),
                calculate_sharpe_ratio(df_long),
                calculate_z_score(df_long),
                calculate_cumulative_return(df_long),
                calculate_ema(df_long),
                calculate_drawdown(df_long),
                add_lagged_features(df_long, [3, 5, 7], "returns"),
                add_lagged_features(df_long, [3, 5, 7], "rank"),
                calculate_all_possible_features(df_long),
                calculate_momentum_abs_roll(df_long, windows = [2,5,10,21,63,126,252]),
                calculate_macd_multi(df_long, periods=[(8, 21, 9), (12, 26, 9), (20, 40, 10)]),
                calculate_rsi_multi(df_long, windows=[3, 14, 30]),
                calculate_stochastic_multi(df_long, k_windows=[14, 21, 30], d_windows=[3, 5, 7]),
                calculate_bollinger_distances(df_long, windows=[20, 30, 50], k=2 ),
                calculate_ema_distances(df_long, spans=[10,15,21,63,100,200]),
                calculate_sharpe_multi_diffs(df_long, windows=[20, 30, 60, 90]),
                calculate_egarch_signals(df_long, windows=[20, 50, 100]),
                calculate_variance_changes(df_long, windows=[20, 50, 100])
            ], axis=1)

        else:
            features = pd.concat([
                calculate_bollinger_bands(df_long),
                calculate_channel_breakouts(df_long),
                calculate_momentum(df_long),
                calculate_moving_average(df_long),
                calculate_macd(df_long),
                calculate_rsi(df_long),
                calculate_support_resistance(df_long),
                calculate_volatility(df_long),
                calculate_sharpe_ratio(df_long),
                calculate_z_score(df_long),
                calculate_cumulative_return(df_long),
                calculate_ema(df_long),
                calculate_drawdown(df_long),
                add_lagged_features(df_long, [3, 5, 7], "returns"),
                add_lagged_features(df_long, [3, 5, 7], "rank"),
                calculate_all_possible_features(df_long)
            ], axis=1)

        return features

    # We may deprecate this as it is currently not being used
    # TODO: Review closely as the LLMs indicated this could be the a source of leakage if used
    def calculate_cross_asset_dynamics(df_long, portfolios=None, lags=[1, 3, 5, 10, 21], windows=[63, 126, 252]):
        """
        Calculate lead-lag correlations between portfolios to identify predictive relationships.
        """
        if portfolios is None:
            portfolios = df_long['portfolio_id'].unique()
        
        # Create an empty DataFrame to store results
        lead_lag_features = pd.DataFrame(index=df_long.index)
        
        # Sort data by date and portfolio
        df_long = df_long.sort_values(['date', 'portfolio_id']).copy()
        
        # Create a pivot table of returns with date as index and portfolio_id as columns
        pivot_returns = df_long.pivot(index='date', columns='portfolio_id', values='lag1_returns')
        
        # For each portfolio pair
        for i, port_a in enumerate(portfolios):
            for port_b in portfolios:
                # Skip self-correlations if desired
                if port_a == port_b:
                    continue
                    
                # Get the return series for both portfolios
                returns_a = pivot_returns[port_a]
                returns_b = pivot_returns[port_b]
                
                # For each lag period
                for lag in lags:
                    # Calculate A(t) leading B(t+lag)
                    a_leads_b = returns_a.shift(lag).to_frame().join(returns_b, how='inner')
                    a_leads_b.columns = ['a_lead', 'b']
                    
                    # Calculate B(t) leading A(t+lag)
                    b_leads_a = returns_b.shift(lag).to_frame().join(returns_a, how='inner')
                    b_leads_a.columns = ['b_lead', 'a']
                    
                    # Calculate rolling correlations for each window size
                    for window in windows:
                        # A leads B: corr(A(t-lag), B(t))
                        a_leads_b_corr = a_leads_b['a_lead'].rolling(window).corr(a_leads_b['b'])
                        a_leads_b_corr_name = f'lag{lag}_corr_{port_a}_leads_{port_b}_{window}d'
                        
                        # B leads A: corr(B(t-lag), A(t))
                        b_leads_a_corr = b_leads_a['b_lead'].rolling(window).corr(b_leads_a['a'])
                        b_leads_a_corr_name = f'lag{lag}_corr_{port_b}_leads_{port_a}_{window}d'
                        
                        # Store correlation series with their timestamps
                        lead_lag_df = pd.DataFrame({
                            a_leads_b_corr_name: a_leads_b_corr,
                            b_leads_a_corr_name: b_leads_a_corr
                        })
                        
                        # Calculate correlation differential (which portfolio has stronger predictive power)
                        diff_name = f'lag{lag}_corr_diff_{port_a}_{port_b}_{window}d'
                        lead_lag_df[diff_name] = a_leads_b_corr - b_leads_a_corr
                        
                        # Calculate correlation sign stability
                        sign_stability_a_leads = lead_lag_df[a_leads_b_corr_name].rolling(window//2).apply(
                            lambda x: np.abs(np.sign(x).mean()), raw=True
                        )
                        lead_lag_df[f'{a_leads_b_corr_name}_sign_stability'] = sign_stability_a_leads
                        
                        sign_stability_b_leads = lead_lag_df[b_leads_a_corr_name].rolling(window//2).apply(
                            lambda x: np.abs(np.sign(x).mean()), raw=True
                        )
                        lead_lag_df[f'{b_leads_a_corr_name}_sign_stability'] = sign_stability_b_leads
                        
                        # Merge with main features DataFrame
                        lead_lag_df = lead_lag_df.reset_index()
                        
                        # Map the correlations back to the original long-format data
                        for feature_name in lead_lag_df.columns:
                            if feature_name != 'date':
                                # Create a mapping of date -> correlation value
                                date_to_corr = dict(zip(lead_lag_df['date'], lead_lag_df[feature_name]))
                                
                                # Add the correlation as a feature to all rows with portfolio_id == port_a
                                mask_a = df_long['portfolio_id'] == port_a
                                lead_lag_features.loc[mask_a, feature_name] = df_long.loc[mask_a, 'date'].map(date_to_corr)
                                
                                # If we want both portfolios to see this feature
                                mask_b = df_long['portfolio_id'] == port_b
                                lead_lag_features.loc[mask_b, feature_name] = df_long.loc[mask_b, 'date'].map(date_to_corr)
        
        # Fill NaN values with 0 (or another appropriate strategy)
        lead_lag_features = lead_lag_features.fillna(0)
        
        return lead_lag_features

    #####################################################################
    #  4. MELT + FEATURE MERGE
    #####################################################################

    # Convert the wide-format DataFrame to long format with date as the index
    df_long = pd.melt(df,
                id_vars=id_vars,
                value_vars=portfolio_cols,
                var_name='portfolio_id',
                value_name='returns')

    # Sort by date and portfolio_id ascending
    df_long = df_long.sort_values(['date', 'portfolio_id']).reset_index(drop=True)

    def calculate_rolling_skewness(df, window):
        """Compute rolling skewness for each portfolio over a specified window."""
        return df.groupby('portfolio_id')['lag1_returns'].transform(
            lambda x: x.rolling(window, min_periods=window).apply(
                lambda y: skew(y, nan_policy='omit'), raw=False
            )
        )

    def calculate_rolling_kurtosis(df, window):
        """Compute rolling kurtosis for each portfolio over a specified window."""
        return df.groupby('portfolio_id')['lag1_returns'].transform(
            lambda x: x.rolling(window, min_periods=window).apply(
                lambda y: kurtosis(y, nan_policy='omit'), raw=False
            )
        )

    # Generate ranks by day
    df_long['rank'] = df_long.groupby('date')['returns'].transform(lambda x: x.rank(ascending=True))

    # Calculate initial lags and all factors
    initial_price=100
    df_long['lag1_returns'] = df_long.groupby('portfolio_id')['returns'].shift(return_lag)
    df_long['price'] = df_long.groupby('portfolio_id')['returns'].transform(
            lambda x: initial_price * (1 + x).cumprod())
    features = calculate_all_features(df_long, initial_price=100)

    # Combine features with original data
    # TODO: Update this to an actual join on date and portfolio_id if possible
    df_long = pd.concat([df_long, features], axis=1)

    # Join economic factors
    df_long = df_long.join(df_factors.shift(factor_lag).add_prefix(f'fact_lag{factor_lag}_'), on="date")
    
    # Factorize date for use in LearnToRank models if needed
    df_long["qid"] = pd.factorize(df_long["date"])[0]

    # Sort by date and portfolio_id
    df_long = df_long.sort_values(by=['date']).reset_index(drop=True)
    
    # Log-transform returnsall columns with "returns" in the name
    return_cols = [x for x in df_long.columns if "returns" in x]
    df_long[return_cols] = np.log(df_long[return_cols]+1)
    
    # Sort by date and portfolio_id
    df_long = df_long.sort_values(["date", "portfolio_id"]).reset_index(drop=True)

    # Add more skewness and kurtosis features
    if include_skew_kurt == 1:
        start_skew = time.time()
        # Apply to 30-day and 252-day windows
        df_long['skewness_30d']  = calculate_rolling_skewness(df_long, 30)
        df_long['kurtosis_30d']  = calculate_rolling_kurtosis(df_long, 30)
        df_long['skewness_60d']  = calculate_rolling_skewness(df_long, 60)
        df_long['kurtosis_60d']  = calculate_rolling_kurtosis(df_long, 60)
        df_long['skewness_180d'] = calculate_rolling_skewness(df_long, 180)
        df_long['kurtosis_180d'] = calculate_rolling_kurtosis(df_long, 180)
        df_long['skewness_252d'] = calculate_rolling_skewness(df_long, 252)
        df_long['kurtosis_252d'] = calculate_rolling_kurtosis(df_long, 252)
        print(f"Skew/Kurtosis calculation time:   {time.time() - start_skew:.3f} s")

    # Add time based features
    if include_time_features == 1:
    # Extract day-of-week and month
        df_long['day_of_week'] = df_long['date'].dt.dayofweek  # 0 = Monday, 6 = Sunday
        df_long['month'] = df_long['date'].dt.month  # 1 = January, 12 = December

        # Cyclical encoding using sine and cosine transformations
        df_long['day_of_week_sin'] = np.sin(2 * np.pi * df_long['day_of_week'] / 7)
        df_long['day_of_week_cos'] = np.cos(2 * np.pi * df_long['day_of_week'] / 7)
        df_long['month_sin'] = np.sin(2 * np.pi * df_long['month'] / 12)
        df_long['month_cos'] = np.cos(2 * np.pi * df_long['month'] / 12)


    # TODO: Add logic to turning cross-asset on/off 
    # NOTE: Previous tests added ~1.5K new columns with 19 portfolios and degraded performance. Revisit with better approach.
    # cross_asset_features = calculate_cross_asset_dynamics(
    #     df_long, 
    #     portfolios=list(df_long['portfolio_id'].unique()),  # Limit to first 5 portfolios to manage feature count
    #     lags=[5],
    #     windows=[63]
    # )
    # # Concatenate with other features
    # df_long = pd.concat([df_long, cross_asset_features], axis=1)


    #####################################################################
    #  4.1 ADD NEW FEATURES
    #####################################################################

    start_date  = "2004-01-01"
    end_date    = "2025-01-17"

    # TODO: Deprecate this 
    def get_features(start_date, end_date, feature_table):
        query = f"""
        SELECT
            *
        FROM `{feature_table}`
        WHERE date >= '{start_date}'
        AND date <= '{end_date}'
        """
        df = pd.read_gbq(query, project_id='issachar-feature-library', use_bqstorage_api=True)
        df_wide = df.pivot(
            index=["date", "cluster"],
            columns=["side", "feature"],
            values="value"
        )
        df_wide.columns = [
            "__".join(col_tuple) for col_tuple in df_wide.columns.to_flat_index()
        ]
        df_wide = df_wide.reset_index()
        df_wide = df_wide[df_wide['cluster'].apply(lambda x: "inverse" not in x)]
        df_wide = df_wide.rename(columns={'cluster': 'portfolio_id'})
        df_wide = df_wide.sort_values(['date', 'portfolio_id'])
        columns_to_lag = [x for x in df_wide.columns if x not in ['date', 'portfolio_id']]

        for col in columns_to_lag:
            df_wide[f'{col}_lag1'] = df_wide.groupby('portfolio_id')[col].shift(1)

        df_wide = df_wide.drop(columns_to_lag, axis=1)
        return df_wide

    if include_fundamentals == 1:
        # Retrieve feature data
        fundamental_features = get_features(start_date, end_date, "issachar-feature-library.qjg.scaled_fund_features_long")
        price_features       = get_features(start_date, end_date, "issachar-feature-library.qjg.scaled_price_features_long")
        lib_v1_features      = get_features(start_date, end_date, "issachar-feature-library.qjg.scaled_lib_v1_features_long")


        # Merge the feature data into df_long on 'date' and 'portfolio_id'
        df_long = (df_long
                .merge(fundamental_features, on=['date', 'portfolio_id'], how='left')
                .merge(price_features, on=['date', 'portfolio_id'], how='left')
                .merge(lib_v1_features, on=['date', 'portfolio_id'], how='left')
                )
    # end deprecate this
    
    # TODO: Deprecate this
    if include_cluster_data == 1:
        query = '''
            SELECT
            data.*,
            yields.`2yr REAL Yield`
        FROM `issachar-feature-library.qjg.clustering_data` AS data
        LEFT OUTER JOIN `issachar-feature-library.qjg.clustering_data_yields` AS yields
        ON data.date = yields.date
        '''
        project_id = "issachar-feature-library"

        client = bigquery.Client(project=project_id)

        query_job = client.query(query)
        cluster_data = query_job.to_dataframe()
        cluster_data['date'] = pd.to_datetime(cluster_data['date']).dt.tz_localize(None)
        cluster_data = cluster_data.sort_values(['date'])
        cluster_data.loc[:, cluster_data.columns != 'date'] = cluster_data.loc[:, cluster_data.columns != 'date'].shift(1)
        df_long = df_long.merge(cluster_data, on=['date'], how='left')

    # NOTE: Not useful features
    if include_coint_regimes == 1:
        query = "SELECT * FROM `issachar-feature-library.qjg.coint_regimes_test_01`"
        project_id = "issachar-feature-library"

        client = bigquery.Client(project=project_id)

        query_job = client.query(query)
        coint_df = query_job.to_dataframe()
        coint_df['date'] = coint_df['date'].dt.tz_localize(None)
        coint_df = coint_df.sort_values(["date", "portfolio_id"])

        # Identify columns to shift (all except 'date' and 'portfolio_id')
        cols_to_shift = [col for col in coint_df.columns if col not in ['date', 'portfolio_id']]

        # Shift columns within each portfolio_id group by one row
        coint_df[cols_to_shift] = coint_df.groupby('portfolio_id')[cols_to_shift].shift(1)

        # Convert the shifted columns to float
        coint_df[cols_to_shift] = coint_df[cols_to_shift].astype(float)

        # Merge the updated DataFrame with df_long on 'date' and 'portfolio_id'
        df_long = df_long.merge(coint_df, on=['date', 'portfolio_id'], how='left')

    # NOTE: Not useful features
    if include_hmm_regimes == 1:
        query = "SELECT * FROM `issachar-feature-library.qjg.hmm_regimes_test_01`"
        project_id = "issachar-feature-library"

        client = bigquery.Client(project=project_id)

        query_job = client.query(query)
        hmm_df = query_job.to_dataframe()
        hmm_df['date'] = pd.to_datetime(hmm_df['date']).dt.tz_localize(None)
        hmm_df = hmm_df.sort_values(["date", "portfolio_id"])

        # Identify columns to shift (all except 'date' and 'portfolio_id')
        cols_to_shift = [col for col in hmm_df.columns if col not in ['date', 'portfolio_id']]

        # Shift the specified columns within each 'portfolio_id' group by one row
        hmm_df[cols_to_shift] = hmm_df.groupby('portfolio_id')[cols_to_shift].shift(1)

        # Convert the shifted columns to float
        hmm_df[cols_to_shift] = hmm_df[cols_to_shift].astype(float)

        # Merge the updated hmm_df with df_long on 'date' and 'portfolio_id'
        df_long = df_long.merge(hmm_df, on=['date', 'portfolio_id'], how='left')

    # Different versions of Will's features
    if include_will_features > 0:

        if include_will_features == 1:
            query = '''
                SELECT *
                FROM (
                    SELECT date, signal_rank_daily, pred_vol
                    FROM issachar-feature-library.wmg.top_signals_info_density
                )
                PIVOT(
                    MAX(pred_vol)
                    FOR signal_rank_daily IN (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
                )
                ORDER BY date;
                '''

        elif include_will_features == 2:
            query = '''
                SELECT *
                FROM (
                    SELECT date, ranking, predicted_target
                    FROM `issachar-feature-library.wmg.top_signals_info_density2`
                )
                PIVOT(
                    MAX(predicted_target)
                    FOR ranking IN (
                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                        31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                        41, 42, 43, 44, 45, 46, 47, 48, 49, 50
                    )
                )
                ORDER BY date;
            '''


        elif include_will_features == 3:
            query = '''
                    SELECT *
                    FROM (
                        SELECT date, CAST(rank AS INT) AS rank, predicted_target
                        FROM `issachar-feature-library.wmg.top_signals_info_density3`
                    )
                    PIVOT(
                        MAX(predicted_target)
                        FOR rank IN (
                            1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                            11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                            21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                            31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                            41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                            51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                            61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
                            71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                            81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
                            91, 92, 93, 94, 95, 96, 97, 98, 99, 100
                        )
                    )
                    ORDER BY date;
                    '''
        elif include_will_features == 4:
            query = '''
                    SELECT *
                    FROM (
                        SELECT date, CAST(rank AS INT) AS rank, predicted_target
                        FROM `issachar-feature-library.wmg.top_signals_info_density4`
                    )
                    PIVOT(
                        MAX(predicted_target)
                        FOR rank IN (
                            1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                            11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                            21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                            31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                            41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                            51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                            61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
                            71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                            81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
                            91, 92, 93, 94, 95, 96, 97, 98, 99, 100
                        )
                    )
                    ORDER BY date;
                    '''
        elif include_will_features == 5:
            query = '''
                    SELECT *
                    FROM (
                        SELECT date, CAST(rank AS INT) AS rank, composite_dynamic
                        FROM `issachar-feature-library.wmg.top_signals_composite_score1`
                    )
                    PIVOT(
                        MAX(composite_dynamic)
                        FOR rank IN (
                            1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                            11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                            21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                            31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                            41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                            51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                            61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
                            71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                            81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
                            91, 92, 93, 94, 95, 96, 97, 98, 99, 100
                        )
                    )
                    ORDER BY date;
                    '''

        project_id = "issachar-feature-library"

        client = bigquery.Client(project=project_id)

        query_job = client.query(query)
        will_df = query_job.to_dataframe()
        will_df = will_df.rename(columns=dict([(x, f"will_feature{x}") for x in will_df.columns if "_" in x]))
        df_long = df_long.merge(will_df, on=['date'], how='left')

    # NOTE: As of 2025-03-07 the precision was not good enough to use
    if will_predictions == 1:
        # Regular IC direction prediction
        query = '''
        SELECT date, signal as portfolio_id, prediction as ic_direction_pred1 FROM `issachar-feature-library.wmg.ic_direction_pred1` 
        WHERE prediction_made = 1 
        ORDER BY DATE asc
            '''

        project_id = "issachar-feature-library"

        client = bigquery.Client(project=project_id)

        query_job = client.query(query)
        ic_pred1_df = query_job.to_dataframe()
        ic_pred1_df["ic_direction_pred1"] = ic_pred1_df["ic_direction_pred1"].fillna(0).astype('int64')

        # IC direction prediction with Z-score update
        query = '''
        SELECT date, signal as portfolio_id, prediction as ic_direction_pred2 FROM `issachar-feature-library.wmg.ic_direction_pred2` 
        WHERE prediction_made = 1 
        ORDER BY DATE asc
            '''
        query_job = client.query(query)
        ic_pred2_df = query_job.to_dataframe()
        ic_pred2_df["ic_direction_pred2"] = ic_pred2_df["ic_direction_pred2"].fillna(0).astype('int64')

        df_long = df_long.merge(ic_pred1_df, on=['date', 'portfolio_id'], how='left')
        df_long = df_long.merge(ic_pred2_df, on=['date', 'portfolio_id'], how='left')

    def calculate_rolling_correlation(df_long, col1, col2, windows):
        """
        Calculate rolling correlation between two columns of the input DataFrame,
        grouped by 'portfolio_id', for multiple rolling window sizes.
        """
        features = {}
        
        for window in windows:
            # Compute the rolling correlation for each portfolio group
            rolling_corr_series = df_long.groupby('portfolio_id').apply(
                lambda group: group[col1].rolling(window, min_periods=window).corr(group[col2])
            )
            # Reset the multi-index (created by groupby.apply) so that it aligns with the original DataFrame
            rolling_corr_series = rolling_corr_series.reset_index(level=0, drop=True)
            # Define the output column name
            col_name = f'rolling_corr_{col1}_{col2}_{window}'
            features[col_name] = rolling_corr_series

        return pd.DataFrame(features)
    
    def calculate_rolling_corr_diffs(df_long, col1, col2, windows):
        """
        Calculate rolling correlations between two columns (col1 and col2) for multiple window lengths,
        along with:
        - Day-to-day differences of each rolling correlation.
        - Cross-window differences between the rolling correlations.
        """
        features = {}
        corr_store = {}

        # 1. Compute rolling correlations and their day-to-day differences for each window.
        for window in windows:
            # Compute rolling correlation per portfolio group.
            rolling_corr_series = df_long.groupby('portfolio_id').apply(
                lambda group: group[col1].rolling(window, min_periods=window).corr(group[col2])
            )
            # Reset the multi-index to align with the original DataFrame.
            rolling_corr_series = rolling_corr_series.reset_index(level=0, drop=True)
            col_name = f'rolling_corr_{col1}_{col2}_{window}'
            features[col_name] = rolling_corr_series
            # Store for later cross-window diff calculation.
            corr_store[window] = rolling_corr_series
            # Day-to-day difference of this rolling correlation.
            features[f'{col_name}_diff'] = rolling_corr_series.groupby(df_long['portfolio_id']).diff()

        # 2. Compute cross-window differences: for each pair of windows, subtract the corresponding correlations.
        for i, w1 in enumerate(windows):
            for w2 in windows[i+1:]:
                diff_name = f'rolling_corr_diff_{col1}_{col2}_{w1}_{w2}'
                features[diff_name] = corr_store[w1] - corr_store[w2]

        return pd.DataFrame(features)

    def calculate_rolling_corr_diffs_with_rsi_vol(df_long, col1, col2, windows,
                                                rsi_windows=[14, 21, 30, 50],
                                                vol_window=20):
        """
        Calculate rolling correlations between two columns (col1 and col2) for multiple window lengths,
        along with:
        - Day-to-day differences for each rolling correlation.
        - Cross-window differences between the rolling correlations.
        - RSI values for each correlation series (using specified RSI windows).
        - Rolling volatility (standard deviation) for each correlation series.
        """
        features = {}
        corr_store = {}

        # Helper: Compute RSI for a given series and window.
        def _compute_rsi(series, window):
            delta = series.diff()
            gain = delta.where(delta > 0, 0).rolling(window, min_periods=window).mean()
            loss = -delta.where(delta < 0, 0).rolling(window, min_periods=window).mean()
            rs = gain / loss
            rsi = 100 - (100.0 / (1.0 + rs))
            return rsi

        # 1. Compute rolling correlations and their day-to-day differences for each window.
        for window in windows:
            # Compute rolling correlation for each portfolio.
            rolling_corr_series = df_long.groupby('portfolio_id').apply(
                lambda group: group[col1].rolling(window, min_periods=window).corr(group[col2])
            )
            # Reset multi-index (from groupby.apply) so that it aligns with df_long.
            rolling_corr_series = rolling_corr_series.reset_index(level=0, drop=True)
            corr_col_name = f'rolling_corr_{col1}_{col2}_{window}'
            features[corr_col_name] = rolling_corr_series
            corr_store[window] = rolling_corr_series
            
            # Day-to-day difference for the rolling correlation.
            features[f'{corr_col_name}_diff'] = rolling_corr_series.groupby(df_long['portfolio_id']).diff()

        # 2. Compute cross-window differences: correlation(window1) - correlation(window2)
        for i, w1 in enumerate(windows):
            for w2 in windows[i+1:]:
                diff_name = f'rolling_corr_diff_{col1}_{col2}_{w1}_{w2}'
                features[diff_name] = corr_store[w1] - corr_store[w2]

        # 3. Compute RSI and volatility for each correlation series (but not for the diffs).
        for window in windows:
            corr_series = corr_store[window]
            base_name = f'rolling_corr_{col1}_{col2}_{window}'

            # Compute RSI for each specified RSI window.
            for rsi_w in rsi_windows:
                rsi_series = corr_series.groupby(df_long['portfolio_id']) \
                                        .transform(lambda x: _compute_rsi(x, rsi_w))
                features[f'{base_name}_rsi_{rsi_w}'] = rsi_series

            # Compute rolling volatility for the correlation series.
            vol_series = corr_series.groupby(df_long['portfolio_id']) \
                                    .transform(lambda x: x.rolling(vol_window, min_periods=vol_window).std())
            features[f'{base_name}_volatility_{vol_window}'] = vol_series

        return pd.DataFrame(features)

    if use_correlations > 0:
        # Recommended rolling windows (in days)
        windows = [20, 63, 126]

        # List of features to compute rolling correlations against "long_momentum"
        factor_list = [
            'short_momentum',
            'thrm_momo_long',
            'long_value',
            'us_small_caps',
            'inflation',
            'xle',
            'tlt',
            'vxn_index',
            'high_beta_cyclicals',
            'qqq',
            'high_vol',
            'two_vs_5_spread',
            'crude_levered_equities',
            'low_vol',
            'two_yr',
            'secular_growth',
            'reflationary_cyclicals',
            'inflation_levered',
            'high_short_interest'
        ]

        # List to hold the rolling correlation DataFrames for each feature
        rolling_corr_dfs = []

        if use_correlations == 1:
            calc_rolling = calculate_rolling_correlation
        elif use_correlations == 2:
            calc_rolling = calculate_rolling_corr_diffs
        elif use_correlations == 3:
            calc_rolling = calculate_rolling_corr_diffs_with_rsi_vol

        # Loop through each feature and compute its rolling correlation with "long_momentum"
        for feature in factor_list:
            # This will calculate rolling correlations for each specified window.
            # Note: When feature is "long_momentum" itself, the rolling correlation will be 1.
            corr_df = calc_rolling(df_long, "fact_lag1_"+feature, 'fact_lag1_'+'long_momentum', windows)
            rolling_corr_dfs.append(corr_df)

        # Loop through each feature and compute its rolling correlation with portfolio "returns"
        for feature in factor_list:
            # This will calculate rolling correlations for each specified window.
            # Note: When feature is "long_momentum" itself, the rolling correlation will be 1.
            corr_df = calc_rolling(df_long, "fact_lag1_"+feature, 'lag1_'+'returns', windows)
            rolling_corr_dfs.append(corr_df)

        # Concatenate all the resulting rolling correlation features into a single DataFrame
        all_rolling_corr = pd.concat(rolling_corr_dfs, axis=1)
        df_long = pd.concat([df_long, all_rolling_corr], axis=1)

    # Fill missing values and sort the resulting dataframe
    df_long = df_long.fillna(0).sort_values(["date", "portfolio_id"])

    # Find the most recent date where 'returns' is not zero
    last_date = df_long.loc[df_long["lag1_returns"] != 0, "date"].max()

    # Subset the dataframe to include only rows from that date
    df_last_day = df_long[df_long["date"] == last_date]

    # Key columns for reshaping
    unique_dates = df_long["date"].unique() # TODO: Remove this if not needed
    unique_portfolios = df_long["portfolio_id"].unique()
    feature_columns = [col for col in df_long.columns if col not in ["returns", "date", "portfolio_id", "portfolio_value", "rank", "qid", "price"]]

    # Minimal leakage logging
    def detect_feature_leakage(df, target_column, feature_columns, threshold=0.95):
        """
        Detect feature leakage by fitting a linear model and checking R² scores.
        """
        leakage_features = {}

        # Fill NaN values with 0
        df_clean = df[feature_columns + [target_column]].copy()
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Min-Max Scaling
        scaler = MinMaxScaler()
        X = df_clean[feature_columns].fillna(0)
        X_scaled = scaler.fit_transform(X)

        y = df_clean[target_column].fillna(0)

        # Fit Linear Model
        model = LinearRegression()
        model.fit(X_scaled, y)
        y_pred = model.predict(X_scaled)

        r2 = r2_score(y, y_pred)
        print(r2)

        leakage_features = {
            "R² Score": r2,
            "Threshold": threshold}

        print("Results from Feature Leakage Test:", leakage_features)

    leakage_results = detect_feature_leakage(df_long, target_column="returns", feature_columns=feature_columns)

    #####################################################################
    #  5. RESHAPING -> 3D ARRAYS
    #####################################################################

    D = {}

    epsilon = 1e-8

    def get_preprocess_stock(data):
        "data is M * F"
        data = np.array(data, dtype = np.float32)
        a = np.zeros((3, data.shape[-1]))
        t = np.nan_to_num(data, nan = np.nan, neginf = 1e9)
        a[0, :] = np.nanmin(t, axis = 0)
        t = np.nan_to_num(data, nan = np.nan, posinf = -1e9)
        a[2, :] = np.nanmax(t, axis = 0)
        for i in range(data.shape[-1]):
            data[:,i] = np.nan_to_num(data[:,i], nan = np.nan, posinf = a[2,i], neginf = a[0,i])
            try:
                if (a[2,i] - a[0,i]) != 0:
                    data[:,i] = (data[:,i] - a[0,i]) / (a[2,i] - a[0,i])
                else:
                    data[:,i] = epsilon
            except:
                if i not in D.keys():
                    D[i] = 0
                D[i] += 1
                print(i)
                print(data[:,i])
        for i in range(data.shape[-1]):
            nan_value = 0.0 if np.nanmean(data[:,i]) == np.nan else np.nanmean(data[:,i])
            data[:,i] = np.nan_to_num(data[:,i], nan = nan_value)
            a[1, i] = nan_value
        return data, a

    def get_preprocess(data):
        A = []
        for i in range(data.shape[1]):
            data[:,i,:], a = get_preprocess_stock(data[:,i,:])
            A.append(a)
        return data, A

    def preprocess_stock(data, a):
        for i in range(data.shape[-1]):
            data[:,i] = np.nan_to_num(data[:,i], nan = a[1,i], posinf = a[2,i], neginf = a[0,i])
        for i in range(data.shape[0]):
            a[0,:] = np.minimum(a[0,:], data[i,:])
            a[2,:] = np.maximum(a[2,:], data[i,:])
            for j in range(data.shape[-1]):
                try:
                    if (a[2,j] - a[0,j]) != 0:
                        data[i,j] = (data[i,j] - a[0,j]) / (a[2,j] - a[0,j])
                    else:
                        data[i,j] = epsilon
                except:
                    print("!!!!!!\n\n")
                    print(i,j)
        return data

    def preprocess(data, A):
        for i in range(data.shape[1]):
            data[:,i,:] = preprocess_stock(data[:,i,:], A[i])
        return data

    # Set up the training dataframe
    n_assets = len(unique_portfolios)
    n_features = len(feature_columns)

    # Sort the unique dates to ensure they are in order
    sorted_dates = pd.to_datetime(np.sort(df_long.date.unique()))

    # Find the closest date to the holdout_start
    closest_holdout_index = np.abs(sorted_dates - holdout_start).argmin()

    # Calculate the starting index for the rolling window
    training_start_index = max(0, closest_holdout_index - rolling_train_length)

    # Find the corresponding training start date
    training_start = sorted_dates[training_start_index]

    # Filter the DataFrame based on the calculated training_start
    training_df = df_long[df_long.date >= training_start]

    # Step 1: Pivot the DataFrame so that features become columns
    df_pivoted = training_df[feature_columns + ["date", "portfolio_id"]].pivot(index='date', columns='portfolio_id')

    # Step 2: MultiIndex columns will now represent (portfolio_id, feature_name)
    df_pivoted = df_pivoted.sort_index(axis=1)  # Sort columns for consistency

    # Step 3: Convert to 3D NumPy array
    M = df_pivoted.shape[0]  # Number of time steps (dates)
    N = len(df_pivoted.columns.levels[1])  # Number of assets (portfolio IDs)
    F = len(df_pivoted.columns.levels[0])  # Number of features

    # Reshape the data to (M, N, F)
    feature_array_3d = df_pivoted.to_numpy().reshape(M, N, F)

    # Second array (returns)
    df_pivoted_returns = training_df.pivot(index='date', columns='portfolio_id', values='returns')
    df_pivoted_returns = df_pivoted_returns.sort_index(axis=1)

    # Convert to 2D numpy array (M days × N portfolios)
    returns_array_2d = df_pivoted_returns.to_numpy()
    
    print(f"Feature generation time:{(time.time() - feature_gen_start):.3f} s\n")
    print(f"Shape of the 3D feature array: {feature_array_3d.shape}")  # Should be (~, 19, 72)
    print(f"Shape of the 2D returns array: {returns_array_2d.shape}")  # Should be (~, 19)

    #####################################################################
    #  6. LISTFOLD MODEL + TRAINING
    #####################################################################

    import numpy as np
    import torch
    import pandas as pd
    from torch import nn, optim
    import torch.nn.functional as F
    from torch.autograd import Variable
    import copy

    import torch.optim as optim
    import torch.nn.functional as F
    from torch.amp import autocast, GradScaler

    # Set these for reproducible results and stable training
    def set_random_seeds(seed=42):
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        # For GPU determinism (slows down performance if True):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    #########################################################################
    # Custom Loss (Vectorized)
    #########################################################################
    class Closs_explained(nn.Module):
        def __init__(self):
            super(Closs_explained, self).__init__()

        def forward(self, f, num_stocks):
            """
            f: Tensor of shape (batch_size, n_assets)
            num_stocks: integer = n_assets
            """
            # s = half of the number of stocks
            if clamp_gradients == 1:
                f = torch.clamp(f, -10, 10)
            s = num_stocks // 2

            # ---------------------------------------------------------
            # 1) Use prefix sums to compute partial sums quickly.
            #    We'll pad at dim=1 by zero so cumsum indices align.
            #    prefix_f[i, j] = sum of f[i, 0..j-1]
            # ---------------------------------------------------------
            prefix_f = torch.cat([torch.zeros_like(f[:, :1]), f], dim=1).cumsum(dim=1)
            prefix_exp_f = torch.cat([torch.zeros_like(f[:, :1]), torch.exp(f)], dim=1).cumsum(dim=1)
            prefix_exp_neg_f = torch.cat([torch.zeros_like(f[:, :1]), torch.exp(-f)], dim=1).cumsum(dim=1)

            # ---------------------------------------------------------
            # 2) Base term: sum of second half minus sum of first half
            #    sum(f[:,s:]) - sum(f[:,:s]) via prefix sums:
            #    sum(f[:,s:]) = prefix_f[:, n_assets] - prefix_f[:, s]
            #    sum(f[:,:s]) = prefix_f[:, s] - prefix_f[:, 0]
            #    So difference = prefix_f[:, n_assets] - 2 * prefix_f[:, s]
            # ---------------------------------------------------------
            l = prefix_f[:, -1] - 2.0 * prefix_f[:, s]   # shape: (batch_size,)

            # ---------------------------------------------------------
            # 3) Accumulate the log(...) terms for i=0..(s-1)
            #    Using prefix sums again for partial sums of exp_f and exp_neg_f
            #    term1 = sum_{k=i}^{n_assets-i-1} exp_f
            #          = prefix_exp_f[:, n_assets-i] - prefix_exp_f[:, i]
            #    term2 = sum_{k=i}^{n_assets-i-1} exp_neg_f
            #          = prefix_exp_neg_f[:, n_assets-i] - prefix_exp_neg_f[:, i]
            #    We'll gather these in one vectorized pass.
            # ---------------------------------------------------------
            i_range = torch.arange(s, device=f.device)          # 0,1,2,...,s-1
            start_idx = i_range
            end_idx = (num_stocks - i_range)

            # For each batch row, gather from prefix sums:
            bsz = f.shape[0]
            start_expand = start_idx.unsqueeze(0).expand(bsz, s)
            end_expand = end_idx.unsqueeze(0).expand(bsz, s)

            term1 = torch.gather(prefix_exp_f, 1, end_expand) - torch.gather(prefix_exp_f, 1, start_expand)
            term2 = torch.gather(prefix_exp_neg_f, 1, end_expand) - torch.gather(prefix_exp_neg_f, 1, start_expand)

            # subtract_const = (num_stocks - 2*i)
            subtract_array = (num_stocks - 2.0 * i_range).unsqueeze(0).expand(bsz, s)

            # # Carefully log(...) everything in one shot
            # inside_log = term1 * term2 - subtract_array
            # logs_sum = torch.log(inside_log).sum(dim=1)  # sum over i dimension

            epsilon = 1e-8  # small constant to avoid log(0)

            # Compute inside_log as before
            inside_log = term1 * term2 - subtract_array

            # Clamp inside_log to ensure it's at least epsilon
            inside_log = torch.clamp(inside_log, min=epsilon)

            logs_sum = torch.log(inside_log).sum(dim=1)

            # ---------------------------------------------------------
            # 4) Add logs_sum to the base term and then mean over batch
            # ---------------------------------------------------------
            l = l + logs_sum   # shape: (batch_size,)
            return l.mean()    # final scalar

    #########################################################################
    # Models (with different activation functions)
    #########################################################################

    def weights_init(m, seed=42):
        """Simple weight initialization with controlled random seed."""
        torch.manual_seed(seed)
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            m.weight.data.normal_(mean=0.0, std=0.05)
            m.bias.data.fill_(0.05)

    def mish(x):
        return x * torch.tanh(F.softplus(x))

    def swish(x):
        return x * torch.sigmoid(x)


    # Base ReLU Model
    class CMLE(nn.Module):
        def __init__(self, n_features, seed=42):
            super(CMLE, self).__init__()
            self.seed = seed
            self.n_features = n_features
            self.linear1 = nn.Linear(self.n_features, self.n_features * 4)
            self.linear2 = nn.Linear(self.n_features * 4, self.n_features * 2)
            self.linear3 = nn.Linear(self.n_features * 2, self.n_features // 2)
            self.linear4 = nn.Linear(self.n_features // 2, 1)
            self.apply(lambda m: weights_init(m, seed))

        def forward(self, x):
            # x: (batch_size, n_assets, n_features)
            x = F.relu(self.linear1(x))
            x = F.relu(self.linear2(x))
            x = F.relu(self.linear3(x))
            x = F.relu(self.linear4(x))
            # Reshape => (batch_size, n_assets)
            return x.view(x.shape[0], x.shape[1])


    # Leaky ReLU Model
    class CMLE_leaky(nn.Module):
        def __init__(self, n_features, seed=42):
            super(CMLE_leaky, self).__init__()
            self.seed = seed
            self.n_features = n_features
            self.linear1 = nn.Linear(self.n_features, self.n_features * 4)
            self.linear2 = nn.Linear(self.n_features * 4, self.n_features * 2)
            self.linear3 = nn.Linear(self.n_features * 2, self.n_features // 2)
            self.linear4 = nn.Linear(self.n_features // 2, 1)
            self.apply(lambda m: weights_init(m, seed))

        def forward(self, x):
            x = F.leaky_relu(self.linear1(x), negative_slope=0.01)
            x = F.leaky_relu(self.linear2(x), negative_slope=0.01)
            x = F.leaky_relu(self.linear3(x), negative_slope=0.01)
            x = F.leaky_relu(self.linear4(x), negative_slope=0.01)
            return x.view(x.shape[0], x.shape[1])


    # Mish Model
    class MISH_CMLE(nn.Module):
        def __init__(self, n_features, seed=42):
            super(MISH_CMLE, self).__init__()
            self.seed = seed
            self.n_features = n_features
            self.linear1 = nn.Linear(self.n_features, self.n_features * 4)
            self.linear2 = nn.Linear(self.n_features * 4, self.n_features * 2)
            self.linear3 = nn.Linear(self.n_features * 2, self.n_features // 2)
            self.linear4 = nn.Linear(self.n_features // 2, 1)
            self.apply(lambda m: weights_init(m, seed))

        def forward(self, x):
            x = mish(self.linear1(x))
            x = mish(self.linear2(x))
            x = mish(self.linear3(x))
            x = mish(self.linear4(x))
            return x.view(x.shape[0], x.shape[1])


    # Swish Model
    class SWISH_CMLE(nn.Module):
        def __init__(self, n_features, seed=42):
            super(SWISH_CMLE, self).__init__()
            self.seed = seed
            self.n_features = n_features
            self.linear1 = nn.Linear(self.n_features, self.n_features * 4)
            self.linear2 = nn.Linear(self.n_features * 4, self.n_features * 2)
            self.linear3 = nn.Linear(self.n_features * 2, self.n_features // 2)
            self.linear4 = nn.Linear(self.n_features // 2, 1)
            self.apply(lambda m: weights_init(m, seed))

        def forward(self, x):
            x = swish(self.linear1(x))
            x = swish(self.linear2(x))
            x = swish(self.linear3(x))
            x = swish(self.linear4(x))
            return x.view(x.shape[0], x.shape[1])
        

    class ResMarket_CMLE(nn.Module):
        def __init__(self, n_features, seed=42):
            super(ResMarket_CMLE, self).__init__()
            self.seed = seed
            torch.manual_seed(seed)

            # Shared layers for each asset
            self.linear1 = nn.Linear(n_features, 128)
            self.layer_norm1 = nn.LayerNorm(128)
            self.linear2 = nn.Linear(128, 64)
            self.skip1 = nn.Linear(128, 64)  # Skip connection
            self.layer_norm2 = nn.LayerNorm(64)
            self.linear3 = nn.Linear(64, 1)

        def forward(self, x):
            # x: (batch_size, n_assets, n_features_per_asset)
            batch_size, n_assets, _ = x.shape

            # Apply shared layers to each asset
            x = swish(self.layer_norm1(self.linear1(x)))  # (batch_size, n_assets, 128)
            x2 = self.linear2(x)  # (batch_size, n_assets, 64)
            skip = self.skip1(x)  # (batch_size, n_assets, 64)
            x = swish(x2 + skip)  # Residual connection
            x = self.layer_norm2(x)  # (batch_size, n_assets, 64)
            x = self.linear3(x)  # (batch_size, n_assets, 1)

            # Squeeze the last dimension to get scores
            x = x.squeeze(-1)  # (batch_size, n_assets)
            return x



    #########################################################################
    # Utility functions
    #########################################################################

    def rank_based_conformal_calibrate(
        true_ranks: np.ndarray,
        pred_ranks: np.ndarray,
        alpha: float = 0.1,
        mode: str = "single_best"
    ):
        """
        Compute the rank-based nonconformity distribution from 'calibration' data and
        return the threshold that ensures coverage 1 - alpha.

        Parameters
        ----------
        true_ranks : np.ndarray, shape (calib_days, n_assets)
            The "true" rank of each asset on each calibration day.
            Lower = better (0 = best, 1 = second best, etc.).
        pred_ranks : np.ndarray, shape (calib_days, n_assets)
            The "predicted" rank for each asset on each calibration day
            from your model. Lower = better predicted.
        alpha : float
            Significance level: e.g. alpha=0.1 => coverage=90%.
        mode : {"single_best", "entire_top_k", "single_worst", ...}
            - "single_best": ensures the single truly best asset is contained in
            your top picks at least (1 - alpha) fraction of the time.
            - "single_worst": ensures the single truly worst asset is contained in
            your bottom picks at least (1 - alpha) fraction of the time.
            - "entire_top_k": can be used if you want coverage for all top-k truly best.

        Returns
        -------
        threshold : int
            The conformal rank threshold. If threshold = 7, that means
            you generally need to pick everyone with predicted_rank <= 7
            to achieve coverage. If threshold=8, you need to pick up to 8, etc.
        """

        # 1) Identify the "nonconformity" each day. For "single_best", we find
        #    which asset is truly best => the one that has the lowest 'true_ranks' value.
        #    Then we see what predicted_rank that asset had => pred_ranks[day, best_idx].
        #    Collect those for all calibration days => that is your distribution.

        n_days, n_assets = true_ranks.shape
        nonconformity_vals = []

        if mode in ["single_best", "entire_top_k"]:  # We want the truly best
            for day in range(n_days):
                # truly best asset = where true_ranks is min
                # (lowest = best, if 0-based)
                best_idx = np.argmin(true_ranks[day, :])
                # predicted rank of that truly best asset
                pred_rank_of_best = pred_ranks[day, best_idx]  # 0-based rank
                nonconformity_vals.append(pred_rank_of_best)

        elif mode == "single_worst":
            # If we want the single truly worst asset in the "bottom picks"
            # the truly worst is the highest rank =>  e.g. argmax
            for day in range(n_days):
                worst_idx = np.argmax(true_ranks[day, :])
                pred_rank_of_worst = pred_ranks[day, worst_idx]
                nonconformity_vals.append(pred_rank_of_worst)

        # If you want "entire_top_k", you'd do a slightly more advanced approach:
        #   For example, find the asset that was the k-th best in the true rank,
        #   then note where that asset appeared in the predicted rank.
        #   That is a bit more specialized; code omitted for brevity.
        #   You can replicate a similar pattern.

        nonconformity_vals = np.array(nonconformity_vals)
        nonconformity_vals.sort()
        n_cal = len(nonconformity_vals)

        # 2) The rank threshold for coverage = the (1-alpha) quantile
        #    => index = ceil((1-alpha)*n_cal)-1 (but clamp >= 0)
        idx = int(np.ceil((1 - alpha) * n_cal)) - 1
        idx = max(idx, 0)

        threshold = int(nonconformity_vals[idx])
        return threshold

    def rank_based_conformal_apply(
        pred_ranks_test: np.ndarray,
        threshold: int,
        default_k: int = 7,
        side: str = "top"
    ):
        """
        Given the predicted ranks for the test set (one or many days),
        decide how many you must pick to ensure the coverage from 'threshold'.
        For example, if threshold=8, you may pick top-8 (for side="top").
        For side="bottom", you pick bottom-8, etc.

        Parameters
        ----------
        pred_ranks_test : np.ndarray, shape (test_days, n_assets)
            Predicted ranks for the holdout/test chunk.
            e.g. pred_ranks_test[day, asset] = 0-based rank of that asset.
        threshold : int
            The conformal rank threshold from rank_based_conformal_calibrate.
        default_k : int
            The baseline number you want to pick (like 7). If threshold says 8,
            we will pick 8. If threshold is less than 7, do you want to pick only 6?
            Typically we do max(default_k, threshold+1) or something similar.
        side : {"top", "bottom"}
            - "top": pick everything with predicted_rank <= pick_size-1.
            - "bottom": pick from the other end.

        Returns
        -------
        picks_array : list of lists
            picks_array[day_idx] = list of asset indices included for that day.
        pick_size_used : int
            The final number of assets actually included each day
            (or it could be a list if you do day-by-day logic).
        """

        test_days, n_assets = pred_ranks_test.shape

        # If threshold = 7 => we pick rank <= 7. But is that 0..7 (8 total)?
        # Usually if rank=7 means the best 8.
        # So let's define an offset.
        # If we want to interpret threshold=7 => we pick top 8, do threshold + 1.
        # You can tweak as desired.
        pick_size = max(default_k, threshold + 1)

        picks_array = []

        for day in range(test_days):
            if side == "top":
                # 1) find assets whose predicted rank <= pick_size-1
                # (lowest rank means best predicted)
                day_ranks = pred_ranks_test[day, :]
                chosen = np.where(day_ranks < pick_size)[0]
                picks_array.append(chosen.tolist())

            elif side == "bottom":
                # we do the symmetrical approach:
                # if threshold=7 => pick the 8 worst assets
                # that means rank >= n_assets - pick_size
                day_ranks = pred_ranks_test[day, :]
                # we define "worst" as the largest ranks
                # so we pick everything with rank >= n_assets - pick_size
                chosen = np.where(day_ranks >= (n_assets - pick_size))[0]
                picks_array.append(chosen.tolist())

        return picks_array, pick_size

    def return_rank(a):
        """
        Returns the rank ordering of array 'a' (ascending).
        For example, if a = [4, 1, 9], then rank = [1, 0, 2].
        """
        a = -1.0 * a
        order = a.argsort()
        return order.argsort()

    def get_predicted_ranks(scores):
        """Convert model scores to ranks (one row at a time)."""
        return np.array([return_rank(s) for s in scores])

    def random_batch_precomputed_gpu(features_sorted_gpu, batch_size, seed=None):
        """
        Randomly select 'batch_size' days from 'features_sorted_gpu' (on GPU).
        features_sorted_gpu: Tensor of shape (num_days, n_assets, n_features)
        """
        if seed is not None:
            torch.manual_seed(seed)
        num_days = features_sorted_gpu.shape[0]
        indices = torch.randint(low=0, high=num_days, size=(batch_size,),
                                device=features_sorted_gpu.device)
        return features_sorted_gpu[indices]
    

    if custom_tie_breaks == 1:

        ############################################################################
        # 1) Define your custom tie-break order
        ############################################################################
        new_default_order = [0,20,19,18,17,16,15,13,12,14,21,11,9,8,7,5,4,3,6,2,1,10,22]
        tie_break_rank = { idx: pos for pos, idx in enumerate(new_default_order) }

        ############################################################################
        # 2) The replacement for `return_rank(a)`
        ############################################################################
        def return_rank(a):
            """
            Returns the rank ordering of array `a` in descending order,
            breaking ties using the custom `new_default_order`.

            rank[i] = 0 means the element at index i is the "best" (highest score);
            rank[i] = 1 means second place, and so on.

            Example:
            If all elements of `a` are equal, the rank of index i is
            tie_break_rank[i] in ascending order.
            """
            # Sort all indices from "largest score" to "smallest score".
            # If two scores tie, compare tie_break_rank[i].
            indices = sorted(range(len(a)), key=lambda i: (-a[i], tie_break_rank[i]))
            
            # Build the rank array: rank[i] = the position of i in the sorted list
            rank = np.empty(len(a), dtype=int)
            for r, i in enumerate(indices):
                rank[i] = r
            return rank

        ############################################################################
        # 3) The replacement for `get_predicted_ranks(scores)`
        ############################################################################
        def get_predicted_ranks(scores):
            """
            Convert model scores to ranks (one row at a time).
            """
            return np.array([return_rank(row) for row in scores])


    #########################################################################
    # 7. BACKTEST & TRAIN LOOP
    #########################################################################

    def backtest(model, features, returns, dates, top_k=8, avg_loss=0,
                lookback_loss=0, short_type='bottom'):
        """
        Backtest the model on the given features & returns data.
        Args:
            model: trained model (on GPU)
            features: np.array, shape (num_days, n_assets, n_features)
            returns:  np.array, shape (num_days, n_assets)
            dates:    np.array or list of date-strings
            top_k:    how many positions to go long/short
            short_type: 'bottom' or 'average'
        """
        device = next(model.parameters()).device

        # Move features to torch tensor on device, get model scores
        features_tensor = torch.from_numpy(features).float().to(device)
        with torch.no_grad():
            scores_tensor = model(features_tensor)
        scores = scores_tensor.cpu().numpy()  # shape: (num_days, n_assets)

        results = []
        positions_long = []
        positions_short = []
        returns_long = []
        returns_short = []
        predicted_ranks = []
        dates_list = []

        n_assets = returns.shape[1]
        for i in range(len(scores)):
            rank = return_rank(scores[i])  # ascending rank
            rank2ind = np.zeros(len(rank), dtype=int)
            for j in range(len(rank)):
                rank2ind[rank[j]] = j

            predicted_ranks.append(rank)
            dates_list.append(dates[i])

            # Calculate returns for that day
            total_return = 0.0
            weights = np.ones(top_k) / top_k

            # Long positions
            long_pos = []
            long_ret = []
            for j in range(top_k):
                idx = rank2ind[j]  # top_k (lowest) ranks => best because we multiplied scores by -1
                total_return += weights[j] * returns[i][idx]
                long_pos.append(idx)
                long_ret.append(returns[i][idx])

            # Short positions
            short_pos = []
            short_ret = []
            if short_type == 'bottom':
                for j in range(top_k):
                    idx = rank2ind[(n_assets - 1) - j]
                    total_return -= weights[j] * returns[i][idx]
                    short_pos.append(idx)
                    short_ret.append(returns[i][idx])
            elif short_type == 'average':
                # short vs. entire average market
                market_return = np.mean(returns[i])
                total_return -= market_return

            results.append(total_return)
            positions_long.append(long_pos)
            positions_short.append(short_pos)
            returns_long.append(long_ret)
            returns_short.append(short_ret)

        return (np.array(results),
                np.array(positions_long),
                np.array(positions_short),
                np.array(returns_long),
                np.array(returns_short),
                np.array(predicted_ranks),
                top_k,
                np.array(dates_list),
                returns,  # actual returns
                scores,   # raw scores
                avg_loss,
                lookback_loss)


    def precompute_sorted_data(features, ranks):
        """
        Pre-sort the features for each day based on its rank ordering.
        features: (num_days, n_assets, n_features)
        ranks:    (num_days, n_assets)
        """
        features_sorted = np.zeros_like(features)
        num_days = len(features)

        for day_index in range(num_days):
            day_rank = return_rank(ranks[day_index])  # ascending rank
            # day_rank tells you how to permute day_index's assets
            for j, stock_idx in enumerate(day_rank):
                # place features[day_index, j, :] in new location
                features_sorted[day_index, stock_idx, :] = features[day_index, j, :]
        return features_sorted

    def train__evaluate_listfold(features,
                                ranks,
                                epochs,
                                test_features,
                                test_returns,
                                test_dates,
                                seed=42):
        """
        Training loop with:
        - Pre-sorting data,
        - Automatic Mixed Precision,
        - Vectorized custom loss,
        - GPU batch sampling.
        Returns backtest results for top_k in range(1, n_k).
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_random_seeds(seed)

        # 1) Precompute & move data to GPU
        features_sorted = precompute_sorted_data(features, ranks)
        features_sorted_gpu = torch.from_numpy(features_sorted).float().to(device)

        # 2) Build model, loss, optimizer
        if act_func == 'leaky':
            model = CMLE_leaky(n_features=n_features, seed=seed).to(device)
        elif act_func == 'mish':
            model = MISH_CMLE(n_features=n_features, seed=seed).to(device)
        elif act_func == 'swish':
            model = SWISH_CMLE(n_features=n_features, seed=seed).to(device)
        elif act_func == 'res_market':
            model = ResMarket_CMLE(n_features=n_features, seed=seed).to(device)
        else:
            model = CMLE(n_features=n_features, seed=seed).to(device)
        # model = torch.compile(model) #, mode='reduce-overhead') 
        loss_fn = Closs_explained()
        opt = optim.Adam(model.parameters(), lr=learning_rate)

        print("Model built and moved to device:", device)
        print(f"Using activation: {act_func}")

        # Automatic Mixed Precision
        scaler = GradScaler(enabled=True) #, device_type='cuda')

        # For tracking time
        time_data_total = 0.0
        time_fwd_total = 0.0
        time_loss_total = 0.0
        time_bwd_total = 0.0

        running_loss = []
        start_all = time.time()

        for itr in range(epochs):
            # -- DATA (random batch) --
            t0 = time.time()
            batch_x = random_batch_precomputed_gpu(features_sorted_gpu, batch_size, seed=seed + itr)
            t1 = time.time()
            time_data_total += (t1 - t0)

            # -- Forward + Loss under autocast --
            model.train()
            with autocast(device_type="cuda"):
                t2 = time.time()
                scores = model(batch_x)  # shape: (batch_size, n_assets)
                t3 = time.time()
                time_fwd_total += (t3 - t2)

                # n_assets constant as tensor
                n_assets_t = torch.tensor(n_assets, device=device, requires_grad=False)
                t4 = time.time()
                l = loss_fn(scores, n_assets_t)
                t5 = time.time()
                time_loss_total += (t5 - t4)

            # -- Backprop + Optim Step (scaled) --
            t6 = time.time()
            opt.zero_grad(set_to_none=True)
            scaler.scale(l).backward()

            # Unscale gradients before clipping (this step is important when using GradScaler)
            scaler.unscale_(opt)

            # Clip gradients to a maximum norm of 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
            t7 = time.time()
            time_bwd_total += (t7 - t6)

            running_loss.append(float(l))
        total_time = time.time() - start_all
        avg_loss = np.mean(running_loss) if len(running_loss) > 0 else 0.0
        print(f"Finished {epochs} epochs, final avg loss: {avg_loss:.4f}, abs: {np.mean(np.abs(running_loss))}")
        # print(running_loss)

        print(f"Total training time: {total_time:.3f} s")
        other_time = total_time - (time_data_total + time_fwd_total + time_loss_total + time_bwd_total)
        print("Time breakdown:")
        print(f"  Data sampling time:   {time_data_total:.3f} s")
        print(f"  Forward pass time:    {time_fwd_total:.3f} s")
        print(f"  Loss computation:     {time_loss_total:.3f} s")
        print(f"  Backward + step time: {time_bwd_total:.3f} s")
        print(f"  Other overhead:       {other_time:.3f} s\n")

        # 3) Evaluate / Backtest across multiple top_k
        all_backtests = []
        for k in range(1, n_k):
            # We pass the final average loss as an example argument
            all_backtests.append(
                backtest(model,
                        test_features,
                        test_returns,
                        test_dates,
                        top_k=k,
                        avg_loss=avg_loss,
                        lookback_loss=np.mean(running_loss[-30:]) if len(running_loss) >= 30 else avg_loss,
                        short_type='bottom')
            )
        return all_backtests, model
    
    def train__evaluate_listfold_live(features,
                                ranks,
                                epochs,
                                seed=42):
        """
        Training loop with:
        - Pre-sorting data,
        - Automatic Mixed Precision,
        - Vectorized custom loss,
        - GPU batch sampling.
        Returns backtest results for top_k in range(1, n_k).
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_random_seeds(seed)

        # 1) Precompute & move data to GPU
        features_sorted = precompute_sorted_data(features, ranks)
        features_sorted_gpu = torch.from_numpy(features_sorted).float().to(device)

        # 2) Build model, loss, optimizer
        if act_func == 'leaky':
            model = CMLE_leaky(n_features=n_features, seed=seed).to(device)
        elif act_func == 'mish':
            model = MISH_CMLE(n_features=n_features, seed=seed).to(device)
        elif act_func == 'swish':
            model = SWISH_CMLE(n_features=n_features, seed=seed).to(device)
        elif act_func == 'res_market':
            model = ResMarket_CMLE(n_features=n_features, seed=seed).to(device)
        else:
            model = CMLE(n_features=n_features, seed=seed).to(device)
        # model = torch.compile(model) #, mode='reduce-overhead')
        loss_fn = Closs_explained()
        opt = optim.Adam(model.parameters(), lr=learning_rate)

        print("Model built and moved to device:", device)
        print(f"Using activation: {act_func}")

        # Automatic Mixed Precision
        scaler = GradScaler(enabled=True) #, device_type='cuda')

        # For tracking time
        time_data_total = 0.0
        time_fwd_total = 0.0
        time_loss_total = 0.0
        time_bwd_total = 0.0

        running_loss = []
        start_all = time.time()

        for itr in range(epochs):
            # -- DATA (random batch) --
            t0 = time.time()
            batch_x = random_batch_precomputed_gpu(features_sorted_gpu, batch_size, seed=seed + itr)
            t1 = time.time()
            time_data_total += (t1 - t0)

            # -- Forward + Loss under autocast --
            model.train()
            with autocast(device_type="cuda"):
                t2 = time.time()
                scores = model(batch_x)  # shape: (batch_size, n_assets)
                t3 = time.time()
                time_fwd_total += (t3 - t2)

                # n_assets constant as tensor
                n_assets_t = torch.tensor(n_assets, device=device, requires_grad=False)
                t4 = time.time()
                l = loss_fn(scores, n_assets_t)
                t5 = time.time()
                time_loss_total += (t5 - t4)

            # -- Backprop + Optim Step (scaled) --
            t6 = time.time()
            opt.zero_grad(set_to_none=True)
            scaler.scale(l).backward()

            # Unscale gradients before clipping (this step is important when using GradScaler)
            scaler.unscale_(opt)

            # Clip gradients to a maximum norm of 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
            t7 = time.time()
            time_bwd_total += (t7 - t6)

            running_loss.append(float(l))
        total_time = time.time() - start_all
        avg_loss = np.mean(running_loss) if len(running_loss) > 0 else 0.0
        print(f"Finished {epochs} epochs, final avg loss: {avg_loss:.4f}, abs: {np.mean(np.abs(running_loss))}")
        # print(running_loss)

        print(f"Total training time: {total_time:.3f} s")
        other_time = total_time - (time_data_total + time_fwd_total + time_loss_total + time_bwd_total)
        print("Time breakdown:")
        print(f"  Data sampling time:   {time_data_total:.3f} s")
        print(f"  Forward pass time:    {time_fwd_total:.3f} s")
        print(f"  Loss computation:     {time_loss_total:.3f} s")
        print(f"  Backward + step time: {time_bwd_total:.3f} s")
        print(f"  Other overhead:       {other_time:.3f} s\n")
        return model

    # =============================================================================
    # CONFIGURATIONS
    # =============================================================================

    PROJECT_ID = "issachar-feature-library"
    REGION = "us-central1"
    STAGING_BUCKET = "gs://qjg-test"
    DESTINATION_DATASET = "qjg_model_runs"

    # Initialize GCS and BigQuery clients once
    storage_client = storage.Client(project=PROJECT_ID)
    bigquery_client = bigquery.Client(project=PROJECT_ID)

    def sanitize_and_convert_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Sanitize column names to meet BigQuery requirements and convert
        datetime64[ns, UTC] columns to datetime64[ns] for compatibility.
        """
        # Sanitize column names
        df = df.rename(columns=lambda col: (
            col.replace(" ", "_")
            .replace("%", "pct")
            .replace("(", "")
            .replace(")", "")
            .replace("-", "_")
            .replace("/", "_")
            .lower()
        ))
        # Convert datetime64[ns, UTC] to datetime64[ns]
        for col in df.select_dtypes(include=["datetime64[ns, UTC]"]).columns:
            df[col] = df[col].dt.tz_localize(None)

        return df

    def append_to_bigquery(
        df: pd.DataFrame,
        dataset_id: str,
        table_name: str,
        project_id: str = PROJECT_ID
    ) -> None:
        """
        Append data to a BigQuery table, automatically creating the table if it
        doesn't exist and inferring the schema from the DataFrame.
        """
        table_id = f"{project_id}.{dataset_id}.{table_name}"

        # Convert datetime64[ns] columns to timestamp-friendly strings
        for col in df.select_dtypes(include=["datetime64[ns]"]).columns:
            df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')

        job_config = bigquery.LoadJobConfig(
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND
        )

        try:
            load_job = bigquery_client.load_table_from_dataframe(
                df, table_id, job_config=job_config
            )
            load_job.result()
            logging.info(f"Appended data to table {table_id} successfully.")
        except Exception as e:
            logging.error(f"Failed to append data to {table_id}: {str(e)}")
            raise

    def create_or_replace_bigquery_table(
        df: pd.DataFrame,
        dataset_id: str,
        table_name: str,
        project_id: str = PROJECT_ID
    ) -> None:
        """
        Create or replace a BigQuery table using the data from a DataFrame.
        The table schema is automatically inferred from the DataFrame,
        and any existing table will be replaced.
        """
        table_id = f"{project_id}.{dataset_id}.{table_name}"

        # Convert datetime64[ns] columns to a format that BigQuery can interpret as a timestamp.
        for col in df.select_dtypes(include=["datetime64[ns]"]).columns:
            df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')

        # Configure the job to create or replace the table.
        job_config = bigquery.LoadJobConfig(
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE
        )

        try:
            load_job = bigquery_client.load_table_from_dataframe(
                df, table_id, job_config=job_config
            )
            load_job.result()  # Wait for the load job to complete.
            logging.info(f"Created or replaced table {table_id} successfully.")
        except Exception as e:
            logging.error(f"Failed to create or replace table {table_id}: {str(e)}")
            raise


    # =============================================================================
    # HELPER FUNCTIONS
    # =============================================================================
    # TODO: Add the following notes from Josh: Is final input an outlier? | Is the volatility higher currently than historical? Z-Score? | Correlation structure changing ? How much divergence is there? | Look at 63/252 days vs bulk?
    def compute_nn_data_quality_metrics(
        df, 
        lookback=2000, 
        std_threshold=4.0,
        perc_diff_thresh=2
    ):
        
        metrics_list = []
        
        # Group the DataFrame by portfolio_id (each ETF)
        percent_changes = []
        for pid, group in df.groupby('portfolio_id'):
            # Select the most recent `lookback` rows for the current portfolio
            group_recent = group.tail(lookback)
            
            # Compute metrics for each numeric column in the subset
            for col in group_recent.columns:
                # Skip the portfolio_id column and any non-numeric columns
                if col in ['portfolio_id', 'qid']:
                    continue
                s = group_recent[col]
                if pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_string_dtype(s):
                    continue
                
                # Drop NA values to avoid issues in calculations
                s = s.dropna()
                s = s.replace([float('inf'), -float('inf')], 0) 
                s_count = len(s)
                s = s[s != 0]  # Exclude zero values for outlier detection
                if s.empty:
                    continue
                
                # ===== Check if the latest value is an outlier =====
                # Compare the latest input against historical values (all but the latest)
                if len(s) > 1:
                    latest_value = s.iloc[-1]
                    historical_data = s.iloc[:-1]
                    hist_min = historical_data.min()
                    hist_max = historical_data.max()

                    # Calculate percentage difference from historical min
                    pct_diff_from_min = ((latest_value - hist_min) / (hist_max-hist_min)) * 100 if latest_value < hist_min else 0

                    # Calculate percentage difference from historical absolute max
                    pct_diff_from_max = ((latest_value - hist_max) / (hist_max-hist_min)) * 100 if latest_value > hist_max else 0

                    percent_changes.append({
                        'portfolio_id': pid, 'feature': col,
                        'latest_value': latest_value, 'pct_diff_from_min': pct_diff_from_min,
                        'pct_diff_from_max': pct_diff_from_max,})

                    if latest_value > hist_max:
                        pct_diff = ((latest_value / hist_max) - 1) * 100
                        if pct_diff > perc_diff_thresh:
                            print("------------------------------------------------------------------------------------------------------------------------")
                            print(
                                f"[ALERT] Latest value {latest_value:.4f} for portfolio '{pid}', "
                                f"feature '{col}' is {pct_diff:.2f}% greater than historical max {hist_max:.4f}."
                            )
                            print("------------------------------------------------------------------------------------------------------------------------")
                    elif latest_value < hist_min:
                        pct_diff = ((hist_min / latest_value) - 1) * 100
                        if pct_diff > perc_diff_thresh:
                            print("------------------------------------------------------------------------------------------------------------------------")
                            print(
                                f"[ALERT] Latest value {latest_value:.4f} for portfolio '{pid}', "
                                f"feature '{col}' is {pct_diff:.2f}% below historical min {hist_min:.4f}."
                            )
                            print("------------------------------------------------------------------------------------------------------------------------")

                # ========== Basic Statistics ==========
                q25 = s.quantile(0.25)
                q75 = s.quantile(0.75)
                iqr = q75 - q25
                q01 = s.quantile(0.01)
                q99 = s.quantile(0.99)
                mean_val = s.mean()
                std_val = s.std()  # sample-based standard deviation
                cv = std_val / mean_val if mean_val != 0 else np.nan
                
                data_range = s.max() - s.min()
                mad = np.mean(np.abs(s - mean_val))  # mean absolute deviation wrt mean
                # For the robust approach, we'll use the median-based MAD:
                median_val = s.median()
                mad_median = np.median(np.abs(s - median_val))
                
                # ========== IQR Outliers ==========
                lower_bound = q25 - 1.5 * iqr
                upper_bound = q75 + 1.5 * iqr
                iqr_outliers = (s < lower_bound) | (s > upper_bound)
                iqr_outliers_count = iqr_outliers.sum()
                iqr_outliers_fraction = iqr_outliers_count / len(s)
                
                # ========== Standard-Dev Outliers ==========
                if (std_val > 0) and (std_val < 1.5):
                    std_outliers = (np.abs(s - mean_val) > std_threshold * std_val)
                    std_outliers_count = std_outliers.sum()
                    std_outliers_fraction = std_outliers_count / len(s)
                else:
                    # If std_val is 0 or NaN, no std-based outliers
                    std_outliers_count = 0
                    std_outliers_fraction = 0.0
                
                # ========== Print alerts if outliers found ==========
                if std_outliers_count > 0:
                    # print(
                    #     f"[STD ALERT] {std_outliers_count} / {len(s)} data points for "
                    #     f"portfolio '{pid}', feature '{col}' exceed {std_threshold} std devs."
                    # )
                    continue

                # Collect metrics for this feature
                metrics_list.append({
                    'portfolio_id': pid,
                    'feature': col,
                    'count': s_count,
                    'mean': mean_val,
                    'std': std_val,
                    'median': median_val,
                    'mad_median': mad_median,
                    'min': s.min(),
                    '1%': q01,
                    '25%': q25,
                    '75%': q75,
                    '99%': q99,
                    'max': s.max(),
                    'IQR': iqr,
                    'range': data_range,
                    'MAD_mean': mad,  # mean-based absolute deviation
                    'skewness': s.skew(),
                    'kurtosis': s.kurtosis(),
                    'coef_variation': cv,
                    'perc_missing': (group_recent[col].isna().sum()) / lookback,
                    'perc_zeros': (s == 0).sum() / len(s),
                    'perc_unique': s.nunique() / len(s),
                    'perc_outliers_IQR': iqr_outliers_fraction,
                    'outliers_std_count': std_outliers_count,
                    'outliers_std_fraction': std_outliers_fraction
                })
        
        # Save percent changes to BigQuery
        percent_changes_df = pd.DataFrame(percent_changes)
        percent_changes_df = sanitize_and_convert_columns(percent_changes_df)
        percent_changes_df['uuid'] = uuid
        percent_changes_df['run_date'] = datetime.now()
        append_to_bigquery(percent_changes_df, DESTINATION_DATASET, f"live_percent_changes_{table_suffix}")

        # Create a DataFrame where each row corresponds to one feature of one ETF
        df_metrics = pd.DataFrame(metrics_list)
        return df_metrics


    #####################################################################
    #  8. ROLLING TRAIN/TEST
    #####################################################################

    # Define lists to collect WW and SHAP analysis results
    ww_results_list = []
    shap_results_list = []

    # Main training loop
    m = feature_array_3d
    returns = returns_array_2d
    dates = df_pivoted_returns.index  # Get dates from your pivoted DataFrame
    uuid = "model=Listfold_" + "__".join(f"{key}={value}" for key, value in args.items())

    all_results = []
    w = 0
    porfolio_names = list(np.sort(df_long.portfolio_id.unique()))

    # Conformal prediction variables
    alpha_list = [0.01, 0.05, 0.1, 0.2, 0.9]  # Adjust as needed
    mode_top = "single_best"
    mode_bottom = "single_worst"
    conformal_results = []

    pd.set_option('display.max_columns', 500)
    pd.set_option('display.expand_frame_repr', False)  # Prevent line wrapping of columns

    print("End of `df_long` DataFrame")
    print(tabulate(df_long.tail(1).T, headers='keys', tablefmt='psql'))

    if live_next_day == 1:
        # --------------------------------------------------
        # 1) TRAIN ON RECENT HISTORY
        # --------------------------------------------------
        # Get last rolling_train_length days for training
        # shape of m: (total_days, n_assets, n_features)
        train = copy.deepcopy(m[-(rolling_train_length+1):-1, :, :])
        train, a = get_preprocess(train)

        # Corresponding returns for the training window
        ranks_train = returns[-(rolling_train_length+1):-1]

        # --------------------------------------------------
        # 2) BUILD & TRAIN MODEL (ONE PASS)
        # --------------------------------------------------
        model = train__evaluate_listfold_live(train, ranks_train, epochs)
        
        # --------------------------------------------------
        # 3) PREPARE THE "LIVE" DATA (MOST RECENT DAY)
        # --------------------------------------------------
        test = copy.deepcopy(m[-1:, :, :])  # the very last slice
        test = preprocess(test, a)

        device = next(model.parameters()).device
        features_tensor = torch.from_numpy(test).float().to(device)

        with torch.no_grad():
            scores_tensor = model(features_tensor)  # shape: (1, n_assets)
        scores = scores_tensor.cpu().numpy()  # convert to NumPy

        # Convert model scores into a rank ordering
        # shape(s) -> (n_assets,)
        predicted_ranks = []
        for i in range(len(scores)):
            rank = return_rank(scores[i])  # ascending rank => 0 = best
            predicted_ranks.append(rank)

        print("LIVE SCORING OUTPUTS:")
        print("Scores:", scores)
        print("Predicted Ranks:", predicted_ranks)


        # --------------------------------------------------
        # 4) PREPARE A SMALL DATAFRAME TO SAVE
        # --------------------------------------------------
        #   We'll store each portfolio's predicted rank & raw score for the live day,
        #   plus any other metadata from the run you’d like (e.g., run_date, any args).
        # --------------------------------------------------

        # The “live” day’s date is presumably the last index in df_pivoted_returns
        # or equivalently the last day in your data pipeline.
        live_date = df_pivoted_returns.index[-1]  # Pandas Timestamp

        run_time = datetime.now()

        # Turn results into a list of dict rows
        bq_live_rows = []
        for asset_idx, asset_name in enumerate(porfolio_names):
            bq_live_rows.append({
                'date':           live_date, 
                'portfolio_id':   asset_name,
                'predicted_score': float(scores[0][asset_idx]),
                'predicted_rank':  int(predicted_ranks[0][asset_idx]),
                'uuid':           uuid,  # same run-identifier used throughout
                'run_date':       run_time,
                'allocation': 7, # n_k-1,
                # Add run configuration arguments (prefix them with "arg_" to avoid conflicts)
                # **{f'arg_{k}': v for k, v in vars(args).items()},
            })

        # Convert to DataFrame
        bq_live_df = pd.DataFrame(bq_live_rows)
        bq_live_df = sanitize_and_convert_columns(bq_live_df)  # from your earlier helper
        print("Live Day Predictions to BQ:")
        print(bq_live_df.head())

        # --------------------------------------------------
        # 5) SAVE RESULTS TO BIGQUERY
        # --------------------------------------------------
        if is_test > 0:
            table_suffix = f"{table_suffix}_test"

        append_to_bigquery(
            df=bq_live_df,
            dataset_id=DESTINATION_DATASET,
            table_name=f"live_next_day_predictions_{table_suffix}"
        )

        print(f"Saved {len(bq_live_df)} live predictions to BigQuery table "
            f"'live_next_day_predictions_{table_suffix}'.")
        
        # Also save the last day
        df_last_day = sanitize_and_convert_columns(df_last_day)
        create_or_replace_bigquery_table(
            df=df_last_day,
            dataset_id=DESTINATION_DATASET,
            table_name=f"live_last_day_values_{table_suffix}"
        )
        
        start = time.time()
        data_metrics_df = compute_nn_data_quality_metrics(df_long, lookback=(rolling_train_length))
        end = time.time()
        print(f"Data quality metrics computed in {end - start:.3f} s")

        data_metrics_df['uuid'] = uuid
        data_metrics_df['run_date'] = datetime.now()
        append_to_bigquery(data_metrics_df, DESTINATION_DATASET, f"live_data_metrics_{table_suffix}")

        return bq_live_df


    for ind, i in enumerate(range(rolling_train_length, len(m), rolling_test_length)):
        start_time = time.time()  # Start timing for the epoch

        print("Predict period:", w)
        w += 1

        # Get train/test splits
        train = copy.deepcopy(m[i-rolling_train_length:i, :, :])
        test = copy.deepcopy(m[i:i+rolling_test_length, :, :])
        train, a = get_preprocess(train)
        test = preprocess(test, a)

        # Get corresponding returns and dates
        ranks_train = returns[i-rolling_train_length:i]
        ranks_test = returns[i:i + rolling_test_length]
        dates_test = dates[i:i + rolling_test_length]  # Get dates for test period

        all_backtests, model = train__evaluate_listfold(train, ranks_train, epochs, test, ranks_test, dates_test)
        all_results.append(all_backtests)

        loop_time = time.time() - start_time  # Calculate elapsed time for this epoch
        print(f"Single day {w} training time: {loop_time:.2f} seconds")

        ########################################################################
        # CONFORMAL PREDICTION 
        ########################################################################
        cp_time = time.time()

        (daily_agg_pnl,
        pos_long_array,
        pos_short_array,
        ret_long_array,
        ret_short_array,
        pred_ranks_array,   # shape => (rolling_test_length, n_assets)
        top_k,
        dates_array,
        daily_asset_returns, # shape => (rolling_test_length, n_assets)
        scores_array,
        loss_value,
        lookback_loss_value
        ) = all_backtests[0]

        device = next(model.parameters()).device

        # Move features to torch tensor on device, get model scores
        features_tensor = torch.from_numpy(train).float().to(device)
        with torch.no_grad():
            scores_tensor = model(features_tensor)
        scores = scores_tensor.cpu().numpy()  # shape: (num_days, n_assets)

        pred_ranks_train_array = return_rank(scores)

        for current_alpha in alpha_list:
            # 1. Calibrate top threshold using current alpha and mode "single_best"
            top_threshold = rank_based_conformal_calibrate(
                true_ranks=ranks_train,
                pred_ranks=pred_ranks_train_array,
                alpha=current_alpha,
                mode=mode_top
            )
            
            # 2. Calibrate bottom threshold using current alpha and mode "single_worst"
            bottom_threshold = rank_based_conformal_calibrate(
                true_ranks=ranks_train,
                pred_ranks=pred_ranks_train_array,
                alpha=current_alpha,
                mode=mode_bottom
            )
        
            # 3. Apply thresholds to the test block's predicted ranks
            top_picks, top_pick_size = rank_based_conformal_apply(
                pred_ranks_test=pred_ranks_array,
                threshold=top_threshold,
                default_k=7,
                side="top"
            )

            bottom_picks, bottom_pick_size = rank_based_conformal_apply(
                pred_ranks_test=pred_ranks_array,
                threshold=bottom_threshold,
                default_k=7,
                side="bottom"
            )
        
            # 4. Save the conformal results for each test day along with the current alpha.
            for day_idx in range(len(dates_array)):
                day_date = dates_array[day_idx]
                top_list = top_picks[day_idx]    # asset indices for top picks
                bot_list = bottom_picks[day_idx]   # asset indices for bottom picks
                top_names = [porfolio_names[idx] for idx in top_list]
                bot_names = [porfolio_names[idx] for idx in bot_list]
                conformal_results.append({
                    'rolling_idx': ind,
                    'date': day_date,
                    'alpha': current_alpha,
                    'top_threshold': top_threshold,
                    'bottom_threshold': bottom_threshold,
                    # 'top_pick_size': top_pick_size,
                    # 'bottom_pick_size': bottom_pick_size,
                    # 'top_list': top_list,
                    # 'bottom_list': bot_list,
                    # 'top_names': top_names,
                    # 'bottom_names': bot_names
                })

        print(f"Time to complete Conformal Prediction: {(time.time() - cp_time):.2f} seconds")

        ########################################################################
        # WEIGHTWATCHER ANALYSIS (for each new model)
        ########################################################################
        if calculate_ww == 1:
            ww_time = time.time()
            ww_analyzer = ww.WeightWatcher(model=model)
            ww_analysis = ww_analyzer.analyze()
            ww_df = ww_analysis["details"] if isinstance(ww_analysis, dict) else ww_analysis
            ww_df = pd.DataFrame(ww_df).reset_index(drop=True)

            ww_df["rolling_idx"] = ind
            ww_df["uuid"] = uuid
            ww_df["run_date"] = datetime.now()
            ww_df['dates'] = str(dates_test.strftime('%Y-%m-%d').tolist())

            ww_df = sanitize_and_convert_columns(ww_df)
            # Instead of pushing to BigQuery immediately, store the DataFrame in a list.
            ww_results_list.append(ww_df)
            print(f"Time to complete WeightWatcher: {time.time() - ww_time:.2f} seconds")

        ########################################################################
        # SHAP VALUES (for each new model)
        ########################################################################
        if calculate_shap == 1: 
            shap_time = time.time()

            # Compute shap_values with GradientExplainer
            explainer = shap.GradientExplainer(
                model,
                torch.from_numpy(train).float().to(device)
            )
            shap_values = explainer.shap_values(
                torch.from_numpy(test).float().to(device)
            )
            # shap_values shape => (1, n_assets, n_features, n_assets)

            # Extract the main array (assume single model output)
            shap_values_batch = shap_values[0]  # shape (n_assets, n_features, n_assets)

            # Compute overall feature importance across all assets, all outputs
            abs_shap = np.abs(shap_values_batch)               # shape => (n_assets, n_features, n_assets)
            mean_abs_per_feature = abs_shap.mean(axis=(0, 2))    # shape (n_features,)

            # Sort and log top features
            sorted_idx = np.argsort(mean_abs_per_feature)[::-1]
            top_features = []
            for rank, idx_f in enumerate(sorted_idx, start=1):
                top_features.append({
                    "rank": rank,
                    "feature": feature_columns[idx_f],
                    "importance": float(mean_abs_per_feature[idx_f])
                })

            # Convert to DataFrame and prepare for upload
            top_features_df = pd.DataFrame(top_features)
            top_features_df["rolling_idx"] = ind
            top_features_df["uuid"] = uuid
            top_features_df["run_date"] = datetime.now()
            top_features_df['dates'] = str(dates_test.strftime('%Y-%m-%d').tolist())

            top_features_df = sanitize_and_convert_columns(top_features_df)
            # Instead of calling append_to_bigquery immediately, store the DataFrame.
            shap_results_list.append(top_features_df)
            print(f"Time to complete SHAP analysis: {time.time() - shap_time:.2f} seconds")

        # Break if test end met
        if is_test > 0 and w >= is_test:
            break

    # After the loop, concatenate and upload the batched results
    if calculate_ww == 1 and ww_results_list:
        final_ww_df = pd.concat(ww_results_list, ignore_index=True)
        append_to_bigquery(final_ww_df, DESTINATION_DATASET, f"weightwatcher_analysis_{table_suffix}")
        print("Uploaded batch WeightWatcher analysis to BigQuery.")

    if calculate_shap == 1 and shap_results_list:
        final_shap_df = pd.concat(shap_results_list, ignore_index=True)
        append_to_bigquery(final_shap_df, DESTINATION_DATASET, f"shap_feature_importance_{table_suffix}")
        print("Uploaded batch SHAP analysis to BigQuery.")

    if conformal_results:
        final_cp_df = pd.DataFrame(conformal_results)
        final_cp_df['run_date'] = datetime.now()
        final_cp_df['uuid'] = uuid
        append_to_bigquery(final_cp_df, DESTINATION_DATASET, f"conformal_prediction_{table_suffix}")
        print("Uploaded batch conformal prediction analysis to BigQuery.")

    ###############################################################################
    # UPDATED LOGGING & ANALYSIS FOR MULTI-DAY TEST PERIODS
    ###############################################################################

    # We'll rename variables for clarity:
    #   daily_agg_pnl: single float per day (the aggregated daily total return)
    #   daily_asset_returns: shape (n_days, n_assets)
    #   scores_array: shape (n_days, n_assets)
    #   etc.

    bq_pred_rows = []
    for rolling_idx, rolling_results_list in enumerate(all_results):
        # rolling_results_list: list of "period_results" from backtest(...)
        #   for each top_k in range(1, n_k)
        for period_results in rolling_results_list:
            (
                daily_agg_pnl,       # shape (rolling_test_length,)
                pos_long_array,      # shape (rolling_test_length, top_k)
                pos_short_array,     # shape (rolling_test_length, top_k)
                ret_long_array,      # shape (rolling_test_length, top_k)
                ret_short_array,     # shape (rolling_test_length, top_k)
                pred_ranks_array,    # shape (rolling_test_length, n_assets)
                top_k,
                dates_array,         # shape (rolling_test_length,)
                daily_asset_returns, # shape (rolling_test_length, n_assets)
                scores_array,        # shape (rolling_test_length, n_assets)
                loss_value,          # scalar
                lookback_loss_value  # scalar
            ) = period_results

            # For each day in the test set
            for day_idx in range(len(dates_array)):
                day_date = dates_array[day_idx]

                # The day's per-asset data
                day_returns = daily_asset_returns[day_idx]  # shape (n_assets,)
                day_scores  = scores_array[day_idx]         # shape (n_assets,)

                # Convert each to ranks
                day_actual_ranks = return_rank(day_returns)
                day_pred_ranks   = return_rank(day_scores)

                # Loop over each asset
                for asset_idx, asset_name in enumerate(porfolio_names):
                    bq_pred_rows.append({
                        'date':          day_date,
                        'names':         asset_name,
                        'returns':       day_returns[asset_idx],
                        'score':         day_scores[asset_idx],
                        'actual_ranks':  day_actual_ranks[asset_idx],
                        'pred_ranks':    day_pred_ranks[asset_idx],
                        # 'top_k':         top_k,
                        'loss':          loss_value,
                        'lookback_loss': lookback_loss_value
                    })

    # Convert to a DataFrame
    bq_pred_df = pd.DataFrame(bq_pred_rows).drop_duplicates()

    ###############################################################################
    # Similarly, if you want a day-level df_results (one row per day):
    ###############################################################################
    all_dates = []
    all_daily_agg_pnl = []
    all_ks = []
    all_losses = []
    all_lookback_losses = []

    for rolling_idx, rolling_results_list in enumerate(all_results):
        for period_results in rolling_results_list:
            (
                daily_agg_pnl,       # shape (n_days,)
                pos_long_array,
                pos_short_array,
                ret_long_array,
                ret_short_array,
                pred_ranks_array,
                top_k,
                dates_array,
                daily_asset_returns,
                scores_array,
                loss_value,
                lookback_loss_value
            ) = period_results

            n_days = len(dates_array)
            all_dates.extend(dates_array.tolist())
            all_daily_agg_pnl.extend(daily_agg_pnl.tolist())  # aggregated daily PnL
            all_ks.extend([top_k]*n_days)
            all_losses.extend([loss_value]*n_days)
            all_lookback_losses.extend([lookback_loss_value]*n_days)

    df_results = pd.DataFrame({
        'date': all_dates,
        'return': all_daily_agg_pnl,  # single daily return across all assets
        'k': all_ks,
        'loss': all_losses,
        'lookback_loss': all_lookback_losses
    })

    # Sort by date (if desired)
    df_results = df_results.sort_values('date').reset_index(drop=True)

    # If 'return' above is indeed the daily log return:
    df_results['simple_return'] = np.exp(df_results['return']) - 1

    def calculate_summary_with_yearly_stats(df_results, args):
        """
        Create a summary DataFrame of overall and yearly portfolio statistics.

        All keys from the args dictionary are added to each row with an "arg_" prefix.
        """

        # ------------------------------------------------------------------------------------------------
        # ORIGINAL CODE: No changes here
        # ------------------------------------------------------------------------------------------------
        df_results['year'] = pd.to_datetime(df_results['date']).dt.year
        unique_years = sorted(df_results['year'].unique())
        unique_ks = np.sort(df_results['k'].unique())

        summary_rows = []

        for k in unique_ks:
            df_k = df_results[df_results['k'] == k].copy().sort_values('date')
            df_k['portfolio_value'] = 100 * (1 + df_k['simple_return']).cumprod()
            mean_daily_return = df_k['simple_return'].mean()
            annualized_return = ((df_k['portfolio_value'].iloc[-1] / 100) ** (252 / len(df_k))) - 1
            win_rate = (df_k['simple_return'] > 0).mean() * 100
            rolling_sharpe = df_k['simple_return'].rolling(window=252).apply(
                lambda x: (x.mean() / x.std()) * np.sqrt(252) if x.std() != 0 else np.nan
            )
            std_rolling_sharpe = rolling_sharpe.std()
            neg_returns = df_k['simple_return'][df_k['simple_return'] < 0]
            downside_deviation = neg_returns.std() * np.sqrt(252) if len(neg_returns) > 0 else 0
            sortino_ratio = (mean_daily_return / neg_returns.std()) * np.sqrt(252) if len(neg_returns) > 0 and neg_returns.std() != 0 else np.nan
            min_drawdown = (df_k['portfolio_value'] / df_k['portfolio_value'].cummax() - 1).min()
            calmar_ratio = annualized_return / (-min_drawdown) if min_drawdown != 0 else np.nan
            drawdown = (df_k['portfolio_value'] / df_k['portfolio_value'].cummax() - 1) * 100
            ulcer_index = np.sqrt(np.mean(drawdown ** 2))
            wins = df_k['simple_return'][df_k['simple_return'] > 0]
            losses = df_k['simple_return'][df_k['simple_return'] < 0]
            avg_win = wins.mean() if len(wins) > 0 else 0
            avg_loss = losses.abs().mean() if len(losses) > 0 else 0
            win_loss_ratio = avg_win / avg_loss if avg_loss != 0 else np.inf
            total_wins = wins.sum() if len(wins) > 0 else 0
            total_losses = losses.abs().sum() if len(losses) > 0 else 0
            profit_factor = total_wins / total_losses if total_losses != 0 else np.inf
            omega_ratio = total_wins / total_losses if total_losses != 0 else np.inf
            skewness = df_k['simple_return'].skew()
            kurtosis = df_k['simple_return'].kurtosis()
            sharpe_ratio = (mean_daily_return / df_k['simple_return'].std()) * np.sqrt(252) if df_k['simple_return'].std() != 0 else np.nan

            # ------------------------------------------------------------------------------------------------
            # CHANGES: Use negative streaks instead of drawdown periods
            # ------------------------------------------------------------------------------------------------
            in_negative = df_k['simple_return'] < 0
            negative_periods = []
            current_period = 0

            for is_neg in in_negative:
                if is_neg:
                    current_period += 1
                else:
                    if current_period > 0:
                        negative_periods.append(current_period)
                        current_period = 0
            if current_period > 0:
                negative_periods.append(current_period)

            if negative_periods:
                longest_dd_days = -max(negative_periods)  # negative to match original
                avg_dd_days = -sum(negative_periods) / len(negative_periods)
            else:
                longest_dd_days = 0
                avg_dd_days = 0

            # Count specific lengths of negative-streak periods
            dd_3  = sum(1 for x in negative_periods if x == 3)
            dd_4  = sum(1 for x in negative_periods if x == 4)
            dd_6  = sum(1 for x in negative_periods if x == 6)
            dd_7  = sum(1 for x in negative_periods if x == 7)
            dd_8  = sum(1 for x in negative_periods if x == 8)
            dd_9  = sum(1 for x in negative_periods if x == 9)
            dd_10 = sum(1 for x in negative_periods if x == 10)
            dd_10_plus = sum(1 for x in negative_periods if x > 10)

            # ------------------------------------------------------------------------------------------------
            # CHANGES: ADDED 30, 60, 120 day portfolio volatility, returns, and Sharpe
            # ------------------------------------------------------------------------------------------------
            def lookback_metrics(period):
                if len(df_k) < period:
                    return (np.nan, np.nan, np.nan, np.nan, np.nan)  
                lookback_returns = df_k['simple_return'].iloc[-period:]

                # daily average & standard deviation over the window
                avg_return = lookback_returns.mean() * 100
                std_return = lookback_returns.std() * 100

                # total return, annualized volatility, sharpe as before
                total_return = (df_k['portfolio_value'].iloc[-1] / df_k['portfolio_value'].iloc[-period] - 1) * 100
                vol = lookback_returns.std() * np.sqrt(252) * 100
                sharpe = (lookback_returns.mean() / lookback_returns.std()) * np.sqrt(252) if lookback_returns.std() != 0 else np.nan
                return avg_return, std_return, total_return, vol, sharpe

            avg_30, std_30, ret_30, vol_30, sharpe_30 = lookback_metrics(30)
            avg_60, std_60, ret_60, vol_60, sharpe_60 = lookback_metrics(60)
            avg_120, std_120, ret_120, vol_120, sharpe_120 = lookback_metrics(120)

            # Overall daily average & daily std
            overall_avg_daily = df_k['simple_return'].mean() * 100
            overall_std_daily = df_k['simple_return'].std() * 100

            # ------------------------------------------------------------------------------------------------
            # ORIGINAL CODE: No changes, except for storing new fields in row
            # ------------------------------------------------------------------------------------------------
            var_95 = np.percentile(df_k['simple_return'], 5)
            daily_var = var_95 * 100
            df_k['year_month'] = pd.to_datetime(df_k['date']).dt.to_period('M')
            monthly_returns = df_k.groupby('year_month')['simple_return'].apply(
                lambda x: (1 + x).prod() - 1
            )
            worst_month_perf = monthly_returns.min() * 100 if not monthly_returns.empty else 0

            row = {
                'Ranks Predicted': k,
                'Average Loss': df_k['loss'].mean(),
                'Average Lookback Loss': df_k['loss'][-60:].mean(),
                'Total Return (%)': (df_k['portfolio_value'].iloc[-1] / 100 - 1) * 100,
                'Annualized Return (%)': annualized_return * 100,
                'Annualized Volatility (%)': np.sqrt(252) * df_k['simple_return'].std() * 100,
                'Sharpe Ratio': sharpe_ratio,
                'Max Drawdown (%)': min_drawdown * 100,
                'Win Rate (%)': win_rate,
                'Std of Rolling Sharpe': std_rolling_sharpe,
                'Sortino Ratio': sortino_ratio,
                'Calmar Ratio': calmar_ratio,
                'Ulcer Index': ulcer_index,
                'Win/Loss Ratio': win_loss_ratio,
                'Profit Factor': profit_factor,
                'Omega Ratio': omega_ratio,
                'Skewness': skewness,
                'Kurtosis': kurtosis,
                'Longest DD Days': longest_dd_days,
                'Daily VaR': daily_var,
                'Avg Drawdown Days': avg_dd_days,
                'Worst Month Perf %': worst_month_perf,

                # CHANGES: NEW FIELDS FOR LOOKBACK VOL, RETURN, SHARPE
                'Overall Avg Return (%)': overall_avg_daily,   # NEW
                'Overall Std Dev (%)': overall_std_daily,      # NEW

                '30d Avg Return (%)': avg_30,                  # NEW
                '30d Std Dev (%)': std_30,                     # NEW
                '30d Ret (%)': ret_30,
                '30d Vol (%)': vol_30,
                '30d Sharpe': sharpe_30,

                '60d Avg Return (%)': avg_60,                  # NEW
                '60d Std Dev (%)': std_60,                     # NEW
                '60d Ret (%)': ret_60,
                '60d Vol (%)': vol_60,
                '60d Sharpe': sharpe_60,

                '120d Avg Return (%)': avg_120,                # NEW
                '120d Std Dev (%)': std_120,                   # NEW
                '120d Ret (%)': ret_120,
                '120d Vol (%)': vol_120,
                '120d Sharpe': sharpe_120,

                # CHANGES: DRAWDOWN PERIOD COUNTS
                'DD_Count_3': dd_3,
                'DD_Count_4': dd_4,
                'DD_Count_6': dd_6,
                'DD_Count_7': dd_7,
                'DD_Count_8': dd_8,
                'DD_Count_9': dd_9,
                'DD_Count_10': dd_10,
                'DD_Count_10plus': dd_10_plus,

                'UUID': uuid
            }

            for year in unique_years:
                df_year = df_k[df_k['year'] == year].copy()
                df_year['year_month'] = pd.to_datetime(df_k['date']).dt.to_period('M')
                df_year['year_month'] = pd.to_datetime(df_k['date']).dt.to_period('M')
                year_monthly_returns = df_year.groupby('year_month')['simple_return'].apply(
                    lambda x: (1 + x).prod() - 1
                )
                worst_year_month_perf = year_monthly_returns.min() * 100 if not year_monthly_returns.empty else 0

                if not df_year.empty:
                    mean_daily_return_year = df_year['simple_return'].mean()
                    start_value = df_year['portfolio_value'].iloc[0]
                    end_value = df_year['portfolio_value'].iloc[-1]
                    yearly_return = (end_value / start_value) - 1
                    yearly_volatility = np.sqrt(252) * df_year['simple_return'].std()
                    yearly_max_drawdown = (df_year['portfolio_value'] / df_year['portfolio_value'].cummax() - 1).min()
                    yearly_win_rate = (df_year['simple_return'] > 0).mean() * 100
                    neg_returns_year = df_year['simple_return'][df_year['simple_return'] < 0]
                    yearly_downside_deviation = neg_returns_year.std() * np.sqrt(252) if len(neg_returns_year) > 0 else 0
                    yearly_sortino = (mean_daily_return_year / neg_returns_year.std()) * np.sqrt(252) if len(neg_returns_year) > 0 and neg_returns_year.std() != 0 else np.nan
                    yearly_calmar = yearly_return / (-yearly_max_drawdown) if yearly_max_drawdown != 0 else np.nan
                    yearly_drawdowns = (df_year['portfolio_value'] / df_year['portfolio_value'].cummax() - 1) * 100
                    yearly_ulcer_index = np.sqrt(np.mean(yearly_drawdowns ** 2))
                    wins_year = df_year['simple_return'][df_year['simple_return'] > 0]
                    losses_year = df_year['simple_return'][df_year['simple_return'] < 0]
                    avg_win_year = wins_year.mean() if len(wins_year) > 0 else 0
                    avg_loss_year = losses_year.abs().mean() if len(losses_year) > 0 else 0
                    win_loss_ratio_year = avg_win_year / avg_loss_year if avg_loss_year != 0 else np.inf
                    total_wins_year = wins_year.sum() if len(wins_year) > 0 else 0
                    total_losses_year = losses_year.abs().sum() if len(losses_year) > 0 else 0
                    profit_factor_year = total_wins_year / total_losses_year if total_losses_year != 0 else np.inf
                    omega_ratio_year = total_wins_year / total_losses_year if total_losses_year != 0 else np.inf
                    skewness_year = df_year['simple_return'].skew()
                    kurtosis_year = df_year['simple_return'].kurtosis()
                    yearly_sharpe = (mean_daily_return_year / df_year['simple_return'].std()) * np.sqrt(252) if df_year['simple_return'].std() != 0 else np.nan

                    row.update({
                        f'{year}_Return (%)': yearly_return * 100,
                        f'{year}_Volatility (%)': yearly_volatility * 100,
                        f'{year}_Sharpe': yearly_sharpe,
                        f'{year}_Max Drawdown (%)': yearly_max_drawdown * 100,
                        f'{year}_Avg_Loss': df_year['loss'].mean(),
                        f'{year}_Avg_Lookback_Loss': df_year['loss'][-30:].mean(),
                        f'{year}_Win_Rate (%)': yearly_win_rate,
                        f'{year}_Sortino_Ratio': yearly_sortino,
                        f'{year}_Calmar_Ratio': yearly_calmar,
                        f'{year}_Ulcer_Index': yearly_ulcer_index,
                        f'{year}_Win_Loss_Ratio': win_loss_ratio_year,
                        f'{year}_Profit_Factor': profit_factor_year,
                        f'{year}_Omega_Ratio': omega_ratio_year,
                        f'{year}_Skewness': skewness_year,
                        f'{year}_Kurtosis': kurtosis_year,
                        f'{year}_Worst_Month_Perf%': worst_year_month_perf
                    })

            for key, value in args.items():
                row[f'arg_{key}'] = value

            summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)

        # ------------------------------------------------------------------------------------------------
        # ORIGINAL CODE: Just adding new columns to the base_cols
        # ------------------------------------------------------------------------------------------------
        base_cols = [
            'Ranks Predicted', 'Average Loss', 'Average Lookback Loss',
            'Total Return (%)', 'Annualized Return (%)', 'Annualized Volatility (%)',
            'Sharpe Ratio', 'Max Drawdown (%)', 'Win Rate (%)', 'Std of Rolling Sharpe',
            'Sortino Ratio', 'Calmar Ratio', 'Ulcer Index', 'Win/Loss Ratio',
            'Profit Factor', 'Omega Ratio', 'Skewness', 'Kurtosis',
            'Longest DD Days', 'Daily VaR', 'Avg Drawdown Days', 'Worst Month Perf %',

            # New overall average & std
            'Overall Avg Return (%)', 'Overall Std Dev (%)',

            # 30d
            '30d Avg Return (%)', '30d Std Dev (%)', '30d Ret (%)', '30d Vol (%)', '30d Sharpe',

            # 60d
            '60d Avg Return (%)', '60d Std Dev (%)', '60d Ret (%)', '60d Vol (%)', '60d Sharpe',

            # 120d
            '120d Avg Return (%)', '120d Std Dev (%)', '120d Ret (%)', '120d Vol (%)', '120d Sharpe',

            'DD_Count_3', 'DD_Count_4', 'DD_Count_6', 'DD_Count_7', 'DD_Count_8',
            'DD_Count_9', 'DD_Count_10', 'DD_Count_10plus'
        ]
        year_cols = sorted(
            [col for col in summary_df.columns
            if col.split('_')[0].isdigit() and len(col.split('_')[0]) == 4],
            key=lambda x: (int(x.split('_')[0]), x)
        )
        args_cols = sorted([col for col in summary_df.columns if col.startswith('arg_')])

        summary_df = summary_df[base_cols + year_cols + args_cols + ['UUID']]

        return summary_df

    # Sort by date
    df_results = df_results.sort_values('date').reset_index(drop=True)

    # Convert log returns to simple returns
    df_results['simple_return'] = np.exp(df_results['return']) - 1

    # Use in the main code:
    summary_df = calculate_summary_with_yearly_stats(
        df_results, args = args

    )

    ###############################################################################
    # 4) Plot cumulative portfolio value for each k
    ###############################################################################
    unique_ks = np.sort(df_results['k'].unique())
    performance_fig = plt.figure(figsize=(15, 8))

    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_ks)))

    for idx, k_val in enumerate(unique_ks):
        df_k = df_results[df_results['k'] == k_val].copy()
        df_k = df_k.sort_values('date')
        # Multiply up from 100
        df_k['portfolio_value'] = 100 * (1 + df_k['simple_return']).cumprod()

        plt.plot(df_k['date'], df_k['portfolio_value'],
                label=f'K={k_val}',
                color=colors[idx],
                linewidth=2)

    plt.title('Portfolio Value Over Time by K Configuration\n($100 Initial Investment)')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True, alpha=0.3)
    plt.legend(title='Configuration', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()

    def generate_file_name(config_dict, file_type, extension):
        """Generate filename with UUID"""
        date_str = datetime.now().strftime("%Y-%m-%d")
        return f"{date_str}_{config_dict['UUID']}|{file_type}.{extension}"

    def create_directory(base_path):
        """Create directory structure with date-based folders"""
        date_str = datetime.now().strftime("%Y-%m-%d")
        date_folder = os.path.join(base_path, date_str)
        os.makedirs(date_folder, exist_ok=True)
        return date_folder

    def add_filename_column(df, file_path):
        """Add RunID column based on filename for future joining"""
        filename = os.path.splitext(os.path.basename(file_path))[0]
        truncated_filename = '_'.join(filename.split('|')[:-1])
        df['RunID'] = truncated_filename
        return df

    def save_summary_df(summary_df, file_path):
        """Save summary DataFrame with model information"""
        df = summary_df.copy()
        df["model"] = "ListFold"
        df = add_filename_column(df, file_path)
        df.to_csv(file_path, index=False)

    def save_portfolio_values(df_results, file_path):
        """Save portfolio values with RunID"""
        portfolio_values = []

        for k in df_results['k'].unique():
            df_k = df_results[df_results['k'] == k].copy()
            df_k = df_k.sort_values('date')
            df_k['portfolio_value'] = 100 * (1 + df_k['simple_return']).cumprod()
            portfolio_values.append(df_k[['date', 'k', 'portfolio_value']])

        combined_df = pd.concat(portfolio_values)
        combined_df = add_filename_column(combined_df, file_path)
        combined_df.to_csv(file_path, index=False)

    def save_plot(plot_object, file_path):
        """Save a plot to disk"""
        plot_object.savefig(file_path)
        plt.close()

    def save_listfold_results(base_path, summary_df, df_results, performance_fig):
        """Save all ListFold results using the summary DataFrame format"""
        # Get configuration from first row of summary_df
        config = {
            'UUID': summary_df['UUID'].iloc[0]
        }

        # Create folder and generate file paths
        folder_path = create_directory(base_path)
        file_paths = {
            "summary": os.path.join(folder_path, generate_file_name(config, "summary_stats", "csv")),
            "portfolio_values": os.path.join(folder_path, generate_file_name(config, "portfolio_values", "csv")),
            "performance_plot": os.path.join(folder_path, generate_file_name(config, "performance", "png"))
        }

        # Save all files
        save_summary_df(summary_df, file_paths["summary"])
    #     save_portfolio_values(df_results, file_paths["portfolio_values"])
        save_plot(performance_fig, file_paths["performance_plot"])

        print(f"Files saved successfully in {folder_path}")
        return file_paths

    import os
    import logging
    from datetime import datetime

    import pandas as pd
    from google.cloud import storage, bigquery

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s %(message)s'
    )




    def save_to_gcs(bucket_name: str, file_name: str, content: bytes) -> None:
        """
        Save in-memory content (e.g., CSV string, image bytes) to a file in GCS.

        Parameters
        ----------
        bucket_name : str
            The name of the GCS bucket (with or without "gs://").
        file_name : str
            The destination file path in the bucket.
        content : bytes
            The data to upload.
        """
        bucket_name_clean = bucket_name.replace("gs://", "")
        bucket = storage_client.bucket(bucket_name_clean)
        blob = bucket.blob(file_name)

        if isinstance(content, str):
            content = content.encode('utf-8')  # Convert to bytes if necessary

        blob.upload_from_string(content)
        logging.info(f"File {file_name} saved to {bucket_name_clean}.")


    # =============================================================================
    # MAIN SAVE FUNCTION
    # =============================================================================

    def save_listfold_results_to_gcs_and_bigquery(
        base_path: str,
        summary_df: pd.DataFrame,
        df_results: pd.DataFrame,
        bq_pred_df: pd.DataFrame,
        data_metrics_df: pd.DataFrame,
        performance_fig
    ) -> None:
        """
        Save DataFrames (summary_df, df_results, bq_pred_df) and performance figure
        to both GCS and BigQuery.
        """
        # Prepare paths and file names
        date_str = datetime.now().strftime("%Y-%m-%d")
        if "uuid" not in summary_df.columns:
            raise ValueError("summary_df must contain a column named 'uuid'.")

        unique_id = summary_df['uuid'].iloc[0]
        file_prefix = f"{date_str}/{unique_id}"

        # CSV filenames
        summary_file_name = f"{file_prefix}_summary_stats.csv"
        portfolio_file_name = f"{file_prefix}_portfolio_values.csv"
        predictions_file_name = f"{file_prefix}_predictions.csv"
        performance_plot_name = f"{file_prefix}_performance.png"

        # -------------------------------------------------------------------------
        # Save CSV files to GCS
        # -------------------------------------------------------------------------
        # 1. summary_df
        summary_csv = summary_df.to_csv(index=False)
        save_to_gcs(STAGING_BUCKET, summary_file_name, summary_csv)

        # 2. df_results
        results_csv = df_results.to_csv(index=False)
        save_to_gcs(STAGING_BUCKET, portfolio_file_name, results_csv)

        # 3. bq_pred_df
        pred_csv = bq_pred_df.to_csv(index=False)
        save_to_gcs(STAGING_BUCKET, predictions_file_name, pred_csv)

        # -------------------------------------------------------------------------
        # Save performance figure to GCS
        # -------------------------------------------------------------------------
        save_performance_figure_to_gcs(performance_fig, STAGING_BUCKET, performance_plot_name)

        # -------------------------------------------------------------------------
        # Append DataFrames to BigQuery
        # -------------------------------------------------------------------------
        # NOTE: Update table names as needed.
        try:
            append_to_bigquery(summary_df, DESTINATION_DATASET, f"test_summary_table_{table_suffix}")

            logging.info("All data appended to BigQuery successfully.")
        except Exception as e:
            logging.error(f"Error during BigQuery appends: {str(e)}")
            # Depending on your use case, decide whether to raise or continue
            raise

        try:
            append_to_bigquery(df_results, DESTINATION_DATASET, f"test_results_table_{table_suffix}")

            logging.info("All data appended to BigQuery successfully.")
        except Exception as e:
            logging.error(f"Error during BigQuery appends: {str(e)}")
            # Depending on your use case, decide whether to raise or continue
            raise

        try:
            append_to_bigquery(bq_pred_df, DESTINATION_DATASET, f"test_predictions_table_{table_suffix}")

            logging.info("All data appended to BigQuery successfully.")
        except Exception as e:
            logging.error(f"Error during BigQuery appends: {str(e)}")
            # Depending on your use case, decide whether to raise or continue
            raise


        try:
            append_to_bigquery(data_metrics_df, DESTINATION_DATASET, f"test_data_metrics_{table_suffix}")

            logging.info("All data appended to BigQuery successfully.")
        except Exception as e:
            logging.error(f"Error during BigQuery appends: {str(e)}")
            # Depending on your use case, decide whether to raise or continue
            raise

        logging.info("Files saved to GCS and data appended to BigQuery successfully.")

    def save_performance_figure_to_gcs(performance_fig, staging_bucket, performance_plot_name):
        """
        Save a Matplotlib figure directly to GCS using an in-memory buffer.
        """
        # Create an in-memory bytes buffer
        buffer = io.BytesIO()

        # Save the figure to the buffer in PNG format
        performance_fig.savefig(buffer, format='png')
        buffer.seek(0)  # Move to the beginning of the buffer

        # Read the bytes from the buffer
        plot_content = buffer.getvalue()

        # Upload to GCS
        save_to_gcs(staging_bucket, performance_plot_name, plot_content)

        # Close the buffer
        buffer.close()

    # =============================================================================
    # EXAMPLE USAGE
    # =============================================================================

    # 1. Sanitize DataFrames
    summary_df = sanitize_and_convert_columns(summary_df)
    df_results = sanitize_and_convert_columns(df_results)
    bq_pred_df = sanitize_and_convert_columns(bq_pred_df)
    data_metrics_df = compute_nn_data_quality_metrics(df_long, lookback=n_features * rolling_train_length)


    # 2. Add UUID columns
    df_results["uuid"] = summary_df.uuid.unique()[0]
    bq_pred_df["uuid"] = summary_df.uuid.unique()[0]
    data_metrics_df['uuid'] = summary_df.uuid.unique()[0]
    # summary_df["uuid"] = datetime.now().strftime("%Y-%m-%d-%Hh-%Mm_")+summary_df.uuid.unique()[0]


    # 2.1 Add Dates
    df_results["run_date"] = datetime.now()
    bq_pred_df["run_date"] = datetime.now()
    summary_df["run_date"] = datetime.now()
    data_metrics_df["run_date"] = datetime.now()

    # # 3. Save
    save_listfold_results_to_gcs_and_bigquery(
        base_path="Rank Optimized Portfolio Selection_test",
        summary_df=summary_df,
        df_results=df_results,
        bq_pred_df=bq_pred_df,
        data_metrics_df=data_metrics_df,
        performance_fig=performance_fig
    )