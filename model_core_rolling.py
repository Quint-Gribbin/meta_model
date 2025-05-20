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
        table_suffix="V001_testing",
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
        return_lag=1,
        core_model_column="long_return",
        l0_config="base",
        use_rank_features=1,
        rolling_window=20,
):
    
    # Show all the input arguments to the model
    args = locals().copy()

    # Popping out environment variables and other non-args
    args.pop("cluster_df")
    keys_to_pop = [x for x in list(args.keys()) if x[0] == "_"] + ['In', 'Out', 'get_ipython', 'exit', 'quit']
    for key in keys_to_pop:
        if key in args:
            args.pop(key)
    print("Here are all the input arguments to the model:")
    print(args)

    YIELDS_TABLE = "issachar-feature-library.core_raw.factor_yields"
    INDEX_RETURNS_TABLE = "josh-risk.IssacharReporting.Index_Returns"

    # Import key libraries
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    from datetime import datetime
    import os
    import io
    import pandas_gbq
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    from sklearn.preprocessing import MinMaxScaler
    import time
    import shap
    import weightwatcher as ww
    from scipy.stats import skew, kurtosis
    import random
    import subprocess
    import pytz
    from datetime import datetime, timedelta

    # Set other enviornment variables
    holdout_start = pd.to_datetime(holdout_start)
    current_time = pd.Timestamp.now()
    uuid = str(current_time) + "__model=Meta_Model__" + "__".join(f"{key}={value}" for key, value in args.items())

    # Google Cloud imports
    from google.cloud import bigquery, storage

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
    set_random_seeds(random_seed)

    # Initialize the GCS clients
    PROJECT_ID = "issachar-feature-library"
    STAGING_BUCKET = "gs://qjg-test"
    DESTINATION_DATASET = "qjg_meta_model"
    client = bigquery.Client(project="issachar-feature-library")
    storage_client = storage.Client(project=PROJECT_ID)

    # Function to fetch data from the specified table
    def read_table_to_dataframe(client, table_id, query=None):
        if query is None:
            query = f"SELECT * FROM `{table_id}`"
        else:
            query = query
        query_job = client.query(query)  # Make an API request to execute the query
        df = query_job.to_dataframe()   # Convert the results to a Pandas DataFrame
        return df
    
    ##########################################################################
    # 2. Update with the core model predictions
    ##########################################################################

    test = pd.read_csv("risk_v2.csv")
    test['date'] = pd.to_datetime(test['date'])
    test = test.sort_values('date', ascending=True)
    # test = test[test['date'] > "2007-01-01"]
    test['portfolio_id'] = 'core_model'
    test['core_model'] = test[core_model_column] / 100 # total_return
    # test = test.set_index("date")
    df_port = test[['core_model', 'date']]
    df_port['date'] = df_port['date'].shift(-1)
    df_port = df_port.dropna()


    ##########################################################################
    # 3. Feature Engineering
    ##########################################################################

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
                        mfr.date
            '''

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
                'high_short_interest',
                'long_momentum'
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

    # TODO: Add more logic here
    cluster_rank_query = f"SELECT * FROM `issachar-feature-library.qjg.meta-model-daily-spreads`"
    cluster_rank_df = pd.read_gbq(cluster_rank_query, project_id='issachar-feature-library', use_bqstorage_api=True)
    cluster_rank_pivoted = cluster_rank_df.pivot_table(index='date', columns='cluster', values='total_return').sort_values("date").shift(1)
    df_long = df_long.merge(cluster_rank_pivoted.add_prefix(f'clust_lag{1}_'), on='date', how='left')


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

    if use_rank_features == 1:
        import numpy as np
        import pandas as pd
        from typing import Iterable

        # ──────────────────────────────────────────────────────────────────────────────
        # 0.  Helpers
        # ──────────────────────────────────────────────────────────────────────────────

        def _prepare_ranks(df: pd.DataFrame) -> pd.DataFrame:
            """Add an integer `rank` column (1 = highest return within each date)."""
            out = df.copy()
            out["date"] = pd.to_datetime(out["date"])
            out["returns"] = out["returns"].astype(float)
            out["rank"] = (
                out.groupby("date")["returns"].rank(ascending=False, method="first").astype(int)
            )
            return out


        # — Inequality metrics ————————————————————————————————————————————————

        def _gini(x: np.ndarray) -> float:
            """Cross‑sectional Gini coefficient on |x| (robust to sign)."""
            if x.size == 0:
                return np.nan
            y = np.sort(np.abs(x))
            n = y.size
            cum = y.cumsum()
            return (n + 1 - 2 * (cum / cum[-1]).sum()) / n


        def _theil(x: np.ndarray) -> float:
            """Theil entropy index on |x| (0 = equal, ↑ for concentration)."""
            x = np.abs(x)
            if x.size == 0 or np.allclose(x, 0):
                return np.nan
            mu = x.mean()
            r = x / mu
            r = r[r > 0]
            return np.mean(r * np.log(r))


        def _herfindahl(x: np.ndarray) -> float:
            """Herfindahl‑Hirschman index on |x| (range (0,1])."""
            x = np.abs(x)
            s = x.sum()
            if s == 0:
                return np.nan
            p = x / s
            return np.sum(p ** 2)


        def _var_cvar(r: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
            """Value‑at‑Risk and Conditional VaR (CVaR) for tail `alpha`."""
            if r.size == 0:
                return np.nan, np.nan
            var = np.quantile(r, alpha)
            cvar = r[r <= var].mean() if (r <= var).any() else np.nan
            return var, cvar


        # ──────────────────────────────────────────────────────────────────────────────
        # 1.  Daily cross‑sectional statistics
        # ──────────────────────────────────────────────────────────────────────────────

        def _daily_stats(g: pd.DataFrame) -> pd.Series:
            """Return one row of features for a single trading date (groupby‑apply)."""
            r = g["returns"].values
            ranks = g["rank"].values

            n = len(r)
            top_q = r[ranks <= n * 0.25]
            bot_q = r[ranks > n * 0.75]
            top_d = r[ranks <= n * 0.10]
            bot_d = r[ranks > n * 0.90]

            # Tail metrics
            tail_flag = np.abs(r - r.mean()) > 2 * r.std(ddof=0)
            var5, cvar5 = _var_cvar(r, 0.05)
            var95, cvar95 = _var_cvar(-r, 0.05)  # symmetric right‑tail on returns
            var95 = -var95
            cvar95 = -cvar95

            return pd.Series({
                # — Central tendency & dispersion —
                "cs_mean"         : r.mean(),
                "cs_median"       : np.median(r),
                "cs_min"          : r.min(),
                "cs_max"          : r.max(),
                "cs_std"          : r.std(ddof=0),
                "cs_cv"           : r.std(ddof=0) / (np.abs(r.mean()) + 1e-12),
                "cs_iqr"          : np.quantile(r, 0.75) - np.quantile(r, 0.25),
                "cs_mad"          : np.mean(np.abs(r - np.median(r))),
                # — Shape —
                "cs_skew"         : pd.Series(r).skew(),
                "cs_kurt"         : pd.Series(r).kurt(),  # excess
                # — Inequality & concentration —
                "cs_gini"         : _gini(r),
                "cs_theil"        : _theil(r),
                "cs_hhi"          : _herfindahl(r),
                "cs_eff_winners"  : 1 / _herfindahl(r) if not np.isnan(_herfindahl(r)) else np.nan,
                # — Cross‑sectional spreads —
                "cs_q1_minus_q4"  : top_q.mean() - bot_q.mean(),
                "cs_d1_minus_d10": top_d.mean() - bot_d.mean(),
                # — Sign & tail risk —
                "cs_pct_pos"      : np.mean(r > 0),
                "cs_pct_neg"      : np.mean(r < 0),
                "cs_tail_2sigma"  : tail_flag.mean(),
                "cs_var_5"        : var5,
                "cs_cvar_5"       : cvar5,
                "cs_var_95"       : var95,
                "cs_cvar_95"      : cvar95,
                # — Rank entropy —
                "rank_entropy"    : -(
                    (np.bincount(ranks, minlength=n + 1)[1:] / n)
                    * np.log(np.bincount(ranks, minlength=n + 1)[1:] / n + 1e-9)
                ).sum(),
            })


        # ──────────────────────────────────────────────────────────────────────────────
        # 2.  Rank‑based temporal features
        # ──────────────────────────────────────────────────────────────────────────────

        def _spearman_autocorr(df: pd.DataFrame, lag: int) -> pd.Series:
            """Spearman ρ between today’s and t‑lag’s ranks (one value per date)."""
            df = df.sort_values("date")
            dates = df["date"].unique()
            out = pd.Series(index=dates, dtype=float, name=f"rank_autocorr_{lag}d")

            per_date = {d: g[["requestId", "rank"]] for d, g in df.groupby("date")}

            for i, d in enumerate(dates):
                j = i - lag
                if j < 0:
                    continue
                merged = per_date[d].merge(per_date[dates[j]], on="requestId", suffixes=("_t", "_tlag"))
                if len(merged) > 1:
                    out.loc[d] = merged["rank_t"].corr(merged["rank_tlag"], method="spearman")
            return out


        def _top_k_persistence(df: pd.DataFrame, k: int, window: int) -> pd.Series:
            """Fraction of today’s top‑k IDs that also appeared in the top‑k set during the previous `window` days."""
            df = df.sort_values("date")
            dates = df["date"].unique()
            top_per_date = {d: set(g.loc[g["rank"] <= k, "requestId"]) for d, g in df.groupby("date")}
            out = pd.Series(index=dates, dtype=float, name=f"top{k}_persist_{window}d")

            for i, d in enumerate(dates):
                prev_union = set().union(*(top_per_date[dates[j]] for j in range(max(0, i - window), i)))
                inter = top_per_date[d] & prev_union
                if prev_union:
                    out.loc[d] = len(inter) / k
            return out


        # ──────────────────────────────────────────────────────────────────────────────
        # 3.  Public API – build the full feature matrix
        # ──────────────────────────────────────────────────────────────────────────────

        def build_features(
            cluster_df: pd.DataFrame,
            autocorr_lags: Iterable[int] = (1, 5, 20),
            top_k: Iterable[int] = (10, 50, 100),
            persistence_windows: Iterable[int] = (3, 5, 20),
        ) -> pd.DataFrame:
            """Construct cross‑sectional feature matrix.

            Parameters
            ----------
            cluster_df : DataFrame with columns ['date', 'requestId', 'returns', ...]
            autocorr_lags : lags for Spearman rank autocorrelation.
            top_k : k‑values for top‑k persistence.
            persistence_windows : rolling windows for persistence metrics.

            Returns
            -------
            DataFrame indexed by date with engineered features.
            """
            df_r = _prepare_ranks(cluster_df)

            # 1) Static daily metrics
            features = df_r.groupby("date").apply(_daily_stats).sort_index()

            # 2) Rank autocorrelations
            for lag in autocorr_lags:
                features = features.join(_spearman_autocorr(df_r, lag))

            # 3) Top‑k persistence grid
            for k in top_k:
                for w in persistence_windows:
                    features = features.join(_top_k_persistence(df_r, k, w))

            return features


        # Example usage (comment out if running inside production pipeline)
        cluster_rank_query = "SELECT * FROM `issachar-feature-library.qjg.skew-lowest-abstraction`"
        cluster_rank_df = pd.read_gbq(cluster_rank_query, project_id="issachar-feature-library", use_bqstorage_api=True)
        rank_features = build_features(cluster_rank_df).sort_values("date").shift(1)
        rank_features.index = rank_features.index.tz_localize(None)
        df_long = pd.merge(df_long, rank_features, how="left", on='date')

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

    # def calculate_rolling_correlation(df_long, col1, col2, windows):
    #     """
    #     Calculate rolling correlation between two columns of the input DataFrame,
    #     grouped by 'portfolio_id', for multiple rolling window sizes.
    #     """
    #     features = {}

    #     for window in windows:
    #         # Compute the rolling correlation for each portfolio group
    #         rolling_corr_series = df_long.groupby('portfolio_id').apply(
    #             lambda group: group[col1].rolling(window, min_periods=window).corr(group[col2])
    #         )
    #         # Reset the multi-index (created by groupby.apply) so that it aligns with the original DataFrame
    #         rolling_corr_series = rolling_corr_series.reset_index(level=0, drop=True)
    #         # Define the output column name
    #         col_name = f'rolling_corr_{col1}_{col2}_{window}'
    #         features[col_name] = rolling_corr_series

    #         print(features)
    #         print(type(features))
    #         print(features[col_name])
    #         print(type(features[col_name]))
    #     return pd.DataFrame(features)

    def calculate_rolling_correlation(df_long, col1, col2, windows):
        """
        Calculate rolling correlation between two columns of the input DataFrame,
        grouped by 'portfolio_id', for multiple rolling window sizes.
        """
        # Create an empty DataFrame with the same index as df_long
        result_df = pd.DataFrame(index=df_long.index)

        for window in windows:
            # Create a temporary column name for the rolling correlation
            col_name = f'rolling_corr_{col1}_{col2}_{window}'

            # Initialize an empty Series with the same index as df_long
            rolling_corr = pd.Series(index=df_long.index, dtype=float)

            # Calculate rolling correlation for each portfolio separately
            for portfolio_id, group in df_long.groupby('portfolio_id'):
                # Calculate rolling correlation for this group
                group_corr = group[col1].rolling(window, min_periods=window).corr(group[col2])

                # Assign the correlation values to the corresponding indices in the full series
                rolling_corr.loc[group.index] = group_corr

            # Add the correlation series as a new column to the result DataFrame
            result_df[col_name] = rolling_corr

        return result_df

    # def calculate_rolling_corr_diffs(df_long, col1, col2, windows):
    #     """
    #     Calculate rolling correlations between two columns (col1 and col2) for multiple window lengths,
    #     along with:
    #     - Day-to-day differences of each rolling correlation.
    #     - Cross-window differences between the rolling correlations.
    #     """
    #     features = {}
    #     corr_store = {}

    #     # 1. Compute rolling correlations and their day-to-day differences for each window.
    #     for window in windows:
    #         # Compute rolling correlation per portfolio group.
    #         rolling_corr_series = df_long.groupby('portfolio_id').apply(
    #             lambda group: group[col1].rolling(window, min_periods=window).corr(group[col2])
    #         )
    #         # Reset the multi-index to align with the original DataFrame.
    #         rolling_corr_series = rolling_corr_series.reset_index(level=0, drop=True)
    #         col_name = f'rolling_corr_{col1}_{col2}_{window}'
    #         features[col_name] = rolling_corr_series
    #         # Store for later cross-window diff calculation.
    #         corr_store[window] = rolling_corr_series
    #         # Day-to-day difference of this rolling correlation.
    #         features[f'{col_name}_diff'] = rolling_corr_series.groupby(df_long['portfolio_id']).diff()

    #     # 2. Compute cross-window differences: for each pair of windows, subtract the corresponding correlations.
    #     for i, w1 in enumerate(windows):
    #         for w2 in windows[i+1:]:
    #             diff_name = f'rolling_corr_diff_{col1}_{col2}_{w1}_{w2}'
    #             features[diff_name] = (corr_store[w1] - corr_store[w2])


    #     # print(features)
    #     # print(type(features))
    #     # print(features['rolling_corr_diff_fact_lag1_short_momentum_fact_lag1_long_momentum_63_126'])
    #     # print(type(features['rolling_corr_diff_fact_lag1_short_momentum_fact_lag1_long_momentum_63_126']))
    #     print(col1, col2)
    #     return pd.DataFrame(features)

    def calculate_rolling_corr_diffs(df_long, col1, col2, windows):
        """
        Calculate rolling correlations between two columns (col1 and col2) for multiple window lengths,
        along with:
        - Day-to-day differences of each rolling correlation.
        - Cross-window differences between the rolling correlations.
        """
        # Create an empty DataFrame with the same index as df_long
        result_df = pd.DataFrame(index=df_long.index)
        corr_store = {}

        # 1. Compute rolling correlations and their day-to-day differences for each window.
        for window in windows:
            # Initialize an empty Series with the same index as df_long
            rolling_corr = pd.Series(index=df_long.index, dtype=float)

            # Calculate rolling correlation for each portfolio separately
            for portfolio_id, group in df_long.groupby('portfolio_id'):
                # Calculate rolling correlation for this group
                group_corr = group[col1].rolling(window, min_periods=window).corr(group[col2])

                # Assign the correlation values to the corresponding indices in the full series
                rolling_corr.loc[group.index] = group_corr

            # Define column name and add to result DataFrame
            col_name = f'rolling_corr_{col1}_{col2}_{window}'
            result_df[col_name] = rolling_corr

            # Store for later cross-window diff calculation
            corr_store[window] = rolling_corr

            # Day-to-day difference of this rolling correlation
            # We need to group by portfolio_id to avoid calculating differences across different portfolios
            day_diff = pd.Series(index=df_long.index, dtype=float)
            for portfolio_id, group in df_long.groupby('portfolio_id'):
                portfolio_indices = group.index
                portfolio_corr = rolling_corr.loc[portfolio_indices]
                day_diff.loc[portfolio_indices] = portfolio_corr.diff()

            result_df[f'{col_name}_diff'] = day_diff

        # 2. Compute cross-window differences: for each pair of windows, subtract the corresponding correlations.
        for i, w1 in enumerate(windows):
            for w2 in windows[i+1:]:
                diff_name = f'rolling_corr_diff_{col1}_{col2}_{w1}_{w2}'
                result_df[diff_name] = corr_store[w1] - corr_store[w2]

        print(col1, col2)
        return result_df

    # def calculate_rolling_corr_diffs_with_rsi_vol(df_long, col1, col2, windows,
    #                                             rsi_windows=[14, 21, 30, 50],
    #                                             vol_window=20):
    #     """
    #     Calculate rolling correlations between two columns (col1 and col2) for multiple window lengths,
    #     along with:
    #     - Day-to-day differences for each rolling correlation.
    #     - Cross-window differences between the rolling correlations.
    #     - RSI values for each correlation series (using specified RSI windows).
    #     - Rolling volatility (standard deviation) for each correlation series.
    #     """
    #     features = {}
    #     corr_store = {}

    #     # Helper: Compute RSI for a given series and window.
    #     def _compute_rsi(series, window):
    #         delta = series.diff()
    #         gain = delta.where(delta > 0, 0).rolling(window, min_periods=window).mean()
    #         loss = -delta.where(delta < 0, 0).rolling(window, min_periods=window).mean()
    #         rs = gain / loss
    #         rsi = 100 - (100.0 / (1.0 + rs))
    #         return rsi

    #     # 1. Compute rolling correlations and their day-to-day differences for each window.
    #     for window in windows:
    #         # Compute rolling correlation for each portfolio.
    #         rolling_corr_series = df_long.groupby('portfolio_id').apply(
    #             lambda group: group[col1].rolling(window, min_periods=window).corr(group[col2])
    #         )
    #         # Reset multi-index (from groupby.apply) so that it aligns with df_long.
    #         rolling_corr_series = rolling_corr_series.reset_index(level=0, drop=True)
    #         corr_col_name = f'rolling_corr_{col1}_{col2}_{window}'
    #         features[corr_col_name] = rolling_corr_series
    #         corr_store[window] = rolling_corr_series

    #         # Day-to-day difference for the rolling correlation.
    #         features[f'{corr_col_name}_diff'] = rolling_corr_series.groupby(df_long['portfolio_id']).diff()

    #     # 2. Compute cross-window differences: correlation(window1) - correlation(window2)
    #     for i, w1 in enumerate(windows):
    #         for w2 in windows[i+1:]:
    #             diff_name = f'rolling_corr_diff_{col1}_{col2}_{w1}_{w2}'
    #             features[diff_name] = corr_store[w1] - corr_store[w2]

    #     # 3. Compute RSI and volatility for each correlation series (but not for the diffs).
    #     for window in windows:
    #         corr_series = corr_store[window]
    #         base_name = f'rolling_corr_{col1}_{col2}_{window}'

    #         # Compute RSI for each specified RSI window.
    #         for rsi_w in rsi_windows:
    #             rsi_series = corr_series.groupby(df_long['portfolio_id']) \
    #                                     .transform(lambda x: _compute_rsi(x, rsi_w))
    #             features[f'{base_name}_rsi_{rsi_w}'] = rsi_series

    #         # Compute rolling volatility for the correlation series.
    #         vol_series = corr_series.groupby(df_long['portfolio_id']) \
    #                                 .transform(lambda x: x.rolling(vol_window, min_periods=vol_window).std())
    #         features[f'{base_name}_volatility_{vol_window}'] = vol_series

    #     return pd.DataFrame(features)

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
        # Create an empty DataFrame with the same index as df_long
        result_df = pd.DataFrame(index=df_long.index)
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
            # Initialize an empty Series with the same index as df_long
            rolling_corr = pd.Series(index=df_long.index, dtype=float)

            # Calculate rolling correlation for each portfolio separately
            for portfolio_id, group in df_long.groupby('portfolio_id'):
                # Calculate rolling correlation for this group
                group_corr = group[col1].rolling(window, min_periods=window).corr(group[col2])

                # Assign the correlation values to the corresponding indices in the full series
                rolling_corr.loc[group.index] = group_corr

            # Define column name and add to result DataFrame
            corr_col_name = f'rolling_corr_{col1}_{col2}_{window}'
            result_df[corr_col_name] = rolling_corr

            # Store for later cross-window diff calculation
            corr_store[window] = rolling_corr

            # Day-to-day difference of this rolling correlation
            day_diff = pd.Series(index=df_long.index, dtype=float)
            for portfolio_id, group in df_long.groupby('portfolio_id'):
                portfolio_indices = group.index
                portfolio_corr = rolling_corr.loc[portfolio_indices]
                day_diff.loc[portfolio_indices] = portfolio_corr.diff()

            result_df[f'{corr_col_name}_diff'] = day_diff

        # 2. Compute cross-window differences: correlation(window1) - correlation(window2)
        for i, w1 in enumerate(windows):
            for w2 in windows[i+1:]:
                diff_name = f'rolling_corr_diff_{col1}_{col2}_{w1}_{w2}'
                result_df[diff_name] = corr_store[w1] - corr_store[w2]

        # 3. Compute RSI and volatility for each correlation series (but not for the diffs).
        for window in windows:
            corr_series = corr_store[window]
            base_name = f'rolling_corr_{col1}_{col2}_{window}'

            # Compute RSI for each specified RSI window.
            for rsi_w in rsi_windows:
                # Calculate RSI for each portfolio group separately
                rsi_values = pd.Series(index=df_long.index, dtype=float)
                for portfolio_id, group in df_long.groupby('portfolio_id'):
                    portfolio_indices = group.index
                    portfolio_corr = corr_series.loc[portfolio_indices]
                    portfolio_rsi = _compute_rsi(portfolio_corr, rsi_w)
                    rsi_values.loc[portfolio_indices] = portfolio_rsi

                result_df[f'{base_name}_rsi_{rsi_w}'] = rsi_values

            # Compute rolling volatility for the correlation series.
            vol_values = pd.Series(index=df_long.index, dtype=float)
            for portfolio_id, group in df_long.groupby('portfolio_id'):
                portfolio_indices = group.index
                portfolio_corr = corr_series.loc[portfolio_indices]
                portfolio_vol = portfolio_corr.rolling(vol_window, min_periods=vol_window).std()
                vol_values.loc[portfolio_indices] = portfolio_vol

            result_df[f'{base_name}_volatility_{vol_window}'] = vol_values

        return result_df

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
            load_job = client.load_table_from_dataframe(
                df,
                table_id,
                job_config=job_config,
                num_retries=20,
                timeout=5,
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
            load_job = client.load_table_from_dataframe(
                df, table_id, job_config=job_config
            )
            load_job.result()  # Wait for the load job to complete.
            logging.info(f"Created or replaced table {table_id} successfully.")
        except Exception as e:
            logging.error(f"Failed to create or replace table {table_id}: {str(e)}")
            raise

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


    #################################################################################
    # 3. Train Meta Model based on the features and returns
    #################################################################################

    CORR_THRESHOLD  = 0.97
    MI_THRESHOLD    = 0.001
    RANDOM_STATE = 42

    from sklearn.feature_selection import mutual_info_regression
    from sklearn.preprocessing import StandardScaler

    def correlation_prune(df: pd.DataFrame, thresh: float = CORR_THRESHOLD) -> pd.DataFrame:
        corr = df.corr().abs()
        drop = {corr.columns[i] for i in range(len(corr)) for j in range(i) if corr.iloc[i, j] > thresh}
        return df.drop(columns=list(drop))


    def mi_screen(X: pd.DataFrame, y: pd.Series, thresh: float = MI_THRESHOLD) -> pd.DataFrame:
        mi = mutual_info_regression(X, y, random_state=RANDOM_STATE)
        return X.loc[:, mi > thresh]


    # 1) load the PCA exposures
    pca_df = pd.read_excel("PCA Exposures.xlsx")
    pca_df['date'] = pd.to_datetime(pca_df['date']).dt.tz_localize(None)
    pca_df = pca_df.sort_values("date").fillna(0)

    # 2) choose your windows
    windows = [20, 60, 120]

    orig_pcs = [c for c in pca_df.columns if c.startswith("PC_")]

    # 3) helper: for a Series, return its rolling decile (1–10) at each point
    def rolling_decile(s: pd.Series) -> pd.Series:
        return s.expanding().apply(
            # for each window x, compute:
            #  - decile=1 if it's the only value
            #  - else ceil(percentile_rank_of_last * 10)
            lambda x: 1 if x.shape[0] == 1
                    else int(np.ceil(x.rank(pct=True).iloc[-1] * 10)),
            raw=False
        ).astype(int)

    # 4) apply to each PC
    for col in orig_pcs:
        pca_df[f"{col}_decile"] = rolling_decile(pca_df[col])

    # 3) for each window & each PC, compute the rolling z-score at each date
    for w in windows:
        roll = pca_df[orig_pcs].rolling(window=w, min_periods=1)
        means = roll.mean()
        stds  = roll.std(ddof=0)
        for col in orig_pcs:
            pca_df[f"{col}_z_{w}d"] = (pca_df[col] - means[col]) / stds[col]

    df_long = df_long.sort_values("date")
    df_long = pd.merge(df_long, pca_df, on='date', how='left')

    # 2) Add the fractional differentiation features
    from statsmodels.tsa.stattools import adfuller

    class FractionalDifferentiation:
        """
        FractionalDifferentiation class encapsulates the functions that can
        be used to compute fractionally differentiated series.
        """

        @staticmethod
        def get_weights(diff_amt, size):

            # The algorithm below executes the iterative estimation (section 5.4.2, page 78)
            weights = [1.]  # create an empty list and initialize the first element with 1.
            for k in range(1, size):
                weights_ = -weights[-1] * (diff_amt - k + 1) / k  # compute the next weight
                weights.append(weights_)

            # Now, reverse the list, convert into a numpy column vector
            weights = np.array(weights[::-1]).reshape(-1, 1)
            return weights

        @staticmethod
        def frac_diff(series, diff_amt, thresh=0.01):

            # 1. Compute weights for the longest series
            weights = get_weights(diff_amt, series.shape[0])

            # 2. Determine initial calculations to be skipped based on weight-loss threshold
            weights_ = np.cumsum(abs(weights))
            weights_ /= weights_[-1]
            skip = weights_[weights_ > thresh].shape[0]

            # 3. Apply weights to values
            output_df = {}
            for name in series.columns:
                series_f = series[[name]].fillna(method='ffill').dropna()
                output_df_ = pd.Series(index=series.index, dtype='float64')

                for iloc in range(skip, series_f.shape[0]):
                    loc = series_f.index[iloc]

                    # At this point all entries are non-NAs so no need for the following check
                    # if np.isfinite(series.loc[loc, name]):
                    output_df_[loc] = np.dot(weights[-(iloc + 1):, :].T, series_f.loc[:loc])[0, 0]

                output_df[name] = output_df_.copy(deep=True)
            output_df = pd.concat(output_df, axis=1)
            return output_df

        @staticmethod
        def get_weights_ffd(diff_amt, thresh, lim):

            weights = [1.]
            k = 1

            # The algorithm below executes the iterativetive estimation (section 5.4.2, page 78)
            # The output weights array is of the indicated length (specified by lim)
            ctr = 0
            while True:
                # compute the next weight
                weights_ = -weights[-1] * (diff_amt - k + 1) / k

                if abs(weights_) < thresh:
                    break

                weights.append(weights_)
                k += 1
                ctr += 1
                if ctr == lim - 1:  # if we have reached the size limit, exit the loop
                    break

            # Now, reverse the list, convert into a numpy column vector
            weights = np.array(weights[::-1]).reshape(-1, 1)
            return weights

        @staticmethod
        def frac_diff_ffd(series, diff_amt, thresh=1e-5):

            # 1) Compute weights for the longest series
            weights = get_weights_ffd(diff_amt, thresh, series.shape[0])
            width = len(weights) - 1

            # 2) Apply weights to values
            # 2.1) Start by creating a dictionary to hold all the fractionally differenced series
            output_df = {}

            # 2.2) compute fractionally differenced series for each stock
            for name in series.columns:
                series_f = series[[name]].fillna(method='ffill').dropna()
                temp_df_ = pd.Series(index=series.index, dtype='float64')
                for iloc1 in range(width, series_f.shape[0]):
                    loc0 = series_f.index[iloc1 - width]
                    loc1 = series.index[iloc1]

                    # At this point all entries are non-NAs, hence no need for the following check
                    # if np.isfinite(series.loc[loc1, name]):
                    temp_df_[loc1] = np.dot(weights.T, series_f.loc[loc0:loc1])[0, 0]

                output_df[name] = temp_df_.copy(deep=True)

            # transform the dictionary into a data frame
            output_df = pd.concat(output_df, axis=1)
            return output_df


    def get_weights(diff_amt, size):
        """ This is a pass-through function """
        return FractionalDifferentiation.get_weights(diff_amt, size)


    def frac_diff(series, diff_amt, thresh=0.01):
        """ This is a pass-through function """
        return FractionalDifferentiation.frac_diff(series, diff_amt, thresh)


    def get_weights_ffd(diff_amt, thresh, lim):
        """ This is a pass-through function """
        return FractionalDifferentiation.get_weights_ffd(diff_amt, thresh, lim)


    def frac_diff_ffd(series, diff_amt, thresh=1e-5):
        return FractionalDifferentiation.frac_diff_ffd(series, diff_amt, thresh)

    def plot_min_ffd(series):

        results = pd.DataFrame(columns=['adfStat', 'pVal', 'lags', 'nObs', '95% conf', 'corr'])

        # Iterate through d values with 0.1 step
        for d_value in np.linspace(0, 1, 11):
            close_prices = np.log(series[['close']]).resample('1D').last()  # Downcast to daily obs
            close_prices.dropna(inplace=True)

            # Applying fractional differentiation
            differenced_series = frac_diff_ffd(close_prices, diff_amt=d_value, thresh=0.01).fillna(0)

            # Correlation between the original and the differentiated series
            corr = np.corrcoef(close_prices.loc[differenced_series.index, 'close'],
                            differenced_series['close'])[0, 1]
            # Applying ADF
            differenced_series = adfuller(differenced_series['close'], maxlag=1, regression='c', autolag=None)

            # Results to dataframe
            results.loc[d_value] = list(differenced_series[:4]) + [differenced_series[4]['5%']] + [corr]  # With critical value

        # Plotting
        plot = results[['adfStat', 'corr']].plot(secondary_y='adfStat', figsize=(10, 8))
        plt.axhline(results['95% conf'].mean(), linewidth=1, color='r', linestyle='dotted')

        return plot

    series = pd.DataFrame({"close":df_long["lag1_price"].values}, index=df_long["date"].values).iloc[1:]
    ds = [0.3, 0.5, 0.7]
    features = pd.concat(
        [frac_diff_ffd(series[['close']], diff_amt=d, thresh=0.01)
            .rename(columns=lambda c: f"{c}_fd{d}")
        for d in ds],
        axis=1
    ).dropna()
    features.index.name = 'date'

    df_long = pd.merge(df_long, features, on='date', how='left')

    # 3) Add the features to the main dataframe

    df_long['positive_return'] = df_long['returns'].apply(lambda x: 0 if x < 0 else 1)
    # df_long['positive_return'] = df_long['returns'].shift(-4).apply(lambda x: 0 if x < 0 else 1

    X = df_long[feature_columns]
    y = df_long["returns"]

    # -------------------- 2. Chronological train / test split --------
    split = int(len(X)*0.8)                          # 80 / 20 time‑ordered split
    X_train, X_test = X.iloc[:split],  X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # ─── 4. Scale ────────────────────────────────────────────────────────────────
    scaler = StandardScaler().fit(X_train)
    X_train = pd.DataFrame(
        scaler.transform(X_train),
        index=X_train.index,
        columns=X_train.columns
    )
    X_test = pd.DataFrame(
        scaler.transform(X_test),
        index=X_test.index,
        columns=X_test.columns
    )

    # ─── 5. Prune on TRAIN (corr → MI) ───────────────────────────────────────────
    X_train = mi_screen(
        correlation_prune(X_train, CORR_THRESHOLD),
        y_train,
        MI_THRESHOLD
    )

    # apply the *same* selected features to TEST
    X_test = X_test[X_train.columns]

    final_columns = X_train.columns.tolist()

    # standardise *after* the split
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test  = scaler.transform(X_test)

    # -------------------- 3.  Base models (all *classification*) -----
    from catboost import CatBoostClassifier
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import (
        ExtraTreesClassifier,
        RandomForestClassifier,
        HistGradientBoostingClassifier,
        AdaBoostClassifier,
        GradientBoostingClassifier,
    )

    from sklearn.linear_model  import (
        LogisticRegression,
        SGDClassifier,
    )
    from sklearn.svm           import SVC
    from sklearn.neighbors     import KNeighborsClassifier
    from sklearn.naive_bayes   import GaussianNB
    from pytorch_tabnet.tab_model import TabNetClassifier
    from imblearn.ensemble     import BalancedBaggingClassifier
    import torch

    if l0_config == 'base':
        models = {
                "XGBoost": XGBClassifier(
            objective="multi:softprob",    # multiclass probability output
            num_class=3,                   # three target labels
            n_estimators=1000,
            learning_rate=0.01,
            tree_method="hist",
            # use_label_encoder=False,
            eval_metric="mlogloss",        # multiclass log‐loss
            random_state=42,
            verbosity=0,
            device="cpu"
        ),
        # "CatBoost": CatBoostClassifier(
        #     loss_function="MultiClass",   # multiclass loss
        #     random_seed=42,
        #     task_type="CPU",
        #     depth=8,
        #     learning_rate=0.03,
        #     iterations=800,
        #     verbose=False
        # ),
        # If you decide to re-enable LightGBM:
        # "LightGBM": LGBMClassifier(
        #     objective="multiclass",
        #     num_class=3,
        #     boosting_type="dart",
        #     learning_rate=0.02,
        #     drop_rate=0.2,
        #     subsample=0.8,
        #     feature_fraction=0.8,
        #     max_depth=-1,
        #     device="cpu",
        #     seed=42,
        #     n_estimators=800,
        #     metric="multi_logloss"
        # ),
        # "XGBoost2": XGBClassifier(
        #     objective="multi:softprob",
        #     num_class=3,
        #     n_estimators=2000,
        #     learning_rate=0.01,
        #     max_depth=8,
        #     min_child_weight=5,
        #     gamma=1.0,
        #     subsample=0.8,
        #     colsample_bytree=0.8,
        #     reg_alpha=0.1,
        #     reg_lambda=1.0,
        #     tree_method="hist",
        #     # use_label_encoder=False,
        #     eval_metric="mlogloss",
        #     random_state=42,
        #     verbosity=1,
        #     device="cpu"
        # ),
        # "XGBoost3": XGBClassifier(
        #     objective="multi:softprob",
        #     num_class=3,
        #     n_estimators=2000,
        #     learning_rate=0.001,
        #     max_depth=18,
        #     min_child_weight=5,
        #     gamma=1.0,
        #     subsample=0.9,
        #     colsample_bytree=0.9,
        #     reg_alpha=0.1,
        #     reg_lambda=1.0,
        #     tree_method="hist",
        #     # use_label_encoder=False,
        #     eval_metric="mlogloss",
        #     random_state=42,
        #     verbosity=1,
        #     device="cpu"
        # ),
    }

    elif l0_config == 'base++':
        models = {
            # 1) Gradient-boosted trees (oblivious)
            "CatBoost": CatBoostClassifier(
                loss_function      = "Logloss",
                eval_metric        = "AUC",
                depth              = 6,              # shallower → less over-fit
                l2_leaf_reg        = 3.0,            # stronger regularisation
                learning_rate      = 0.03,
                bagging_temperature= 1.0,            # Bayesian bootstrap
                random_strength    = 0.8,
                subsample          = 0.8,
                iterations         = 2000,           # early-stop inside .fit()
                od_type            = "Iter",
                od_wait            = 100,
                random_seed        = RANDOM_STATE,
                verbose            = False,
                task_type          = "CPU"
            ),

            # 2) Gradient-boosting with GOSS sampling
            # "LightGBM_GOSS": LGBMClassifier(
            #     boosting_type      = "goss",
            #     objective          = "binary",
            #     metric             = "auc",
            #     num_leaves         = 63,             # ≈ 2^(max_depth)
            #     max_depth          = -1,
            #     learning_rate      = 0.015,
            #     n_estimators       = 2500,
            #     feature_fraction   = 0.9,
            #     bagging_fraction   = 0.9,
            #     bagging_freq       = 0,
            #     min_child_samples  = 20,
            #     lambda_l1          = 1e-2,
            #     lambda_l2          = 1e-1,
            #     random_state       = RANDOM_STATE,
            #     device             = "cpu"
            # ),

            # 3) XGBoost histogram
            "XGBoost_hist": XGBClassifier(
                objective          = "binary:logistic",
                eval_metric        = "auc",
                tree_method        = "hist",
                learning_rate      = 0.02,
                n_estimators       = 1500,
                max_depth          = 6,
                min_child_weight   = 1.0,
                subsample          = 0.8,
                colsample_bytree   = 0.8,
                gamma              = 0.5,
                reg_alpha          = 0.05,
                reg_lambda         = 1.0,
                random_state       = RANDOM_STATE,
                verbosity          = 0,
                device             = "cpu"
            ),

            # 4) Extremely-Randomised Trees (high-variance, high-diversity)
            "ExtraTrees": ExtraTreesClassifier(
                n_estimators       = 800,
                max_depth          = None,
                min_samples_split  = 4,
                min_samples_leaf   = 2,
                max_features       = "sqrt",
                criterion          = "entropy",
                bootstrap          = False,
                random_state       = RANDOM_STATE,
                n_jobs             = -1
            ),

            # 5) Elastic-Net logistic regression (linear baseline)
            "LogReg_EN": LogisticRegression(
                penalty            = "elasticnet",
                solver             = "saga",
                l1_ratio           = 0.2,
                C                  = 1.0,
                max_iter           = 20_000,
                n_jobs             = -1,
                random_state       = RANDOM_STATE
            ),
            "CatBoost" : CatBoostClassifier(
                    loss_function="Logloss",
                    random_seed=42,
                    task_type="CPU",
                    depth=8,
                    learning_rate=0.03,
                    iterations=800,
                    verbose=False
                ),
                # "LightGBM" : LGBMClassifier(
                #     objective="binary",
                #     boosting_type="dart",
                #     learning_rate=0.02,
                #     drop_rate=0.2,
                #     subsample=0.8,
                #     feature_fraction=0.8,
                #     max_depth=-1,
                #     device="cpu",
                #     seed=42,
                #     n_estimators=800
                # ),
                "XGBoost": XGBClassifier(
                    objective="binary:logistic",
                    n_estimators=1000,
                    learning_rate=0.01,
                    tree_method="hist",       # GPU‑accelerated histogram algorithm
                    # predictor="gpu_predictor",    # GPU for prediction, too
                    # gpu_id=0,                     # which GPU to use
                    # use_label_encoder=False,      # disable legacy label encoder
                    verbosity=0,                  # silent
                    random_state=42,
                    device="cpu"
                ),
                "XGBoost2": XGBClassifier(
                objective="binary:logistic",
                # -------------------
                # Core boosting
                n_estimators=2000,          # more rounds to compensate for stronger regularization
                learning_rate=0.01,         # slightly higher than 0.005 to converge faster
                # -------------------
                # Tree complexity
                max_depth=8,                # capture richer patterns, but not too deep
                min_child_weight=5,         # minimum sum hessian in leaf to avoid over‑fitting small samples
                gamma=1.0,                  # require a 1.0 loss reduction to make a split
                # -------------------
                # Randomness for bagging
                subsample=0.8,              # row subsampling per tree
                colsample_bytree=0.8,       # feature subsampling per tree
                # -------------------
                # Regularization
                reg_alpha=0.1,              # L1 regularization
                reg_lambda=1.0,             # L2 regularization
                # -------------------
                # GPU settings
                tree_method="hist",
                # predictor="gpu_predictor",
                # gpu_id=0,
                # -------------------
                # Misc
                # use_label_encoder=False,
                eval_metric="auc",          # optimize for area‑under‑ROC
                random_state=42,
                verbosity=1,
                device="cpu"
            ),
                "XGBoost3": XGBClassifier(
                objective="binary:logistic",
                # -------------------
                # Core boosting
                n_estimators=2000,          # more rounds to compensate for stronger regularization
                learning_rate=0.001,         # slightly higher than 0.005 to converge faster
                # -------------------
                # Tree complexity
                max_depth=18,                # capture richer patterns, but not too deep
                min_child_weight=5,         # minimum sum hessian in leaf to avoid over‑fitting small samples
                gamma=1.0,                  # require a 1.0 loss reduction to make a split
                # -------------------
                # Randomness for bagging
                subsample=0.9,              # row subsampling per tree
                colsample_bytree=0.9,       # feature subsampling per tree
                # -------------------
                # Regularization
                reg_alpha=0.1,              # L1 regularization
                reg_lambda=1.0,             # L2 regularization
                # -------------------
                # GPU settings
                tree_method="hist",
                # predictor="gpu_predictor",
                # gpu_id=0,
                # -------------------
                # Misc
                # use_label_encoder=False,
                eval_metric="auc",          # optimize for area‑under‑ROC
                random_state=42,
                verbosity=1,
                device="cpu"
            ),
        }

        from pytorch_tabnet.tab_model import TabNetClassifier
        import torch

        models["TabNet"] = TabNetClassifier(
            # Model capacity
            n_d               = 32,        # decision layer width
            n_a               = 32,        # attention layer width
            n_steps           = 5,
            n_independent     = 2,
            n_shared          = 2,
            gamma             = 1.5,       # feature-re-use penalty

            # Regularisation / sparsity
            lambda_sparse     = 1e-4,
            mask_type         = "entmax",  # smoother sparsity

            # Optimiser
            optimizer_fn      = torch.optim.Adam,
            optimizer_params  = dict(lr=2e-3, weight_decay=1e-5),

            # Learning-rate schedule
            scheduler_fn      = torch.optim.lr_scheduler.StepLR,
            scheduler_params  = dict(step_size=50, gamma=0.9),

            # Misc
            seed              = RANDOM_STATE,
            verbose           = 10,
            device_name       = "cpu"      # switch to "cuda" if available
        )

    elif l0_config == 'v2':
        # ─── Imports ──────────────────────────────────────────────────────

        # helper: class imbalance weight (≈1 if perfectly balanced)
        pos_weight = (1 - y_train.mean()) / y_train.mean()

        # ------------------------------------------------------------------
        models = {
            # ────────────────────────────────────────────────────────────
            # Gradient-boosting family
            # ────────────────────────────────────────────────────────────
            "CatBoost": CatBoostClassifier(
                loss_function       = "Logloss",
                eval_metric         = "AUC",
                depth               = 8,
                learning_rate       = 0.02,
                iterations          = 5000,
                grow_policy         = "Lossguide",
                l2_leaf_reg         = 6.0,
                subsample           = 0.65,
                random_strength     = 1.2,
                bagging_temperature = 2.0,
                auto_class_weights  = "Balanced",
                od_type             = "Iter",
                od_wait             = 300,
                random_seed         = RANDOM_STATE,
                verbose             = False,
                task_type           = "CPU"
            ),

            # "LightGBM_DART": LGBMClassifier(
            #     boosting_type       = "dart",
            #     objective           = "binary",
            #     metric              = "auc",
            #     num_leaves          = 255,
            #     learning_rate       = 0.007,
            #     n_estimators        = 6000,
            #     feature_fraction    = 0.85,
            #     bagging_fraction    = 0.8,
            #     drop_rate           = 0.05,
            #     skip_drop           = 0.4,
            #     min_child_samples   = 25,
            #     lambda_l1           = 5e-3,
            #     lambda_l2           = 5e-2,
            #     scale_pos_weight    = pos_weight,
            #     random_state        = RANDOM_STATE,
            #     device              = "cpu"
            # ),

            "XGBoost_hist": XGBClassifier(
                objective           = "binary:logistic",
                eval_metric         = "auc",
                tree_method         = "hist",
                learning_rate       = 0.01,
                n_estimators        = 8000,
                max_depth           = 6,
                min_child_weight    = 2.0,
                subsample           = 0.7,
                colsample_bytree    = 0.8,
                gamma               = 0.2,
                reg_alpha           = 0.1,
                reg_lambda          = 1.2,
                scale_pos_weight    = pos_weight,
                random_state        = RANDOM_STATE,
                verbosity           = 0,
                device              = "cpu"
            ),

            "HistGB": HistGradientBoostingClassifier(
                loss                = "log_loss",
                max_depth           = 6,
                learning_rate       = 0.03,
                max_iter            = 2000,
                l2_regularization   = 0.1,
                early_stopping      = True,
                validation_fraction = 0.1,
                random_state        = RANDOM_STATE,
                class_weight        = {0:1, 1:pos_weight}
            ),

            "Scikit_GB": GradientBoostingClassifier(
                learning_rate       = 0.05,
                n_estimators        = 1500,
                max_depth           = 3,
                subsample           = 0.7,
                random_state        = RANDOM_STATE
            ),

            "AdaBoost": AdaBoostClassifier(
                n_estimators        = 1500,
                learning_rate       = 0.01,
                random_state        = RANDOM_STATE
            ),

            # ────────────────────────────────────────────────────────────
            # Bagging / Randomised trees
            # ────────────────────────────────────────────────────────────
            "ExtraTrees": ExtraTreesClassifier(
                n_estimators        = 1500,
                max_depth           = None,
                min_samples_split   = 4,
                min_samples_leaf    = 2,
                max_features        = "sqrt",
                criterion           = "entropy",
                bootstrap           = False,
                class_weight        = "balanced",
                random_state        = RANDOM_STATE,
                n_jobs              = -1
            ),

            "BalancedRF": RandomForestClassifier(
                n_estimators        = 1200,
                max_depth           = None,
                min_samples_split   = 3,
                min_samples_leaf    = 1,
                max_features        = "sqrt",
                class_weight        = "balanced_subsample",
                random_state        = RANDOM_STATE,
                n_jobs              = -1
            ),

            "BalancedBagging_DT": BalancedBaggingClassifier(          # pip install imbalanced-learn
                n_estimators        = 800,
                estimator           = None,   # default = DecisionTree
                sampling_strategy    = "auto",
                replacement          = False,
                random_state         = RANDOM_STATE,
                n_jobs              = -1
            ),

            # ────────────────────────────────────────────────────────────
            # Linear / kernel / instance-based
            # ────────────────────────────────────────────────────────────
            "LogReg_EN": LogisticRegression(
                penalty             = "elasticnet",
                solver              = "saga",
                l1_ratio            = 0.25,
                C                   = 0.8,
                max_iter            = 25_000,
                class_weight        = "balanced",
                n_jobs              = -1,
                random_state        = RANDOM_STATE
            ),

            "SGD_Log": SGDClassifier(          # stochastic logistic regression
                loss                = "log_loss",      # enables predict_proba
                alpha               = 1e-4,
                max_iter            = 15_000,
                class_weight        = "balanced",
                random_state        = RANDOM_STATE
            ),

            "SVC_RBF": SVC(
                C                   = 2.0,
                gamma               = "scale",
                probability         = True,
                class_weight        = "balanced",
                random_state        = RANDOM_STATE
            ),

            "KNN": KNeighborsClassifier(
                n_neighbors         = 25,
                weights             = "distance",
                metric              = "minkowski",
                p                   = 2,          # Euclidean
                n_jobs              = -1
            ),

            "GaussianNB": GaussianNB(),

            # ────────────────────────────────────────────────────────────
            # Neural (tabular) models
            # ────────────────────────────────────────────────────────────
            "TabNet": TabNetClassifier(
                n_d                 = 64,
                n_a                 = 64,
                n_steps             = 8,
                gamma               = 1.3,
                lambda_sparse       = 1e-4,
                mask_type           = "entmax",
                optimizer_fn        = torch.optim.Adam,
                optimizer_params    = dict(lr=8e-4, weight_decay=1e-5),
                scheduler_fn        = torch.optim.lr_scheduler.ReduceLROnPlateau,
                scheduler_params    = dict(mode="max", factor=0.5, patience=20),
                seed                = RANDOM_STATE,
                verbose             = 10,
                device_name         = "cpu"          # set "cuda" if available
            ),
        }

    # -------------------- 5.  Classification metrics -----------------
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                f1_score, roc_auc_score, brier_score_loss,
                                confusion_matrix, roc_curve,
                                precision_recall_curve)
    from sklearn.calibration import calibration_curve

    # ——— 1) background for SHAP ——
    background = X_train#.sample(n=100, random_state=42)

    # ——— 2) storage ——
    metrics_list = []
    shap_values_dict = {}   # temporary storage of raw arrays

    import logging
    from typing import Tuple, List, Union, Optional, Dict, Any
    import copy
    from sklearn.preprocessing import label_binarize, LabelEncoder
    from sklearn.base import clone

    def rolling_backtest_period(
        X: pd.DataFrame,
        y_cont: pd.Series,  # 1‑day ahead *continuous* returns
        *,
        initial_train_fraction: float = 0.80,
        period_length: int = 5,  # ⬅️ EXACT *row* count per chunk (trading‑day hop)
        start_test_date: str | pd.Timestamp = "2022-01-01",
        model_cls=KNeighborsClassifier,
        model_kwargs: Optional[Dict[str, Any]] = None,
        log_level: int = logging.INFO,
        neg_thresh: float = -0.002,  # −0.2 %
        pos_thresh: float = 0.002,  # +0.2 %
    ):


        if period_length < 1:
            raise ValueError("period_length must be >= 1 row")

        # ---------- build the period‑ahead target ------------------------------
        period_mean = (
            y_cont.rolling(period_length).mean().shift(-(period_length)).dropna()
        )
        y = (
            pd.cut(
                period_mean,
                bins=[-np.inf, neg_thresh, pos_thresh, np.inf],
                labels=[0, 1, 2],
                include_lowest=True,
            )
            .astype(int)
            .rename("class")
        )

        # ---------- align features ---------------------------------------------
        X = X.loc[y.index]
        if not isinstance(X.index, pd.DatetimeIndex):
            X = X.copy()
            X.index = pd.to_datetime(X.index)
        y.index = X.index

        # ---------- logging setup ----------------------------------------------
        logging.basicConfig(level=log_level, format="[%(asctime)s] %(message)s")
        log = logging.getLogger(__name__)

        # ---------- derive *row‑based* chunk identifiers -----------------------
        n_rows = len(X)
        full_chunks = n_rows // period_length  # ignore trailing partial chunk

        if full_chunks < 2:
            raise ValueError("Not enough data to form at least two full chunks.")

        chunk_id = np.arange(n_rows) // period_length
        chunk_id = chunk_id.astype(int)

        # Mask to discard rows belonging to an incomplete tail chunk (if present)
        valid_mask = chunk_id < full_chunks
        X = X.loc[valid_mask]
        y = y.loc[valid_mask]
        chunk_id = chunk_id[valid_mask]

        chunk_starts = {
            cid: X.index[cid * period_length] for cid in range(full_chunks)
        }

        # ---------- split into initial‑train / rolling sequence ----------------
        split_idx = int(full_chunks * initial_train_fraction)
        start_test_date = pd.to_datetime(start_test_date)

        test_chunks = [
            cid
            for cid in range(split_idx, full_chunks)
            if chunk_starts[cid] >= start_test_date
        ]

        log.info("Class dispersion (entire sample):\n%s", y.value_counts())

        # ---------- instantiate / prepare model --------------------------------
        if isinstance(model_cls, type):
            model = model_cls(**(model_kwargs or {}))
        else:
            model = model_cls

        classes = [0, 1, 2]
        metrics, preds = [], []

        # rows to keep for *rolling* 4‑year look‑back (~252 trading days per yr)
        rows_per_year = 252
        lookback_rows = rows_per_year * 4

        for cid in tqdm(test_chunks, desc=f"Back‑test ({period_length}‑row hop)"):
            train_mask = chunk_id < cid
            test_mask = chunk_id == cid

            X_train = X.loc[train_mask].tail(lookback_rows)
            y_train = y.loc[train_mask].tail(lookback_rows)
            X_test = X.loc[test_mask]
            y_test = y.loc[test_mask]

            if X_test.empty:
                continue  # safety – should not happen with strict chunk sizing

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)

            predicted_classes = pd.Series(y_pred.ravel()).unique()
            missing_classes = set(classes) - set(predicted_classes)
            if (len(missing_classes) >= 1) and (y_proba.shape[1] !=3):
                y_proba = [list(x) for x in y_proba]
                for missed_class in missing_classes:
                    for row in y_proba:
                        row.insert(missed_class,0)
                y_proba = np.array(y_proba)

            # ---- evaluation metrics ------------------------------------------
            y_test_ohe = label_binarize(y_test, classes=classes)

            metrics.append(
                {
                    "period": str(chunk_starts[cid].date()),
                    "n_test": len(y_test),
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(
                        y_test,
                        y_pred,
                        labels=classes,
                        average="macro",
                        zero_division=0,
                    ),
                    "recall": recall_score(
                        y_test,
                        y_pred,
                        labels=classes,
                        average="macro",
                        zero_division=0,
                    ),
                    "f1": f1_score(
                        y_test,
                        y_pred,
                        labels=classes,
                        average="macro",
                        zero_division=0,
                    ),
                    "roc_auc_ovr": roc_auc_score(
                        y_test_ohe, y_proba, multi_class="ovr", average="macro"
                    ),
                    "roc_auc_ovo": roc_auc_score(
                        y_test_ohe, y_proba, multi_class="ovo", average="macro"
                    ),
                }
            )

            print("\n #### Evaluation Metrics")
            print("Missing Classes:", (len(missing_classes) >= 1) and (y_proba.shape[1] !=3))
            print("Predictions Value Counts")
            print(pd.Series(y_pred.ravel()).value_counts())
            print("Accuracy Score", accuracy_score(y_test, y_pred))


            data = {
                "period": [str(chunk_starts[cid].date())] * len(y_pred),
                "prediction": y_pred.ravel(),
                "y_test": y_test.values,
                "y_test_index": list(range(len(y_test))),
                "date": X_test.index,
            }
            for idx, lbl in enumerate(classes):
                data[f"pred_proba_{lbl}"] = y_proba[:, idx]

            preds.append(pd.DataFrame(data))

            log.info(
                "Test chunk starting %s – train rows: %s, test rows: %s",
                chunk_starts[cid].date(),
                len(y_train),
                len(y_test),
            )

        # ---------- aggregate ---------------------------------------------------
        metrics_df = pd.DataFrame(metrics)
        aggregate = metrics_df.drop(columns=["period", "n_test"]).mean().to_dict()
        preds_df = pd.concat(preds).sort_index().reset_index(drop=False)

        return metrics_df, aggregate, preds_df
    
    import psutil
    import re
    from tqdm import tqdm

    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, log_loss
    )

    from sklearn.preprocessing import label_binarize, LabelEncoder

    proba_dfs = []

    # ——— 3) loop over models ——
    for name, mdl in models.items():
        print(f"\n\nCurrent model being trained: {name}")
        model_start = time.time()

        y_temp = df_long["returns"]
        y_temp.index = df_long['date']

        print(f"{psutil.virtual_memory().total/1024**3:.2f} GB")

        metrics_df, agg_df, preds_df = rolling_backtest_period(
            X.set_index(df_long['date']),
            y_temp,
            period_length=rolling_window,                # e.g. "M", "W", "D", etc.
            start_test_date="2022-01-03",
            model_cls=mdl    )

        agg_df['model'] = name

        print(agg_df)

        # — compute metrics dict —
        metrics_list.append(agg_df)
        preds_df["model"] = name
        preds_df["uuid"] = uuid
        preds_df["runtime"] = current_time
        proba_dfs.append(preds_df)
        append_to_bigquery(preds_df, DESTINATION_DATASET, f'model-predictions-{table_suffix}')
        print(f"Time to compute whole model {name}: {time.time() - model_start}")

    all_predictions = pd.concat(proba_dfs)
    all_predictions = pd.concat(proba_dfs)
    all_predictions = all_predictions[all_predictions["y_test_index"] == 0]

    ###################################################################################
    # 4. Evaluate the model performance ‑‑‑‑‑ simple average version
    ###################################################################################

    preds_df = (
        all_predictions
        .groupby(['date'])
        [['pred_proba_0', 'pred_proba_1', 'pred_proba_2']]
        .mean()
        .reset_index()
    ).sort_values("date")

    preds_df['prediction'] = np.vstack([
        preds_df['pred_proba_0'],
        preds_df['pred_proba_1'],
        preds_df['pred_proba_2']
    ]).T.argmax(axis=1)

    # -------------------- 4. Simple equal‑weight ensemble ------------
    preds_df["model"] = "simple average"
    preds_df["uuid"] = uuid
    preds_df["runtime"] = current_time
    preds_df = pd.merge(preds_df, all_predictions[["date", "y_test"]], on="date", how="left").drop_duplicates()
    append_to_bigquery(preds_df, DESTINATION_DATASET, f'model-predictions-{table_suffix}')

    import numpy as np
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                f1_score, roc_auc_score, log_loss,
                                confusion_matrix, roc_curve,
                                precision_recall_curve)
    from sklearn.calibration import calibration_curve
    from sklearn.preprocessing import label_binarize

    # ---------------------------------------------------------------------
    # 1.  Assemble y_test, y_pred and y_proba ------------------------------
    # ---------------------------------------------------------------------
    y_test   = preds_df["y_test"].to_numpy()          # shape (n, )
    y_pred   = preds_df["prediction"].to_numpy()      # shape (n, )

    # Probability columns must be in the same order as `classes`
    prob_cols = [c for c in preds_df.columns if c.startswith("pred_proba_")]
    y_proba   = preds_df[prob_cols].to_numpy()        # shape (n, n_classes)

    classes   = np.arange(y_proba.shape[1])           # e.g. array([0,1,2])
    y_test_ohe = label_binarize(y_test, classes=classes)  # shape (n, n_classes)

    # ---------------------------------------------------------------------
    # 2.  Scalar metrics ---------------------------------------------------
    # ---------------------------------------------------------------------
    accuracy     = accuracy_score(y_test, y_pred)

    precision    = precision_score(y_test, y_pred,
                                labels=classes,
                                average="macro",
                                zero_division=0)

    recall       = recall_score(y_test, y_pred,
                                labels=classes,
                                average="macro",
                                zero_division=0)

    f1           = f1_score(y_test, y_pred,
                            labels=classes,
                            average="macro",
                            zero_division=0)

    try:
        roc_auc_ovr_val = roc_auc_score(
            y_test_ohe, y_proba, multi_class="ovr", average="macro"
        )
        roc_auc_ovo_val = roc_auc_score(
            y_test_ohe, y_proba, multi_class="ovo", average="macro"
        )
    except ValueError:
        roc_auc_ovr_val = np.nan
        roc_auc_ovo_val = np.nan

    logloss      = 0

    print(f"Accuracy     : {accuracy     :.4f}")
    print(f"Precision    : {precision    :.4f}")
    print(f"Recall       : {recall       :.4f}")
    print(f"F1           : {f1           :.4f}")
    print(f"ROC AUC (ovr): {roc_auc_ovr_val  :.4f}")
    print(f"ROC AUC (ovo): {roc_auc_ovo_val  :.4f}")
    print(f"Log‑loss     : {logloss     :.4f}")

    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    # Store to BigQuery as before
    ensemble   = "simple average"
    metrics_df = sanitize_and_convert_columns(
                    pd.DataFrame([{
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "roc_auc_ovr": roc_auc_ovr_val,
                    "roc_auc_ovo": roc_auc_ovo_val,
                    "log_loss": logloss,
                    "ensemble": ensemble,
                    "uuid": uuid,
                    "runtime": current_time
                    }])
                )
    append_to_bigquery(metrics_df, DESTINATION_DATASET,
                    f"meta_model_metrics_{table_suffix}")

    ###################################################################################
    # 6. Diagnostic plots ‑‑‑‑‑ multiclass version
    ###################################################################################
    import matplotlib.pyplot as plt
    import itertools

    # a) Confusion‑matrix heat‑map ---------------------------------------
    cm  = confusion_matrix(y_test, y_pred, labels=classes)
    confusion_matrix_plot = plt.figure(figsize=(6,4))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xticks(classes, classes)
    plt.yticks(classes, classes)
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max()/2 else "black")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.show()

    # b) ROC curves (one‑vs‑rest) ----------------------------------------
    roc_plot = plt.figure(figsize=(6,4))
    for idx, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_test_ohe[:, idx], y_proba[:, idx])
        auc_val     = roc_auc_score(y_test_ohe[:, idx], y_proba[:, idx])
        plt.plot(fpr, tpr, label=f"class {cls} (AUC={auc_val:.3f})")
    plt.plot([0,1], [0,1], linestyle="--", label="Chance")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (one‑vs‑rest)")
    plt.legend(); plt.tight_layout(); plt.show()

    # c) Precision‑Recall curves -----------------------------------------
    precision_recall_plot = plt.figure(figsize=(6,4))
    for idx, cls in enumerate(classes):
        prec, rec, _ = precision_recall_curve(y_test_ohe[:, idx], y_proba[:, idx])
        plt.plot(rec, prec, label=f"class {cls}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision‑Recall Curve"); plt.legend(); plt.tight_layout(); plt.show()

    # d) Calibration curves ----------------------------------------------
    calibration_curve_plot = plt.figure(figsize=(6,4))
    for idx, cls in enumerate(classes):
        prob_true, prob_pred = calibration_curve(y_test_ohe[:, idx],
                                                y_proba[:, idx], n_bins=20)
        plt.plot(prob_pred, prob_true, marker="o", linestyle="-", label=f"class {cls}")
    plt.plot([0,1],[0,1], linestyle="--", color="gray")
    plt.xlabel("Mean Predicted Probability"); plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curves"); plt.legend(); plt.tight_layout(); plt.show()

    # e) Histogram of predicted probabilities ----------------------------
    predicted_probability_plot = plt.figure(figsize=(6,4))
    plt.hist(y_proba, bins=20, edgecolor="k", label=[f"class {c}" for c in classes])
    plt.xlabel("Predicted probability"); plt.ylabel("Frequency")
    plt.title("Distribution of Predicted Probabilities"); plt.legend()
    plt.tight_layout(); plt.show()

    date_str = datetime.now().strftime("%Y-%m-%d")
    file_prefix = f"{date_str}/{uuid+'__'+ensemble}"

    save_performance_figure_to_gcs(confusion_matrix_plot, STAGING_BUCKET, f"{file_prefix}_confusion_matrix.png")
    save_performance_figure_to_gcs(roc_plot, STAGING_BUCKET, f"{file_prefix}_ROC.png")
    save_performance_figure_to_gcs(precision_recall_plot, STAGING_BUCKET, f"{file_prefix}_precision_recall_plot.png")
    save_performance_figure_to_gcs(calibration_curve_plot, STAGING_BUCKET, f"{file_prefix}_calibration_curve.png")
    save_performance_figure_to_gcs(predicted_probability_plot, STAGING_BUCKET, f"{file_prefix}_predicted_probability.png")

if __name__ == "__main__":
    # put any one‑off code or calls to your main() function here
    main()
