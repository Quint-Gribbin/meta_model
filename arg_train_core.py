def main():
    # Import key libraries
    import pandas as pd
    import argparse
    import subprocess
    # from model import main as train_main_function
    from model_core_rolling import main as train_main_function

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

    #####################################################################
    #  1. ARGPARSE & CONFIG
    #####################################################################

    def parse_arguments():
        parser = argparse.ArgumentParser(description='ListFold Portfolio Optimization')

        parser.add_argument('--rolling_train_length',
                            type=int,
                            default=900,
                            help='Length of rolling training window')

        parser.add_argument('--rolling_test_length',
                            type=int,
                            default=1,
                            help='Length of rolling test window')

        parser.add_argument('--n_k',
                            type=int,
                            default=8,
                            help='Number of k values to test')

        parser.add_argument('--holdout_start',
                            type=str,
                            default="2022-01-01",
                            help='Start date for holdout period (YYYY-MM-DD)')

        parser.add_argument('--batch_size',
                            type=int,
                            default=32,
                            help='Batch size for training')

        parser.add_argument('--learning_rate',
                            type=float,
                            default=5e-5,
                            help='Learning rate for optimization')

        parser.add_argument('--optuna_runs',
                            type=int,
                            default=50,
                            help='Number of Optuna optimization trials')

        parser.add_argument('--random_seed',
                            type=int,
                            default=42,
                            help='Random seed for reproducibility')

        parser.add_argument('--gcs_bucket',
                            type=str,
                            default='test_model_plots',
                            help='Name of the GCS bucket where plots will be saved')

        parser.add_argument('--include_fundamentals',
                            type=int,
                            default=0,
                            help='Flag to indicate whether fundamentals should be included (default: False)'
                        )

        parser.add_argument('--include_will_features',
                            type=int,
                            default=0,
                            help='Flag to indicate whether fundamentals should be included (default: False)'
                        )

        parser.add_argument('--notes',
                            type=str,
                            default='Update notes here',
                            help='Notes for this run')

        parser.add_argument('--is-test',
                            type=int,
                            default=0,
                            help='Determines if the run needs to iterate over only 5 days')

        parser.add_argument('--epochs',
                            type=int,
                            default=1000,
                            help='Determines if the run needs to iterate over only 5 days')

        parser.add_argument('--additional-technical',
                            type=int,
                            default=1,
                            help='Expand technical features')

        parser.add_argument('--additional-factors',
                            type=int,
                            default=1,
                            help='Expand the economic factors features')
        
        parser.add_argument('--act-func',
                            type=str,
                            default="swish",
                            help='Flag for using Leaky ReLu')
        
        parser.add_argument('--table-suffx',
                            type=str,
                            default=1,
                            help='The table name to save')
        
        parser.add_argument('--use-correlations',
                            type=int,
                            default=1,
                            help='Calculate and include correlations')
        
        parser.add_argument('--will-portfolios',
                            type=int,
                            default=0,
                            help='Add new portfolios')
        
        parser.add_argument('--calculate-shap',
                            type=int,
                            default=0,
                            help='Calculate and persist SHAP values')
        
        parser.add_argument('--calculate-ww',
                            type=int,
                            default=0,
                            help='Calculate and persist WeightWatcher values')

        parser.add_argument('--clamp-gradients',
                            type=int,
                            default=0,
                            help='Add mechanism to prevent gradients from exploding')
        
        parser.add_argument('--tie-breaks',
                            type=int,
                            default=0,
                            help='Update default allocation logic used to break ties')
        
        parser.add_argument('--will-predictions',
                            type=int,
                            default=0,
                            help='Predicted direction')

        parser.add_argument('--include-coint-regimes',
                            type=int,
                            default=0,
                            help='Predicted direction')       
        
        parser.add_argument('--include-cluster-data',
                            type=int,
                            default=0,
                            help='Predicted direction')     

        parser.add_argument('--include-hmm-regimes',
                            type=int,
                            default=0,
                            help='Predicted direction')
        
        parser.add_argument('--include-skew-kurt',
                            type=int,
                            default=0,
                            help='Skew and kurtoisis')
        
        parser.add_argument('--include-time-features',
                            type=int,
                            default=0,
                            help='Day of week and week of month features')        

        parser.add_argument('--portfolio-returns-table',
                            type=str,
                            default='cluster_portfolio_returns_runtime',
                            help='The table name to save')

        parser.add_argument('--factor-lag',
                            type=int,
                            default=1,
                            help='Day of week and week of month features')      
        
        parser.add_argument('--live-next-day',
                    type=int,
                    default=0,
                    help='Trigger live trading recommendation') 
        
        parser.add_argument('--return-lag',
                    type=int,
                    default=1,
                    help='How long to lag the returns') 
        
        parser.add_argument('--core-model-column',
                type=str,
                default="long_return",
                help='Which returns to select from core model') 

        parser.add_argument('--l0-config',
                type=str,
                default="base",
                help='Which returns to select from core model') 
        
        parser.add_argument('--use-rank-features',
                type=int,
                default=0,
                help='Whether to use rank features or not') 
        
        parser.add_argument('--rolling-window',
                type=int,
                default=20,
                help='Whether to use rank features or not') 

        args = parser.parse_args()
        return args

    #####################################################################
    #  2. READ & PREPARE DATA
    #####################################################################

    args = parse_arguments()

    rolling_train_length = args.rolling_train_length
    rolling_test_length  = args.rolling_test_length
    n_k                  = args.n_k
    holdout_start        = pd.to_datetime(args.holdout_start, utc=False)
    batch_size           = args.batch_size
    learning_rate        = args.learning_rate
    optuna_runs          = args.optuna_runs
    random_seed          = args.random_seed
    gcs_bucket           = args.gcs_bucket
    include_fundamentals = args.include_fundamentals
    include_will_features = args.include_will_features
    notes                = args.notes
    is_test              = args.is_test
    epochs              = args.epochs
    addition_technical   = args.additional_technical
    additional_factors   = args.additional_factors
    act_func       = args.act_func
    table_suffix   = args.table_suffx
    use_correlations = args.use_correlations
    will_portfolios = args.will_portfolios
    calculate_shap = args.calculate_shap
    calculate_ww = args.calculate_ww
    clamp_gradients = args.clamp_gradients
    custom_tie_breaks = args.tie_breaks
    will_predictions = args.will_predictions
    include_coint_regimes = args.include_coint_regimes
    include_cluster_data = args.include_cluster_data
    include_hmm_regimes = args.include_hmm_regimes
    include_skew_kurt = args.include_skew_kurt
    include_time_features = args.include_time_features
    new_portfolios = args.portfolio_returns_table
    factor_lag = args.factor_lag
    live_next_day = args.live_next_day
    return_lag = args.return_lag
    core_model_column = args.core_model_column
    l0_config = args.l0_config
    use_rank_features = args.use_rank_features
    rolling_window = args.rolling_window

    train_main_function(
        rolling_train_length=rolling_train_length,
        rolling_test_length=rolling_test_length,
        n_k=n_k,
        holdout_start=holdout_start,
        batch_size=batch_size,
        learning_rate=learning_rate,
        optuna_runs=optuna_runs,
        random_seed=random_seed,
        gcs_bucket=gcs_bucket,
        include_fundamentals=include_fundamentals,
        include_will_features=include_will_features,
        notes=notes,
        epochs=epochs,
        addition_technical=addition_technical,
        additional_factors=additional_factors,
        act_func=act_func,
        table_suffix=table_suffix,
        use_correlations=use_correlations,
        will_portfolios=will_portfolios,
        calculate_shap=calculate_shap,
        calculate_ww=calculate_ww,
        clamp_gradients=clamp_gradients,
        custom_tie_breaks=custom_tie_breaks,
        will_predictions=will_predictions,
        include_coint_regimes=include_coint_regimes,
        include_cluster_data=include_cluster_data,
        include_hmm_regimes=include_hmm_regimes,
        include_skew_kurt=include_skew_kurt,
        include_time_features=include_time_features,
        new_portfolios=new_portfolios,
        factor_lag=factor_lag,
        live_next_day=live_next_day,
        is_test=is_test,
        return_lag=return_lag,
        core_model_column=core_model_column,
        l0_config=l0_config,
        use_rank_features=use_rank_features,
        rolling_window=rolling_window,
        )

if __name__ == "__main__":
    main()