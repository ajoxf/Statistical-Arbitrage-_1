# -*- coding: utf-8 -*-
"""
Created on Sat Jun 28 01:18:13 2025

@author: AJ
"""

# -*- coding: utf-8 -*-
"""
Comprehensive Pairs Trading Strategy with Multiple Asset Classes
@author: AJ
Enhanced version with extensive pair library and advanced features
"""
import statsmodels.api as sm
import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, coint
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')
import sys

# =============================================================================
# COMPREHENSIVE PAIRS LIBRARY - MULTIPLE ASSET CLASSES
# =============================================================================

PAIRS_LIBRARY = {
    # TECHNOLOGY SECTOR
    'TECH_FAANG': [
        ('AAPL', 'MSFT'),    # Apple vs Microsoft
        ('GOOGL', 'META'),   # Google vs Meta
        ('AMZN', 'MSFT'),    # Amazon vs Microsoft
        ('NFLX', 'DIS'),     # Netflix vs Disney
        ('AAPL', 'GOOGL'),   # Apple vs Google
        ('META', 'SNAP'),    # Meta vs Snapchat
        ('ORCL', 'MSFT'),    # Oracle vs Microsoft
        ('CRM', 'ORCL'),     # Salesforce vs Oracle
    ],
    
    'TECH_CHIPS': [
        ('NVDA', 'AMD'),     # NVIDIA vs AMD
        ('INTC', 'AMD'),     # Intel vs AMD
        ('TSM', 'NVDA'),     # Taiwan Semi vs NVIDIA
        ('QCOM', 'INTC'),    # Qualcomm vs Intel
        ('AVGO', 'QCOM'),    # Broadcom vs Qualcomm
        ('MU', 'WDC'),       # Micron vs Western Digital
        ('AMAT', 'LRCX'),    # Applied Materials vs Lam Research
    ],
    
    # FINANCIAL SECTOR
    'BANKS_MAJOR': [
        ('JPM', 'BAC'),      # JPMorgan vs Bank of America
        ('WFC', 'C'),        # Wells Fargo vs Citigroup
        ('GS', 'MS'),        # Goldman Sachs vs Morgan Stanley
        ('JPM', 'WFC'),      # JPMorgan vs Wells Fargo
        ('BAC', 'C'),        # Bank of America vs Citigroup
        ('USB', 'PNC'),      # US Bank vs PNC
        ('TFC', 'BBT'),      # Truist vs BB&T (if available)
    ],
    
    'FINANCIAL_SERVICES': [
        ('V', 'MA'),         # Visa vs Mastercard (classic pair)
        ('AXP', 'V'),        # American Express vs Visa
        ('BRK-B', 'JPM'),    # Berkshire vs JPMorgan
        ('GS', 'BLK'),       # Goldman Sachs vs BlackRock
        ('SCHW', 'IBKR'),    # Schwab vs Interactive Brokers
        ('COF', 'DFS'),      # Capital One vs Discover
    ],
    
    # ENERGY SECTOR
    'OIL_GAS': [
        ('XOM', 'CVX'),      # Exxon vs Chevron (classic)
        ('COP', 'EOG'),      # ConocoPhillips vs EOG Resources
        ('SLB', 'HAL'),      # Schlumberger vs Halliburton
        ('VLO', 'MPC'),      # Valero vs Marathon Petroleum
        ('KMI', 'EPD'),      # Kinder Morgan vs Enterprise Products
        ('OXY', 'DVN'),      # Occidental vs Devon Energy
        ('PXD', 'FANG'),     # Pioneer vs Diamondback Energy
    ],
    
    'ENERGY_ETFS': [
        ('XLE', 'VDE'),      # Energy ETFs
        ('OIH', 'XES'),      # Oil service ETFs
        ('USO', 'UCO'),      # Oil ETFs (different leverage)
        ('UNG', 'UGAZ'),     # Natural gas ETFs
    ],
    
    # RETAIL & CONSUMER
    'RETAIL': [
        ('WMT', 'TGT'),      # Walmart vs Target
        ('HD', 'LOW'),       # Home Depot vs Lowe's (classic)
        ('COST', 'WMT'),     # Costco vs Walmart
        ('AMZN', 'WMT'),     # Amazon vs Walmart
        ('NKE', 'ADDYY'),    # Nike vs Adidas
        ('SBUX', 'MCD'),     # Starbucks vs McDonald's
        ('KO', 'PEP'),       # Coca-Cola vs Pepsi (classic)
    ],
    
    'CONSUMER_GOODS': [
        ('PG', 'UL'),        # Procter & Gamble vs Unilever
        ('JNJ', 'PFE'),      # Johnson & Johnson vs Pfizer
        ('MRK', 'PFE'),      # Merck vs Pfizer
        ('ABT', 'JNJ'),      # Abbott vs J&J
        ('CL', 'PG'),        # Colgate vs P&G
    ],
    
    # TELECOMMUNICATIONS
    'TELECOM': [
        ('VZ', 'T'),         # Verizon vs AT&T (classic)
        ('TMUS', 'VZ'),      # T-Mobile vs Verizon
        ('CMCSA', 'CHTR'),   # Comcast vs Charter
        ('NFLX', 'CMCSA'),   # Netflix vs Comcast
    ],
    
    # GOLD & PRECIOUS METALS
    'GOLD_MINERS': [
        ('GLD', 'GDX'),      # Physical gold vs miners (classic)
        ('GLD', 'NEM'),      # Gold vs Newmont
        ('NEM', 'AEM'),      # Newmont vs Agnico Eagle
        ('GDX', 'GDXJ'),     # Large vs junior miners
        ('NEM', 'GOLD'),     # Newmont vs Barrick
        ('SLV', 'AG'),       # Silver vs First Majestic
        ('GLD', 'GOLD'),     # Physical vs Barrick
    ],
    
    # UTILITIES
    'UTILITIES': [
        ('NEE', 'DUK'),      # NextEra vs Duke Energy
        ('SO', 'D'),         # Southern vs Dominion
        ('AEP', 'EXC'),      # American Electric vs Exelon
        ('XEL', 'WEC'),      # Xcel vs WEC Energy
    ],
    
    # REAL ESTATE (REITs)
    'REITS': [
        ('SPG', 'MAC'),      # Simon Property vs Macerich
        ('PLD', 'EXR'),      # Prologis vs Extended Stay
        ('AMT', 'CCI'),      # American Tower vs Crown Castle
        ('EQIX', 'DLR'),     # Equinix vs Digital Realty
        ('VNO', 'BXP'),      # Vornado vs Boston Properties
    ],
    
    # TRANSPORTATION
    'TRANSPORT': [
        ('UPS', 'FDX'),      # UPS vs FedEx
        ('LUV', 'AAL'),      # Southwest vs American Airlines
        ('DAL', 'UAL'),      # Delta vs United
        ('CSX', 'UNP'),      # CSX vs Union Pacific
    ],
    
    # INTERNATIONAL PAIRS
    'INTERNATIONAL': [
        ('EWJ', 'FXI'),      # Japan vs China ETFs
        ('EFA', 'EEM'),      # Developed vs Emerging Markets
        ('SPY', 'EFA'),      # US vs International
        ('VTI', 'VXUS'),     # US vs International stocks
        ('GLD', 'PPLT'),     # Gold vs Platinum
    ],
    
    # SECTOR ETFS
    'SECTOR_ETFS': [
        ('XLF', 'XLK'),      # Financials vs Technology
        ('XLE', 'XLU'),      # Energy vs Utilities
        ('XLV', 'XLB'),      # Healthcare vs Materials
        ('XLY', 'XLP'),      # Consumer Disc vs Staples
        ('QQQ', 'SPY'),      # NASDAQ vs S&P 500
        ('IWM', 'SPY'),      # Small cap vs Large cap
    ],
    
    # BONDS & RATES
    'FIXED_INCOME': [
        ('TLT', 'IEF'),      # Long vs Intermediate Treasury
        ('TLT', 'HYG'),      # Treasury vs High Yield
        ('LQD', 'HYG'),      # Investment Grade vs High Yield
        ('TIP', 'TLT'),      # TIPS vs Treasury
        ('EMB', 'LQD'),      # Emerging vs Investment Grade
    ],
    
    # CURRENCY HEDGED PAIRS
    'CURRENCY' : [
        # === MAJOR FX PAIRS ===
        ('FXE', 'UUP'),       # EUR/USD – Euro vs US Dollar
        ('FXY', 'UUP'),       # JPY/USD – Japanese Yen vs US Dollar
        ('FXB', 'UUP'),       # GBP/USD – British Pound vs US Dollar
        ('FXA', 'UUP'),       # AUD/USD – Australian Dollar vs US Dollar
        ('FXC', 'UUP'),       # CAD/USD – Canadian Dollar vs US Dollar
        ('FXF', 'UUP'),       # CHF/USD – Swiss Franc vs US Dollar
        ('FXSG', 'UUP'),      # SGD/USD – Singapore Dollar vs US Dollar
        ('FXE', 'FXY'),       # EUR/JPY – Euro vs Yen
        ('FXB', 'FXE'),       # GBP/EUR – Pound vs Euro
        ('FXB', 'FXY'),       # GBP/JPY – Pound vs Yen
    
        # === MINOR / CROSS FX PAIRS ===
        ('FXA', 'FXE'),       # AUD/EUR – Aussie vs Euro
        ('FXC', 'FXE'),       # CAD/EUR – Loonie vs Euro
        ('FXA', 'FXB'),       # AUD/GBP – Aussie vs Pound
        ('FXC', 'FXB'),       # CAD/GBP – Loonie vs Pound
        ('FXA', 'FXC'),       # AUD/CAD – Aussie vs Loonie
        ('FXF', 'FXE'),       # CHF/EUR – Franc vs Euro
        ('FXF', 'FXB'),       # CHF/GBP – Franc vs Pound
        ('FXF', 'FXY'),       # CHF/JPY – Franc vs Yen
        ('FXSG', 'FXE'),      # SGD/EUR – Singapore Dollar vs Euro

        # === COMMODITY / HARD CURRENCY CROSS OVERRIDES ===
        ('GLD', 'UUP'),       # Gold vs US Dollar
        ('SLV', 'UUP'),       # Silver vs US Dollar
        ('GLD', 'FXE'),       # Gold vs Euro
        ('GLD', 'FXY'),       # Gold vs Yen
        ('GLD', 'FXB'),       # Gold vs Pound

        # === LATAM & EM FX ETFs (if desired) ===
        ('BRL=X', 'UUP'),     # Brazilian Real vs US Dollar (FX spot rate)
        ('MXN=X', 'UUP'),     # Mexican Peso vs US Dollar
    ],
    
    # VOLATILITY PAIRS
    'VOLATILITY': [
        ('VXX', 'UVXY'),     # VIX ETFs different leverage
        ('SPY', 'VXX'),      # S&P 500 vs VIX (inverse relationship)
        ('QQQ', 'SQQQ'),     # QQQ vs inverse QQQ
    ],
    
    # BITCOIN MINERS    
    'BITCOIN_MINERS': [

        ('BTC-USD', 'MARA'),      # Riot Platforms vs Marathon Digital
        ('BTC-USD', 'BITF'),       # Hut 8 Mining vs Bitfarms
        ('BTC-USD', 'IREN'),      # CleanSpark vs Iris Energy
        ('BTC-USD', 'BITF'),      # Hive Blockchain vs Bitfarms
        ('BTC-USD', 'RIGZ'),      # Bitcoin miner ETFs (WGMI vs RIGZ)
        ('BTC-USD', 'RIOT'),       # Hut 8 Mining vs Riot Platforms
        ('BTC-USD', 'CLSK'),      # Marathon vs CleanSpark
        ('BTC-USD', 'HIVE'),      # Iris Energy vs Hive Blockchain
        ('BTC-USD', 'SDIG'),      # Bitdeer Technologies vs Stronghold 

       # ('RIOT', 'MARA'),      # Riot Platforms vs Marathon Digital
       # ('HUT', 'BITF'),       # Hut 8 Mining vs Bitfarms
       # ('CLSK', 'IREN'),      # CleanSpark vs Iris Energy
       # ('HIVE', 'BITF'),      # Hive Blockchain vs Bitfarms
       # ('WGMI', 'RIGZ'),      # Bitcoin miner ETFs (WGMI vs RIGZ)
       # ('HUT', 'RIOT'),       # Hut 8 Mining vs Riot Platforms
       # ('MARA', 'CLSK'),      # Marathon vs CleanSpark
       # ('IREN', 'HIVE'),      # Iris Energy vs Hive Blockchain
       # ('BTDR', 'SDIG'),      # Bitdeer Technologies vs Stronghold
   ],
}

# =============================================================================
# ENHANCED CONFIGURATION WITH AUTO-PAIR TESTING
# =============================================================================

# Choose which category to test (or 'ALL' for comprehensive scan)
CATEGORY_TO_TEST = 'ALL'  # Change this to test different categories
# Options: 'TECH_FAANG', 'BANKS_MAJOR', 'OIL_GAS', 'RETAIL', 'GOLD_MINERS', 'ALL'

# If testing specific pair
MANUAL_PAIR = None  # Set to ('SYMBOL1', 'SYMBOL2') to test specific pair

# Time period
START_DATE = '2018-01-01'
END_DATE = '2025-06-01'

# Strategy parameters (will be optimized per pair)
BASE_LOOKBACK_PERIOD = 90
BASE_STD_DEV_THRESHOLD = 1.5
BASE_FORMATION_PERIOD = 120
USE_LOG_PRICES = True

# Portfolio parameters
INITIAL_CAPITAL = 100000

# Advanced settings
MIN_CORRELATION = 0.7        # Minimum correlation to consider
MAX_COINTEGRATION_PVALUE = 0.05  # Maximum p-value for cointegration
MIN_OBSERVATIONS = 500       # Minimum data points required

# IBKR Transaction Cost Structure
IBKR_FEES = {
    'commission_per_share': 0.005,    # $0.005 per share
    'min_commission': 1.00,           # $1.00 minimum per order
    'max_commission_pct': 0.01,       # 1% max of trade value
    'sec_fee_rate': 0.0000278,        # SEC fee (sales only)
    'finra_taf': 0.000166,            # FINRA TAF (sales only)
    'bid_ask_spread_bps': 2.5,        # Estimated 2.5 bps bid-ask spread cost
    'etf_commission': 0.00,           # Many ETFs are commission-free on IBKR
}

# ETF symbols that are typically commission-free on IBKR
COMMISSION_FREE_ETFS = {
    'SPY', 'QQQ', 'IWM', 'GLD', 'SLV', 'TLT', 'HYG', 'LQD', 'EFA', 'EEM', 
    'VTI', 'VEA', 'VWO', 'XLF', 'XLK', 'XLE', 'XLV', 'XLI', 'XLB', 'XLP', 
    'XLY', 'XLU', 'GDX', 'GDXJ', 'USO', 'UNG', 'TIP', 'IEF'
}

# =============================================================================
# PAIR SELECTION AND TESTING FRAMEWORK
# =============================================================================

def get_pairs_to_test():
    """Get list of pairs based on configuration"""
    if MANUAL_PAIR:
        return [MANUAL_PAIR]
    elif CATEGORY_TO_TEST == 'ALL':
        all_pairs = []
        for category_pairs in PAIRS_LIBRARY.values():
            all_pairs.extend(category_pairs)
        return all_pairs
    else:
        return PAIRS_LIBRARY.get(CATEGORY_TO_TEST, [])

def download_pair_data(symbol1, symbol2, start_date, end_date):
    """Download and prepare data for a pair"""
    try:
        # Download data
        data1 = yf.download(symbol1, start=start_date, end=end_date, progress=False)
        data2 = yf.download(symbol2, start=start_date, end=end_date, progress=False)
        
        if data1.empty or data2.empty:
            return None, f"No data for {symbol1} or {symbol2}"
        
        # Create combined dataframe
        df = pd.concat([data1['Close'], data2['Close']], axis=1)
        df.columns = [symbol1, symbol2]
        df = df.dropna()
        
        if len(df) < MIN_OBSERVATIONS:
            return None, f"Insufficient data: {len(df)} < {MIN_OBSERVATIONS}"
        
        return df, None
        
    except Exception as e:
        return None, f"Error downloading: {str(e)}"

def analyze_pair_quality(df, symbol1, symbol2, use_log=True):
    """Comprehensive pair quality analysis"""
    # Transform prices
    if use_log:
        prices1 = np.log(df[symbol1])
        prices2 = np.log(df[symbol2])
    else:
        prices1 = df[symbol1]
        prices2 = df[symbol2]
    
    # Calculate metrics
    correlation = prices1.corr(prices2)
    
    # Cointegration test
    try:
        coint_score, coint_pvalue, _ = coint(prices1, prices2)
        cointegrated = coint_pvalue < MAX_COINTEGRATION_PVALUE
    except:
        coint_pvalue = 1.0
        cointegrated = False
    
    # Hedge ratio and R-squared
    X = sm.add_constant(prices2)
    model = sm.OLS(prices1, X).fit()
    hedge_ratio = model.params[prices2.name]
    r_squared = model.rsquared
    
    # Spread stationarity
    spread = prices1 - hedge_ratio * prices2
    try:
        adf_stat, adf_pvalue, _, _, _, _ = adfuller(spread.dropna(), maxlag=10)
        stationary = adf_pvalue < 0.05
    except:
        adf_pvalue = 1.0
        stationary = False
    
    # Quality score (0-100)
    quality_score = (
        min(correlation, 1.0) * 25 +  # Correlation component
        (1 - coint_pvalue) * 25 +      # Cointegration component  
        r_squared * 25 +               # R-squared component
        (1 - adf_pvalue) * 25          # Stationarity component
    )
    
    return {
        'correlation': correlation,
        'cointegration_pvalue': coint_pvalue,
        'cointegrated': cointegrated,
        'hedge_ratio': hedge_ratio,
        'r_squared': r_squared,
        'adf_pvalue': adf_pvalue,
        'stationary': stationary,
        'quality_score': quality_score,
        'prices1': prices1,
        'prices2': prices2,
        'spread': spread
    }

def optimize_parameters(pair_analysis, df_length):
    """Optimize parameters based on pair characteristics"""
    quality_score = pair_analysis['quality_score']
    
    # Adjust parameters based on quality
    if quality_score > 80:
        # High quality pair - use standard parameters
        lookback = BASE_LOOKBACK_PERIOD
        std_threshold = BASE_STD_DEV_THRESHOLD
    elif quality_score > 60:
        # Medium quality - more conservative
        lookback = min(int(BASE_LOOKBACK_PERIOD * 1.2), df_length // 4)
        std_threshold = BASE_STD_DEV_THRESHOLD * 0.8
    else:
        # Lower quality - very conservative
        lookback = min(int(BASE_LOOKBACK_PERIOD * 1.5), df_length // 3)
        std_threshold = BASE_STD_DEV_THRESHOLD * 0.6
    
    formation_period = min(BASE_FORMATION_PERIOD, df_length // 3)
    
    return {
        'lookback_period': lookback,
        'std_dev_threshold': std_threshold,
        'formation_period': formation_period
    }

def calculate_ibkr_transaction_costs(symbol, price, shares_traded):
    """Calculate realistic IBKR transaction costs"""
    trade_value = abs(shares_traded * price)
    
    # Check if ETF is commission-free
    if symbol in COMMISSION_FREE_ETFS:
        commission = 0.0
    else:
        # Calculate commission
        commission = max(
            IBKR_FEES['min_commission'],
            min(abs(shares_traded) * IBKR_FEES['commission_per_share'],
                trade_value * IBKR_FEES['max_commission_pct'])
        )
    
    # Regulatory fees (for sales only)
    sec_fee = trade_value * IBKR_FEES['sec_fee_rate'] if shares_traded < 0 else 0
    finra_fee = trade_value * IBKR_FEES['finra_taf'] if shares_traded < 0 else 0
    
    # Bid-ask spread cost (both buys and sells)
    spread_cost = trade_value * (IBKR_FEES['bid_ask_spread_bps'] / 10000)
    
    total_cost = commission + sec_fee + finra_fee + spread_cost
    return total_cost / trade_value if trade_value > 0 else 0

def calculate_strategy_returns(df, symbol1, symbol2, pair_analysis, parameters):
    """Calculate strategy returns with realistic IBKR transaction costs"""
    spread = pair_analysis['spread']
    hedge_ratio = pair_analysis['hedge_ratio']
    lookback = parameters['lookback_period']
    std_threshold = parameters['std_dev_threshold']
    
    # Calculate bands
    df_strategy = df.copy()
    df_strategy['spread'] = spread
    df_strategy['ma'] = spread.rolling(lookback).mean()
    df_strategy['std'] = spread.rolling(lookback).std()
    df_strategy['upper_band'] = df_strategy['ma'] + std_threshold * df_strategy['std']
    df_strategy['lower_band'] = df_strategy['ma'] - std_threshold * df_strategy['std']
    
    # Generate signals
    df_strategy['position'] = 0.0
    for i in range(lookback, len(df_strategy)):
        prev_pos = df_strategy['position'].iloc[i-1]
        current_spread = df_strategy['spread'].iloc[i]
        ma = df_strategy['ma'].iloc[i]
        upper = df_strategy['upper_band'].iloc[i]
        lower = df_strategy['lower_band'].iloc[i]
        
        # Exit conditions
        if prev_pos == 1 and current_spread >= ma:
            df_strategy.iloc[i, df_strategy.columns.get_loc('position')] = 0
        elif prev_pos == -1 and current_spread <= ma:
            df_strategy.iloc[i, df_strategy.columns.get_loc('position')] = 0
        # Entry conditions
        elif prev_pos == 0:
            if current_spread < lower:
                df_strategy.iloc[i, df_strategy.columns.get_loc('position')] = 1
            elif current_spread > upper:
                df_strategy.iloc[i, df_strategy.columns.get_loc('position')] = -1
            else:
                df_strategy.iloc[i, df_strategy.columns.get_loc('position')] = 0
        else:
            df_strategy.iloc[i, df_strategy.columns.get_loc('position')] = prev_pos
    
    # Calculate returns
    df_strategy['returns_1'] = df[symbol1].pct_change()
    df_strategy['returns_2'] = df[symbol2].pct_change()
    
    # Position sizing - calculate actual shares based on $10,000 per position leg
    position_size_usd = 10000  # $10k per leg when position = 1
    df_strategy['shares_1'] = (df_strategy['position'] * position_size_usd / df[symbol1]).round()
    df_strategy['shares_2'] = (-df_strategy['position'] * hedge_ratio * position_size_usd / df[symbol2]).round()
    
    # Calculate position changes in shares
    df_strategy['shares_1_change'] = df_strategy['shares_1'].diff().fillna(0)
    df_strategy['shares_2_change'] = df_strategy['shares_2'].diff().fillna(0)
    
    # Calculate IBKR transaction costs
    df_strategy['cost_1'] = 0.0
    df_strategy['cost_2'] = 0.0
    
    for i in range(len(df_strategy)):
        if df_strategy['shares_1_change'].iloc[i] != 0:
            df_strategy.iloc[i, df_strategy.columns.get_loc('cost_1')] = calculate_ibkr_transaction_costs(
                symbol1, df[symbol1].iloc[i], df_strategy['shares_1_change'].iloc[i]
            )
        
        if df_strategy['shares_2_change'].iloc[i] != 0:
            df_strategy.iloc[i, df_strategy.columns.get_loc('cost_2')] = calculate_ibkr_transaction_costs(
                symbol2, df[symbol2].iloc[i], df_strategy['shares_2_change'].iloc[i]
            )
    
    # Total transaction costs as percentage of portfolio
    df_strategy['total_transaction_costs'] = df_strategy['cost_1'] + df_strategy['cost_2']
    
    # Position weights for return calculation
    df_strategy['pos_1'] = df_strategy['position']
    df_strategy['pos_2'] = -df_strategy['position'] * hedge_ratio
    
    # Strategy returns before transaction costs
    df_strategy['strategy_returns_gross'] = (
        df_strategy['pos_1'].shift(1) * df_strategy['returns_1'] + 
        df_strategy['pos_2'].shift(1) * df_strategy['returns_2']
    )
    
    # Net returns after IBKR transaction costs
    df_strategy['strategy_returns_net'] = (
        df_strategy['strategy_returns_gross'] - df_strategy['total_transaction_costs']
    )
    
    # Keep both for comparison
    df_strategy['strategy_returns'] = df_strategy['strategy_returns_net']
    
    return df_strategy

def calculate_performance_metrics(returns_series, name="Strategy"):
    """Calculate comprehensive performance metrics"""
    returns_series = returns_series.fillna(0)
    if len(returns_series) == 0 or returns_series.std() == 0:
        return None
    
    total_return = (1 + returns_series).prod() - 1
    annual_return = (1 + total_return) ** (252 / len(returns_series)) - 1
    volatility = returns_series.std() * np.sqrt(252)
    sharpe_ratio = annual_return / volatility if volatility > 0 else 0
    
    # Drawdown
    cumulative = (1 + returns_series).cumprod()
    peak = cumulative.expanding().max()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min()
    
    # Other metrics
    win_rate = (returns_series > 0).sum() / len(returns_series)
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0
    
    # Trade statistics
    num_trades = len(returns_series[returns_series != 0])
    
    return {
        'name': name,
        'total_return': total_return,
        'annual_return': annual_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio,
        'win_rate': win_rate,
        'num_trades': num_trades
    }

# =============================================================================
# COMPREHENSIVE PAIR SCREENING
# =============================================================================

def screen_all_pairs():
    """Screen all pairs and return results sorted by quality"""
    pairs_to_test = get_pairs_to_test()
    results = []
    
    print(f"Screening {len(pairs_to_test)} pairs...")
    print("="*80)
    
    for i, (symbol1, symbol2) in enumerate(pairs_to_test):
        print(f"Testing {i+1}/{len(pairs_to_test)}: {symbol1} vs {symbol2}")
        
        # Download data
        df, error = download_pair_data(symbol1, symbol2, START_DATE, END_DATE)
        if df is None:
            print(f"  ❌ {error}")
            continue
        
        # Analyze pair quality
        pair_analysis = analyze_pair_quality(df, symbol1, symbol2, USE_LOG_PRICES)
        
        # Check minimum requirements
        if (pair_analysis['correlation'] < MIN_CORRELATION or
            not pair_analysis['cointegrated']):
            print(f"  ❌ Low quality (Score: {pair_analysis['quality_score']:.1f})")
            continue
        
        # Optimize parameters
        parameters = optimize_parameters(pair_analysis, len(df))
        
        # Calculate strategy performance
        df_strategy = calculate_strategy_returns(df, symbol1, symbol2, pair_analysis, parameters)
        strategy_metrics = calculate_performance_metrics(df_strategy['strategy_returns'], 
                                                       f"{symbol1}-{symbol2}")
        
        if strategy_metrics is None:
            print(f"  ❌ Strategy calculation failed")
            continue
        
        # Store results
        result = {
            'pair': (symbol1, symbol2),
            'pair_analysis': pair_analysis,
            'parameters': parameters,
            'strategy_metrics': strategy_metrics,
            'df': df,
            'df_strategy': df_strategy
        }
        results.append(result)
        
        print(f"  ✅ Quality: {pair_analysis['quality_score']:.1f}, "
              f"Sharpe: {strategy_metrics['sharpe_ratio']:.2f}, "
              f"Return: {strategy_metrics['annual_return']:.1%}")
    
    # Sort by Sharpe ratio
    results.sort(key=lambda x: x['strategy_metrics']['sharpe_ratio'], reverse=True)
    return results

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Screen pairs
    results = screen_all_pairs()
    
    if not results:
        print("❌ No suitable pairs found!")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f"TOP PERFORMING PAIRS")
    print(f"{'='*80}")
    
    # Display top 10 results
    for i, result in enumerate(results[:10]):
        pair = result['pair']
        analysis = result['pair_analysis']
        metrics = result['strategy_metrics']
        
        print(f"\n{i+1}. {pair[0]} vs {pair[1]}")
        print(f"   Quality Score: {analysis['quality_score']:.1f}/100")
        print(f"   Correlation: {analysis['correlation']:.3f}")
        print(f"   Cointegration p-value: {analysis['cointegration_pvalue']:.4f}")
        print(f"   Annual Return: {metrics['annual_return']:.1%}")
        print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"   Max Drawdown: {metrics['max_drawdown']:.1%}")
        print(f"   Win Rate: {metrics['win_rate']:.1%}")
    
    # Analyze best pair in detail
    if results:
        best_result = results[0]
        best_pair = best_result['pair']
        best_df = best_result['df']
        best_df_strategy = best_result['df_strategy']
        best_analysis = best_result['pair_analysis']
        best_metrics = best_result['strategy_metrics']
        
        print(f"\n{'='*80}")
        print(f"DETAILED ANALYSIS: {best_pair[0]} vs {best_pair[1]} (BEST PAIR)")
        print(f"{'='*80}")
        
        # Create detailed visualization
        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # 1. Price comparison
        ax = axes[0, 0]
        ax.plot(best_df.index, best_df[best_pair[0]], label=best_pair[0], linewidth=2)
        ax_twin = ax.twinx()
        ax_twin.plot(best_df.index, best_df[best_pair[1]], label=best_pair[1], 
                    color='red', linewidth=2)
        ax.set_title(f'{best_pair[0]} vs {best_pair[1]} Prices')
        ax.legend(loc='upper left')
        ax_twin.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # 2. Spread with signals
        ax = axes[0, 1]
        ax.plot(best_df_strategy.index, best_df_strategy['spread'], alpha=0.7, label='Spread')
        ax.plot(best_df_strategy.index, best_df_strategy['ma'], color='black', label='MA')
        ax.plot(best_df_strategy.index, best_df_strategy['upper_band'], 
               color='red', linestyle='--', label='Upper')
        ax.plot(best_df_strategy.index, best_df_strategy['lower_band'], 
               color='green', linestyle='--', label='Lower')
        
        # Add signals
        long_signals = best_df_strategy[best_df_strategy['position'] == 1].index
        short_signals = best_df_strategy[best_df_strategy['position'] == -1].index
        if len(long_signals) > 0:
            ax.scatter(long_signals, best_df_strategy.loc[long_signals, 'spread'], 
                      color='green', marker='^', s=20, alpha=0.8)
        if len(short_signals) > 0:
            ax.scatter(short_signals, best_df_strategy.loc[short_signals, 'spread'], 
                      color='red', marker='v', s=20, alpha=0.8)
        
        ax.set_title('Spread with Trading Signals')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Portfolio performance
        ax = axes[0, 2]
        portfolio_value_net = INITIAL_CAPITAL * (1 + best_df_strategy['strategy_returns_net']).cumprod()
        portfolio_value_gross = INITIAL_CAPITAL * (1 + best_df_strategy['strategy_returns_gross']).cumprod()
        bh1_value = INITIAL_CAPITAL * (1 + best_df_strategy['returns_1']).cumprod()
        bh2_value = INITIAL_CAPITAL * (1 + best_df_strategy['returns_2']).cumprod()
        
        ax.plot(best_df_strategy.index, portfolio_value_net, label='Strategy (Net)', linewidth=3, color='blue')
        ax.plot(best_df_strategy.index, portfolio_value_gross, label='Strategy (Gross)', linewidth=2, color='lightblue', linestyle='--')
        ax.plot(best_df_strategy.index, bh1_value, label=f'{best_pair[0]} B&H', alpha=0.7, color='red')
        ax.plot(best_df_strategy.index, bh2_value, label=f'{best_pair[1]} B&H', alpha=0.7, color='orange')
        ax.axhline(y=INITIAL_CAPITAL, color='black', linestyle=':', alpha=0.5)
        ax.set_title('Portfolio Performance (Net vs Gross)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # 4. Position allocation
        ax = axes[1, 0]
        ax.plot(best_df_strategy.index, best_df_strategy['position'])
        ax.fill_between(best_df_strategy.index, best_df_strategy['position'], 0, 
                       where=(best_df_strategy['position'] > 0), alpha=0.3, color='green')
        ax.fill_between(best_df_strategy.index, best_df_strategy['position'], 0, 
                       where=(best_df_strategy['position'] < 0), alpha=0.3, color='red')
        ax.set_title('Position Allocation')
        ax.set_ylabel('Position')
        ax.grid(True, alpha=0.3)
        
        # 5. Rolling Sharpe ratio
        ax = axes[1, 1]
        rolling_returns = best_df_strategy['strategy_returns'].rolling(252)
        rolling_sharpe = rolling_returns.mean() / rolling_returns.std() * np.sqrt(252)
        ax.plot(best_df_strategy.index, rolling_sharpe)
        ax.axhline(y=1, color='green', linestyle='--', alpha=0.7, label='Sharpe = 1')
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Sharpe = 0')
        ax.set_title('Rolling 1-Year Sharpe Ratio')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. Returns distribution
        ax = axes[1, 2]
        returns = best_df_strategy['strategy_returns'].dropna()
        ax.hist(returns, bins=50, alpha=0.7, density=True)
        ax.axvline(x=returns.mean(), color='red', linestyle='--', label=f'Mean: {returns.mean():.4f}')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        ax.set_title('Returns Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Best Pair Analysis: {best_pair[0]} vs {best_pair[1]}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Summary statistics
        print(f"\nSUMMARY FOR {best_pair[0]} vs {best_pair[1]}:")
        print(f"Quality Score: {best_analysis['quality_score']:.1f}/100")
        print(f"Final Portfolio Value: ${portfolio_value_net.iloc[-1]:,.0f}")
        print(f"Total Return: {best_metrics['total_return']:.1%}")
        print(f"Annual Return: {best_metrics['annual_return']:.1%}")
        print(f"Sharpe Ratio: {best_metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {best_metrics['max_drawdown']:.1%}")
        print(f"Win Rate: {best_metrics['win_rate']:.1%}")
        print(f"Number of Trades: {best_metrics['num_trades']}")
        
        # IBKR Transaction Cost Analysis
        print(f"\n{'='*60}")
        print(f"IBKR TRANSACTION COST ANALYSIS")
        print(f"{'='*60}")
        
        # Calculate gross vs net performance
        gross_returns = best_df_strategy['strategy_returns_gross'].fillna(0)
        net_returns = best_df_strategy['strategy_returns_net'].fillna(0)
        total_costs = best_df_strategy['total_transaction_costs'].sum()
        
        gross_total_return = (1 + gross_returns).prod() - 1
        net_total_return = (1 + net_returns).prod() - 1
        gross_annual_return = (1 + gross_total_return) ** (252 / len(gross_returns)) - 1
        net_annual_return = (1 + net_total_return) ** (252 / len(net_returns)) - 1
        
        # Count transactions
        trades_symbol1 = (best_df_strategy['shares_1_change'] != 0).sum()
        trades_symbol2 = (best_df_strategy['shares_2_change'] != 0).sum()
        total_trades = trades_symbol1 + trades_symbol2
        
        # Calculate average cost per trade
        avg_cost_per_trade = total_costs / total_trades if total_trades > 0 else 0
        cost_impact_annual = (gross_annual_return - net_annual_return)
        
        print(f"Gross Annual Return: {gross_annual_return:.2%}")
        print(f"Net Annual Return: {net_annual_return:.2%}")
        print(f"Annual Cost Impact: {cost_impact_annual:.2%}")
        print(f"Total Transaction Costs: {total_costs:.4%} of portfolio")
        print(f"Total Number of Trades: {total_trades}")
        print(f"  - {best_pair[0]} trades: {trades_symbol1}")
        print(f"  - {best_pair[1]} trades: {trades_symbol2}")
        print(f"Average Cost per Trade: {avg_cost_per_trade:.4%}")
        
        # ETF commission analysis
        etf1_free = best_pair[0] in COMMISSION_FREE_ETFS
        etf2_free = best_pair[1] in COMMISSION_FREE_ETFS
        print(f"\nCommission-Free Status:")
        print(f"  {best_pair[0]}: {'✓ Commission-Free' if etf1_free else '✗ $0.005/share'}")
        print(f"  {best_pair[1]}: {'✓ Commission-Free' if etf2_free else '✗ $0.005/share'}")
        
        if etf1_free and etf2_free:
            print("  💡 Both symbols are commission-free ETFs - Lower transaction costs!")
        elif etf1_free or etf2_free:
            print("  💡 One symbol is commission-free - Moderate transaction costs")
        else:
            print("  ⚠️  Both symbols have commissions - Higher transaction costs")
        
        # Cost breakdown for last trade (if any)
        last_trade_idx = best_df_strategy[best_df_strategy['total_transaction_costs'] > 0].index
        if len(last_trade_idx) > 0:
            last_idx = last_trade_idx[-1]
            print(f"\nLast Trade Cost Breakdown (on {last_idx.strftime('%Y-%m-%d')}):")
            print(f"  {best_pair[0]} cost: {best_df_strategy.loc[last_idx, 'cost_1']:.4%}")
            print(f"  {best_pair[1]} cost: {best_df_strategy.loc[last_idx, 'cost_2']:.4%}")
            print(f"  Total cost: {best_df_strategy.loc[last_idx, 'total_transaction_costs']:.4%}")
        
        # Compare with simplified cost model
        simple_cost = 0.001  # 0.1% per trade
        position_changes = best_df_strategy['position'].diff().abs()
        simple_total_cost = (position_changes * simple_cost).sum()
        print(f"\nCost Model Comparison:")
        print(f"  IBKR Realistic Costs: {total_costs:.4%}")
        print(f"  Simplified 0.1% Model: {simple_total_cost:.4%}")
        print(f"  Difference: {(total_costs - simple_total_cost):.4%}")
        
        if total_costs < simple_total_cost:
            print("  💰 IBKR costs are LOWER than simplified model!")
        else:
            print("  💸 IBKR costs are HIGHER than simplified model")

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE")
print(f"{'='*80}")

print(f"\nNEXT STEPS:")
print(f"1. Change CATEGORY_TO_TEST to explore different sectors")
print(f"2. Set MANUAL_PAIR = ('SYMBOL1', 'SYMBOL2') to test specific pairs")
print(f"3. Adjust parameters in the configuration section")
print(f"4. Review IBKR transaction cost analysis for realistic trading costs")
print(f"5. Consider commission-free ETF pairs to minimize costs")
print(f"6. Add walk-forward analysis for robustness testing")
print(f"7. Implement position sizing and risk management")

print(f"\nIBKR COST FEATURES:")
print(f"✓ Realistic per-share commissions ($0.005/share)")
print(f"✓ Commission-free ETF identification")
print(f"✓ SEC and FINRA regulatory fees")
print(f"✓ Bid-ask spread cost estimation")
print(f"✓ Gross vs Net return comparison")
print(f"✓ Cost breakdown per trade")