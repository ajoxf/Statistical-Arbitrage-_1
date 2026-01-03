# -*- coding: utf-8 -*-
"""
Statistical Arbitrage Pairs Trading Backtester
User-Friendly Version for Non-Technical Users

@author: AJ
@description: Simple interface to backtest any two Yahoo Finance tickers
              Generates an HTML report that can be saved as PDF
"""

import statsmodels.api as sm
import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for HTML generation
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, coint
import warnings
warnings.filterwarnings('ignore')
import base64
from io import BytesIO
import os
from datetime import datetime

# =============================================================================
# CONFIGURATION (Advanced users can modify these defaults)
# =============================================================================

DEFAULT_LOOKBACK_PERIOD = 90
DEFAULT_STD_DEV_THRESHOLD = 1.5
DEFAULT_FORMATION_PERIOD = 120
USE_LOG_PRICES = True
INITIAL_CAPITAL = 100000
POSITION_SIZE_PER_LEG = 50000  # $50k per leg (half of capital)
MIN_OBSERVATIONS = 100

# =============================================================================
# IBKR REALISTIC TRANSACTION COST STRUCTURE
# =============================================================================

IBKR_FEES = {
    'commission_per_share': 0.005,    # $0.005 per share
    'min_commission': 1.00,           # $1.00 minimum per order
    'max_commission_pct': 0.01,       # 1% max of trade value
    'sec_fee_rate': 0.0000278,        # SEC fee (sales only)
    'finra_taf': 0.000166,            # FINRA TAF (sales only)
    'bid_ask_spread_bps': 2.5,        # Estimated 2.5 bps bid-ask spread cost
}

# ETF symbols that are typically commission-free on IBKR
COMMISSION_FREE_ETFS = {
    'SPY', 'QQQ', 'IWM', 'GLD', 'SLV', 'TLT', 'HYG', 'LQD', 'EFA', 'EEM',
    'VTI', 'VEA', 'VWO', 'XLF', 'XLK', 'XLE', 'XLV', 'XLI', 'XLB', 'XLP',
    'XLY', 'XLU', 'GDX', 'GDXJ', 'USO', 'UNG', 'TIP', 'IEF', 'AGG', 'BND',
    'VNQ', 'ARKK', 'DIA', 'VOO', 'IVV', 'VTV', 'VUG', 'VIG', 'SCHD'
}

# =============================================================================
# USER INPUT FUNCTIONS
# =============================================================================

def get_user_inputs():
    """Get inputs from user with validation"""
    print("\n" + "="*60)
    print("   STATISTICAL ARBITRAGE PAIRS TRADING BACKTESTER")
    print("="*60)
    print("\nThis tool tests a pairs trading strategy on two assets.")
    print("You can use any ticker available on Yahoo Finance.")
    print("\nExamples: AAPL, MSFT, GLD, SPY, XOM, JPM, etc.")
    print("-"*60)

    # Get first ticker
    while True:
        ticker1 = input("\nEnter FIRST ticker symbol: ").strip().upper()
        if ticker1:
            break
        print("Please enter a valid ticker symbol.")

    # Get second ticker
    while True:
        ticker2 = input("Enter SECOND ticker symbol: ").strip().upper()
        if ticker2 and ticker2 != ticker1:
            break
        if ticker2 == ticker1:
            print("Second ticker must be different from first.")
        else:
            print("Please enter a valid ticker symbol.")

    # Get date range
    print("\n" + "-"*60)
    print("Enter the backtest period (date format: YYYY-MM-DD)")
    print("Example: 2020-01-01")
    print("-"*60)

    while True:
        start_date = input("\nStart date (YYYY-MM-DD): ").strip()
        try:
            datetime.strptime(start_date, '%Y-%m-%d')
            break
        except ValueError:
            print("Invalid date format. Please use YYYY-MM-DD (e.g., 2020-01-01)")

    while True:
        end_date = input("End date (YYYY-MM-DD): ").strip()
        try:
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            if end_dt > start_dt:
                break
            else:
                print("End date must be after start date.")
        except ValueError:
            print("Invalid date format. Please use YYYY-MM-DD (e.g., 2024-01-01)")

    # Get lookback period
    print("\n" + "-"*60)
    print("STRATEGY PARAMETERS")
    print("-"*60)
    print("\nLookback Period: Number of trading days used to calculate")
    print("the 'normal' spread behavior (moving average & std deviation).")
    print("\n  Suggested values:")
    print("    30 days  - Short-term, more trades, higher risk")
    print("    60 days  - Medium-term, balanced approach")
    print("    90 days  - Default, good for most pairs")
    print("    120 days - Long-term, fewer trades, more stable")
    print("-"*60)

    while True:
        lookback_input = input(f"\nLookback period in days [default: {DEFAULT_LOOKBACK_PERIOD}]: ").strip()
        if lookback_input == "":
            lookback_period = DEFAULT_LOOKBACK_PERIOD
            break
        try:
            lookback_period = int(lookback_input)
            if 10 <= lookback_period <= 365:
                break
            else:
                print("Please enter a value between 10 and 365 days.")
        except ValueError:
            print("Please enter a valid number (or press Enter for default).")

    # Summary
    print("\n" + "="*60)
    print("BACKTEST CONFIGURATION:")
    print("="*60)
    print(f"  Pair: {ticker1} vs {ticker2}")
    print(f"  Period: {start_date} to {end_date}")
    print(f"  Lookback Period: {lookback_period} days")
    print(f"  Entry Threshold: {DEFAULT_STD_DEV_THRESHOLD} standard deviations")
    print(f"  Initial Capital: ${INITIAL_CAPITAL:,}")
    print("="*60)

    confirm = input("\nProceed with backtest? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Backtest cancelled.")
        return None

    return {
        'ticker1': ticker1,
        'ticker2': ticker2,
        'start_date': start_date,
        'end_date': end_date,
        'lookback_period': lookback_period
    }

# =============================================================================
# DATA FUNCTIONS
# =============================================================================

def download_pair_data(symbol1, symbol2, start_date, end_date):
    """Download and prepare data for a pair"""
    print(f"\nDownloading data for {symbol1} and {symbol2}...")

    try:
        data1 = yf.download(symbol1, start=start_date, end=end_date, progress=False)
        data2 = yf.download(symbol2, start=start_date, end=end_date, progress=False)

        if data1.empty:
            return None, f"No data found for {symbol1}. Please check the ticker symbol."
        if data2.empty:
            return None, f"No data found for {symbol2}. Please check the ticker symbol."

        # Handle multi-level columns if present
        if isinstance(data1.columns, pd.MultiIndex):
            data1.columns = data1.columns.get_level_values(0)
        if isinstance(data2.columns, pd.MultiIndex):
            data2.columns = data2.columns.get_level_values(0)

        # Create combined dataframe
        df = pd.concat([data1['Close'], data2['Close']], axis=1)
        df.columns = [symbol1, symbol2]
        df = df.dropna()

        if len(df) < MIN_OBSERVATIONS:
            return None, f"Insufficient data: only {len(df)} trading days. Need at least {MIN_OBSERVATIONS}."

        print(f"Successfully downloaded {len(df)} trading days of data.")
        return df, None

    except Exception as e:
        return None, f"Error downloading data: {str(e)}"

# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def analyze_pair_quality(df, symbol1, symbol2, use_log=True):
    """Comprehensive pair quality analysis"""
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
        cointegrated = coint_pvalue < 0.05
    except:
        coint_pvalue = 1.0
        cointegrated = False

    # Hedge ratio and R-squared
    X = sm.add_constant(prices2)
    model = sm.OLS(prices1, X).fit()
    hedge_ratio = model.params.iloc[1] if hasattr(model.params, 'iloc') else model.params[1]
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
        min(abs(correlation), 1.0) * 25 +
        (1 - min(coint_pvalue, 1.0)) * 25 +
        r_squared * 25 +
        (1 - min(adf_pvalue, 1.0)) * 25
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

def calculate_ibkr_transaction_costs(symbol, price, shares_traded):
    """
    Calculate realistic IBKR transaction costs

    Components:
    - Commission: $0.005/share (min $1, max 1% of trade)
    - SEC fee: 0.00278% (sells only)
    - FINRA TAF: 0.0166% (sells only)
    - Bid-ask spread: ~2.5 bps estimate
    """
    if shares_traded == 0 or price == 0:
        return 0.0, {}

    trade_value = abs(shares_traded * price)
    is_sell = shares_traded < 0

    # Commission (check if commission-free ETF)
    if symbol.upper() in COMMISSION_FREE_ETFS:
        commission = 0.0
        is_commission_free = True
    else:
        commission = max(
            IBKR_FEES['min_commission'],
            min(abs(shares_traded) * IBKR_FEES['commission_per_share'],
                trade_value * IBKR_FEES['max_commission_pct'])
        )
        is_commission_free = False

    # Regulatory fees (sells only)
    sec_fee = trade_value * IBKR_FEES['sec_fee_rate'] if is_sell else 0
    finra_fee = trade_value * IBKR_FEES['finra_taf'] if is_sell else 0

    # Bid-ask spread cost (both buys and sells)
    spread_cost = trade_value * (IBKR_FEES['bid_ask_spread_bps'] / 10000)

    total_cost = commission + sec_fee + finra_fee + spread_cost
    cost_pct = total_cost / trade_value if trade_value > 0 else 0

    # Return breakdown for reporting
    breakdown = {
        'commission': commission,
        'sec_fee': sec_fee,
        'finra_fee': finra_fee,
        'spread_cost': spread_cost,
        'total': total_cost,
        'is_commission_free': is_commission_free
    }

    return cost_pct, breakdown

def calculate_strategy_returns(df, symbol1, symbol2, pair_analysis, lookback_period=None):
    """Calculate strategy returns with realistic IBKR transaction costs"""
    spread = pair_analysis['spread']
    hedge_ratio = pair_analysis['hedge_ratio']
    lookback = lookback_period if lookback_period else DEFAULT_LOOKBACK_PERIOD
    std_threshold = DEFAULT_STD_DEV_THRESHOLD

    # Calculate bands
    df_strategy = df.copy()
    df_strategy['spread'] = spread
    df_strategy['ma'] = spread.rolling(lookback).mean()
    df_strategy['std'] = spread.rolling(lookback).std()
    df_strategy['upper_band'] = df_strategy['ma'] + std_threshold * df_strategy['std']
    df_strategy['lower_band'] = df_strategy['ma'] - std_threshold * df_strategy['std']
    df_strategy['z_score'] = (spread - df_strategy['ma']) / df_strategy['std']

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

    # Position sizing - calculate actual shares based on position size per leg
    df_strategy['shares_1'] = (df_strategy['position'] * POSITION_SIZE_PER_LEG / df[symbol1]).round()
    df_strategy['shares_2'] = (-df_strategy['position'] * hedge_ratio * POSITION_SIZE_PER_LEG / df[symbol2]).round()

    # Calculate position changes in shares
    df_strategy['shares_1_change'] = df_strategy['shares_1'].diff().fillna(0)
    df_strategy['shares_2_change'] = df_strategy['shares_2'].diff().fillna(0)

    # Calculate IBKR transaction costs
    df_strategy['cost_1'] = 0.0
    df_strategy['cost_2'] = 0.0
    df_strategy['commission_1'] = 0.0
    df_strategy['commission_2'] = 0.0
    df_strategy['sec_fee'] = 0.0
    df_strategy['finra_fee'] = 0.0
    df_strategy['spread_cost'] = 0.0

    for i in range(len(df_strategy)):
        if df_strategy['shares_1_change'].iloc[i] != 0:
            cost_pct, breakdown = calculate_ibkr_transaction_costs(
                symbol1, df[symbol1].iloc[i], df_strategy['shares_1_change'].iloc[i]
            )
            df_strategy.iloc[i, df_strategy.columns.get_loc('cost_1')] = cost_pct
            df_strategy.iloc[i, df_strategy.columns.get_loc('commission_1')] = breakdown.get('commission', 0)
            df_strategy.iloc[i, df_strategy.columns.get_loc('sec_fee')] += breakdown.get('sec_fee', 0)
            df_strategy.iloc[i, df_strategy.columns.get_loc('finra_fee')] += breakdown.get('finra_fee', 0)
            df_strategy.iloc[i, df_strategy.columns.get_loc('spread_cost')] += breakdown.get('spread_cost', 0)

        if df_strategy['shares_2_change'].iloc[i] != 0:
            cost_pct, breakdown = calculate_ibkr_transaction_costs(
                symbol2, df[symbol2].iloc[i], df_strategy['shares_2_change'].iloc[i]
            )
            df_strategy.iloc[i, df_strategy.columns.get_loc('cost_2')] = cost_pct
            df_strategy.iloc[i, df_strategy.columns.get_loc('commission_2')] = breakdown.get('commission', 0)
            df_strategy.iloc[i, df_strategy.columns.get_loc('sec_fee')] += breakdown.get('sec_fee', 0)
            df_strategy.iloc[i, df_strategy.columns.get_loc('finra_fee')] += breakdown.get('finra_fee', 0)
            df_strategy.iloc[i, df_strategy.columns.get_loc('spread_cost')] += breakdown.get('spread_cost', 0)

    # Total transaction costs as percentage
    df_strategy['transaction_costs'] = df_strategy['cost_1'] + df_strategy['cost_2']

    # Position weights for return calculation
    df_strategy['pos_1'] = df_strategy['position']
    df_strategy['pos_2'] = -df_strategy['position'] * hedge_ratio

    # Strategy returns before transaction costs
    df_strategy['strategy_returns_gross'] = (
        df_strategy['pos_1'].shift(1) * df_strategy['returns_1'] +
        df_strategy['pos_2'].shift(1) * df_strategy['returns_2']
    )

    # Net returns after IBKR transaction costs
    df_strategy['strategy_returns'] = (
        df_strategy['strategy_returns_gross'] - df_strategy['transaction_costs']
    )

    return df_strategy

def calculate_performance_metrics(returns_series):
    """Calculate comprehensive performance metrics"""
    returns_series = returns_series.fillna(0)
    if len(returns_series) == 0 or returns_series.std() == 0:
        return None

    total_return = (1 + returns_series).prod() - 1
    num_years = len(returns_series) / 252
    annual_return = (1 + total_return) ** (1 / num_years) - 1 if num_years > 0 else 0
    volatility = returns_series.std() * np.sqrt(252)
    sharpe_ratio = annual_return / volatility if volatility > 0 else 0

    # Drawdown
    cumulative = (1 + returns_series).cumprod()
    peak = cumulative.expanding().max()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min()

    # Other metrics
    positive_returns = returns_series[returns_series > 0]
    negative_returns = returns_series[returns_series < 0]
    win_rate = len(positive_returns) / len(returns_series[returns_series != 0]) if len(returns_series[returns_series != 0]) > 0 else 0

    avg_win = positive_returns.mean() if len(positive_returns) > 0 else 0
    avg_loss = abs(negative_returns.mean()) if len(negative_returns) > 0 else 0
    profit_factor = (positive_returns.sum() / abs(negative_returns.sum())) if len(negative_returns) > 0 and negative_returns.sum() != 0 else 0

    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0

    # Trade statistics
    position_changes = returns_series.diff().fillna(0)
    num_trades = (position_changes != 0).sum()

    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'num_trades': num_trades,
        'num_years': num_years
    }

# =============================================================================
# CHART GENERATION (Base64 encoded for HTML)
# =============================================================================

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    buffer.close()
    plt.close(fig)
    return image_base64

def create_charts(df, df_strategy, symbol1, symbol2, pair_analysis, metrics):
    """Create all charts and return as base64 encoded images"""
    charts = {}

    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')

    # 1. Price Comparison Chart
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df[symbol1], label=symbol1, linewidth=2, color='#2196F3')
    ax2 = ax.twinx()
    ax2.plot(df.index, df[symbol2], label=symbol2, linewidth=2, color='#FF5722')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel(f'{symbol1} Price ($)', color='#2196F3', fontsize=12)
    ax2.set_ylabel(f'{symbol2} Price ($)', color='#FF5722', fontsize=12)
    ax.set_title(f'Price Comparison: {symbol1} vs {symbol2}', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    charts['price_comparison'] = fig_to_base64(fig)

    # 2. Spread with Bands
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df_strategy.index, df_strategy['spread'], alpha=0.8, label='Spread', linewidth=1.5, color='#2196F3')
    ax.plot(df_strategy.index, df_strategy['ma'], color='black', label='Moving Average', linewidth=2)
    ax.plot(df_strategy.index, df_strategy['upper_band'], color='#F44336', linestyle='--', label='Upper Band (+1.5σ)', linewidth=1.5)
    ax.plot(df_strategy.index, df_strategy['lower_band'], color='#4CAF50', linestyle='--', label='Lower Band (-1.5σ)', linewidth=1.5)
    ax.fill_between(df_strategy.index, df_strategy['lower_band'], df_strategy['upper_band'], alpha=0.1, color='gray')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Spread Value', fontsize=12)
    ax.set_title('Spread with Bollinger Bands', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    charts['spread_bands'] = fig_to_base64(fig)

    # 3. Portfolio Performance
    fig, ax = plt.subplots(figsize=(12, 5))
    portfolio_value = INITIAL_CAPITAL * (1 + df_strategy['strategy_returns']).cumprod()
    bh1_value = INITIAL_CAPITAL * (1 + df_strategy['returns_1']).cumprod()
    bh2_value = INITIAL_CAPITAL * (1 + df_strategy['returns_2']).cumprod()

    ax.plot(df_strategy.index, portfolio_value, label='Pairs Strategy', linewidth=3, color='#2196F3')
    ax.plot(df_strategy.index, bh1_value, label=f'{symbol1} Buy & Hold', alpha=0.7, linewidth=1.5, color='#4CAF50')
    ax.plot(df_strategy.index, bh2_value, label=f'{symbol2} Buy & Hold', alpha=0.7, linewidth=1.5, color='#FF5722')
    ax.axhline(y=INITIAL_CAPITAL, color='gray', linestyle=':', alpha=0.7, label='Initial Capital')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax.set_title('Portfolio Performance Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    charts['portfolio_performance'] = fig_to_base64(fig)

    # 4. Z-Score with Signals
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df_strategy.index, df_strategy['z_score'], alpha=0.8, linewidth=1.5, color='#2196F3')
    ax.axhline(y=1.5, color='#F44336', linestyle='--', linewidth=1.5, label='Short Entry (+1.5)')
    ax.axhline(y=-1.5, color='#4CAF50', linestyle='--', linewidth=1.5, label='Long Entry (-1.5)')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, label='Exit (Mean)')
    ax.fill_between(df_strategy.index, -1.5, 1.5, alpha=0.1, color='gray')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Z-Score', fontsize=12)
    ax.set_title('Z-Score with Entry/Exit Signals', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-4, 4)
    charts['zscore'] = fig_to_base64(fig)

    # 5. Position Over Time
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(df_strategy.index, df_strategy['position'], 0,
                    where=(df_strategy['position'] > 0), alpha=0.5, color='#4CAF50', label='Long')
    ax.fill_between(df_strategy.index, df_strategy['position'], 0,
                    where=(df_strategy['position'] < 0), alpha=0.5, color='#F44336', label='Short')
    ax.plot(df_strategy.index, df_strategy['position'], linewidth=0.5, color='black')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Position', fontsize=12)
    ax.set_title('Position Over Time', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.5, 1.5)
    charts['position'] = fig_to_base64(fig)

    # 6. Drawdown
    fig, ax = plt.subplots(figsize=(12, 4))
    cumulative = (1 + df_strategy['strategy_returns']).cumprod()
    peak = cumulative.expanding().max()
    drawdown = ((cumulative - peak) / peak) * 100
    ax.fill_between(df_strategy.index, drawdown, 0, alpha=0.5, color='#F44336')
    ax.plot(df_strategy.index, drawdown, linewidth=1, color='#B71C1C')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Drawdown (%)', fontsize=12)
    ax.set_title('Strategy Drawdown', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    charts['drawdown'] = fig_to_base64(fig)

    # 7. Returns Distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    returns = df_strategy['strategy_returns'].dropna()
    returns = returns[returns != 0]  # Remove zero returns for cleaner histogram
    ax.hist(returns * 100, bins=50, alpha=0.7, color='#2196F3', edgecolor='white')
    ax.axvline(x=returns.mean() * 100, color='#F44336', linestyle='--', linewidth=2, label=f'Mean: {returns.mean()*100:.3f}%')
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
    ax.set_xlabel('Daily Return (%)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Daily Returns', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    charts['returns_dist'] = fig_to_base64(fig)

    return charts, portfolio_value.iloc[-1]

# =============================================================================
# HTML REPORT GENERATION
# =============================================================================

def generate_html_report(inputs, df, df_strategy, pair_analysis, metrics, charts, final_value):
    """Generate comprehensive HTML report"""

    symbol1 = inputs['ticker1']
    symbol2 = inputs['ticker2']

    # Calculate trading statistics
    position_changes = df_strategy['position'].diff().fillna(0)
    entries = ((position_changes != 0) & (df_strategy['position'] != 0)).sum()
    exits = ((position_changes != 0) & (df_strategy['position'] == 0)).sum()
    num_round_trips = min(entries, exits)  # Complete round-trip trades

    # Calculate total transaction costs (percentage)
    total_transaction_costs = df_strategy['transaction_costs'].sum()
    total_transaction_costs_dollars = total_transaction_costs * INITIAL_CAPITAL

    # Calculate IBKR fee breakdown in dollars
    total_commissions = df_strategy['commission_1'].sum() + df_strategy['commission_2'].sum()
    total_sec_fees = df_strategy['sec_fee'].sum()
    total_finra_fees = df_strategy['finra_fee'].sum()
    total_spread_costs = df_strategy['spread_cost'].sum()
    total_fees_dollars = total_commissions + total_sec_fees + total_finra_fees + total_spread_costs

    # Check if ETFs are commission-free
    symbol1_commission_free = symbol1.upper() in COMMISSION_FREE_ETFS
    symbol2_commission_free = symbol2.upper() in COMMISSION_FREE_ETFS

    # Calculate gross vs net returns
    gross_return = (1 + df_strategy['strategy_returns_gross'].fillna(0)).prod() - 1
    net_return = metrics['total_return']

    # Final values
    gross_final_value = INITIAL_CAPITAL * (1 + gross_return)
    net_final_value = final_value

    # Determine quality rating
    quality_score = pair_analysis['quality_score']
    if quality_score >= 80:
        quality_rating = "Excellent"
        quality_color = "#4CAF50"
    elif quality_score >= 60:
        quality_rating = "Good"
        quality_color = "#8BC34A"
    elif quality_score >= 40:
        quality_rating = "Fair"
        quality_color = "#FFC107"
    else:
        quality_rating = "Poor"
        quality_color = "#F44336"

    # Determine if pair is suitable
    is_suitable = pair_analysis['cointegrated'] and pair_analysis['correlation'] > 0.5

    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pairs Trading Backtest Report: {symbol1} vs {symbol2}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #1a237e 0%, #3949ab 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        .header .subtitle {{
            font-size: 1.2em;
            opacity: 0.9;
        }}
        .header .date {{
            margin-top: 15px;
            font-size: 0.9em;
            opacity: 0.8;
        }}
        .section {{
            padding: 30px;
            border-bottom: 1px solid #eee;
        }}
        .section:last-child {{
            border-bottom: none;
        }}
        .section-title {{
            font-size: 1.5em;
            color: #1a237e;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #3949ab;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        .summary-card {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            border-left: 4px solid #3949ab;
        }}
        .summary-card.positive {{
            border-left-color: #4CAF50;
        }}
        .summary-card.negative {{
            border-left-color: #F44336;
        }}
        .summary-card .label {{
            font-size: 0.9em;
            color: #666;
            margin-bottom: 5px;
        }}
        .summary-card .value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #1a237e;
        }}
        .summary-card .value.positive {{
            color: #4CAF50;
        }}
        .summary-card .value.negative {{
            color: #F44336;
        }}
        .metrics-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .metrics-table th, .metrics-table td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        .metrics-table th {{
            background: #f8f9fa;
            font-weight: 600;
            color: #1a237e;
        }}
        .metrics-table tr:hover {{
            background: #f8f9fa;
        }}
        .quality-badge {{
            display: inline-block;
            padding: 8px 20px;
            border-radius: 20px;
            font-weight: bold;
            color: white;
            font-size: 1.1em;
        }}
        .chart-container {{
            margin: 20px 0;
            text-align: center;
        }}
        .chart-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .chart-title {{
            font-size: 1.1em;
            color: #333;
            margin-bottom: 10px;
            font-weight: 600;
        }}
        .warning-box {{
            background: #FFF3E0;
            border-left: 4px solid #FF9800;
            padding: 15px 20px;
            margin: 20px 0;
            border-radius: 0 8px 8px 0;
        }}
        .success-box {{
            background: #E8F5E9;
            border-left: 4px solid #4CAF50;
            padding: 15px 20px;
            margin: 20px 0;
            border-radius: 0 8px 8px 0;
        }}
        .info-box {{
            background: #E3F2FD;
            border-left: 4px solid #2196F3;
            padding: 15px 20px;
            margin: 20px 0;
            border-radius: 0 8px 8px 0;
        }}
        .footer {{
            background: #f8f9fa;
            padding: 20px 30px;
            text-align: center;
            color: #666;
            font-size: 0.9em;
        }}
        .two-column {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
        }}
        @media (max-width: 768px) {{
            .two-column {{
                grid-template-columns: 1fr;
            }}
            .header h1 {{
                font-size: 1.8em;
            }}
        }}
        @media print {{
            body {{
                background: white;
                padding: 0;
            }}
            .container {{
                box-shadow: none;
            }}
            .section {{
                page-break-inside: avoid;
            }}
            .chart-container {{
                page-break-inside: avoid;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Pairs Trading Backtest Report</h1>
            <div class="subtitle">{symbol1} vs {symbol2}</div>
            <div class="date">Backtest Period: {inputs['start_date']} to {inputs['end_date']}</div>
            <div class="date">Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        </div>

        <div class="section">
            <h2 class="section-title">Executive Summary</h2>

            <div class="summary-grid">
                <div class="summary-card {'positive' if metrics['total_return'] > 0 else 'negative'}">
                    <div class="label">Total Return</div>
                    <div class="value {'positive' if metrics['total_return'] > 0 else 'negative'}">{metrics['total_return']:.1%}</div>
                </div>
                <div class="summary-card {'positive' if metrics['annual_return'] > 0 else 'negative'}">
                    <div class="label">Annual Return</div>
                    <div class="value {'positive' if metrics['annual_return'] > 0 else 'negative'}">{metrics['annual_return']:.1%}</div>
                </div>
                <div class="summary-card {'positive' if metrics['sharpe_ratio'] > 1 else ''}">
                    <div class="label">Sharpe Ratio</div>
                    <div class="value">{metrics['sharpe_ratio']:.2f}</div>
                </div>
                <div class="summary-card negative">
                    <div class="label">Max Drawdown</div>
                    <div class="value negative">{metrics['max_drawdown']:.1%}</div>
                </div>
            </div>

            <div class="summary-grid">
                <div class="summary-card">
                    <div class="label">Initial Capital</div>
                    <div class="value">${INITIAL_CAPITAL:,.0f}</div>
                </div>
                <div class="summary-card {'positive' if final_value > INITIAL_CAPITAL else 'negative'}">
                    <div class="label">Final Value</div>
                    <div class="value {'positive' if final_value > INITIAL_CAPITAL else 'negative'}">${final_value:,.0f}</div>
                </div>
                <div class="summary-card">
                    <div class="label">Win Rate</div>
                    <div class="value">{metrics['win_rate']:.1%}</div>
                </div>
                <div class="summary-card">
                    <div class="label">Trading Days</div>
                    <div class="value">{len(df):,}</div>
                </div>
            </div>

            {'<div class="success-box"><strong>Good News!</strong> This pair shows statistical properties suitable for pairs trading. The pair is cointegrated and has adequate correlation.</div>' if is_suitable else '<div class="warning-box"><strong>Caution:</strong> This pair may not be ideal for pairs trading. Consider testing other pairs with stronger cointegration.</div>'}
        </div>

        <div class="section">
            <h2 class="section-title">Pair Quality Analysis</h2>

            <div style="text-align: center; margin-bottom: 20px;">
                <span class="quality-badge" style="background-color: {quality_color};">
                    Quality Score: {quality_score:.0f}/100 - {quality_rating}
                </span>
            </div>

            <table class="metrics-table">
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                    <th>Interpretation</th>
                </tr>
                <tr>
                    <td>Correlation</td>
                    <td>{pair_analysis['correlation']:.4f}</td>
                    <td>{'Strong' if abs(pair_analysis['correlation']) > 0.7 else 'Moderate' if abs(pair_analysis['correlation']) > 0.5 else 'Weak'} correlation between assets</td>
                </tr>
                <tr>
                    <td>Cointegration p-value</td>
                    <td>{pair_analysis['cointegration_pvalue']:.4f}</td>
                    <td>{'Cointegrated (Good)' if pair_analysis['cointegrated'] else 'Not Cointegrated (Caution)'}</td>
                </tr>
                <tr>
                    <td>Hedge Ratio</td>
                    <td>{pair_analysis['hedge_ratio']:.4f}</td>
                    <td>For every 1 unit of {symbol1}, trade {abs(pair_analysis['hedge_ratio']):.2f} units of {symbol2}</td>
                </tr>
                <tr>
                    <td>R-Squared</td>
                    <td>{pair_analysis['r_squared']:.4f}</td>
                    <td>{pair_analysis['r_squared']*100:.1f}% of variance explained</td>
                </tr>
                <tr>
                    <td>Spread Stationarity (ADF p-value)</td>
                    <td>{pair_analysis['adf_pvalue']:.4f}</td>
                    <td>{'Stationary (Mean-Reverting)' if pair_analysis['stationary'] else 'Non-Stationary (Caution)'}</td>
                </tr>
            </table>

            <div class="info-box">
                <strong>What does this mean?</strong><br>
                A good pairs trading candidate should have: high correlation (>0.7), low cointegration p-value (<0.05),
                and a stationary spread (ADF p-value <0.05). This ensures the spread between the two assets tends to revert to its mean.
            </div>
        </div>

        <div class="section">
            <h2 class="section-title">Performance Metrics</h2>

            <div class="two-column">
                <div>
                    <h3 style="margin-bottom: 15px; color: #1a237e;">Return Metrics</h3>
                    <table class="metrics-table">
                        <tr><td>Total Return (Net)</td><td><strong>{metrics['total_return']:.2%}</strong></td></tr>
                        <tr><td>Annual Return (CAGR)</td><td><strong>{metrics['annual_return']:.2%}</strong></td></tr>
                        <tr><td>Volatility (Annual)</td><td>{metrics['volatility']:.2%}</td></tr>
                        <tr><td>Sharpe Ratio</td><td><strong>{metrics['sharpe_ratio']:.2f}</strong></td></tr>
                        <tr><td>Calmar Ratio</td><td>{metrics['calmar_ratio']:.2f}</td></tr>
                    </table>
                </div>
                <div>
                    <h3 style="margin-bottom: 15px; color: #1a237e;">Risk Metrics</h3>
                    <table class="metrics-table">
                        <tr><td>Maximum Drawdown</td><td><strong style="color: #F44336;">{metrics['max_drawdown']:.2%}</strong></td></tr>
                        <tr><td>Win Rate</td><td>{metrics['win_rate']:.1%}</td></tr>
                        <tr><td>Profit Factor</td><td>{metrics['profit_factor']:.2f}</td></tr>
                        <tr><td>Average Win</td><td style="color: #4CAF50;">{metrics['avg_win']*100:.3f}%</td></tr>
                        <tr><td>Average Loss</td><td style="color: #F44336;">-{metrics['avg_loss']*100:.3f}%</td></tr>
                    </table>
                </div>
            </div>
        </div>

        <div class="section">
            <h2 class="section-title">Trading Statistics & IBKR Fees</h2>

            <div class="two-column">
                <div>
                    <h3 style="margin-bottom: 15px; color: #1a237e;">Trade Count</h3>
                    <table class="metrics-table">
                        <tr><td>Total Entries</td><td><strong>{entries}</strong></td></tr>
                        <tr><td>Total Exits</td><td><strong>{exits}</strong></td></tr>
                        <tr><td>Round-Trip Trades</td><td><strong>{num_round_trips}</strong></td></tr>
                        <tr><td>Trading Days</td><td>{len(df):,}</td></tr>
                        <tr><td>Avg Days Between Trades</td><td>{len(df) / max(num_round_trips, 1):.1f}</td></tr>
                    </table>
                </div>
                <div>
                    <h3 style="margin-bottom: 15px; color: #1a237e;">Returns Impact</h3>
                    <table class="metrics-table">
                        <tr><td>Gross Return (Before Fees)</td><td style="color: #4CAF50;">{gross_return:.2%}</td></tr>
                        <tr><td>Total Fees Paid</td><td><strong style="color: #F44336;">${total_fees_dollars:,.2f}</strong></td></tr>
                        <tr><td>Fees as % of Capital</td><td style="color: #F44336;">{(total_fees_dollars/INITIAL_CAPITAL)*100:.3f}%</td></tr>
                        <tr><td>Net Return (After Fees)</td><td><strong>{net_return:.2%}</strong></td></tr>
                        <tr><td>Fee Drag on Returns</td><td style="color: #F44336;">{(gross_return - net_return):.2%}</td></tr>
                    </table>
                </div>
            </div>

            <h3 style="margin: 25px 0 15px; color: #1a237e;">IBKR Fee Breakdown</h3>
            <table class="metrics-table">
                <tr>
                    <th>Fee Type</th>
                    <th>Amount</th>
                    <th>Description</th>
                </tr>
                <tr>
                    <td>Commissions</td>
                    <td>${total_commissions:,.2f}</td>
                    <td>$0.005/share (min $1, max 1%) {' - <span style="color: #4CAF50;">Commission-free ETFs detected!</span>' if (symbol1_commission_free or symbol2_commission_free) else ''}</td>
                </tr>
                <tr>
                    <td>SEC Fee</td>
                    <td>${total_sec_fees:,.2f}</td>
                    <td>0.00278% on sales only</td>
                </tr>
                <tr>
                    <td>FINRA TAF</td>
                    <td>${total_finra_fees:,.2f}</td>
                    <td>0.0166% on sales only</td>
                </tr>
                <tr>
                    <td>Bid-Ask Spread</td>
                    <td>${total_spread_costs:,.2f}</td>
                    <td>Estimated 2.5 bps per trade</td>
                </tr>
                <tr style="background: #f8f9fa; font-weight: bold;">
                    <td>TOTAL FEES</td>
                    <td style="color: #F44336;">${total_fees_dollars:,.2f}</td>
                    <td></td>
                </tr>
            </table>

            <div class="{'success-box' if (symbol1_commission_free and symbol2_commission_free) else 'info-box'}">
                <strong>{'Commission-Free Pair!' if (symbol1_commission_free and symbol2_commission_free) else 'IBKR Fee Summary:'}</strong><br>
                {f'Both {symbol1} and {symbol2} are commission-free ETFs on IBKR, significantly reducing trading costs.' if (symbol1_commission_free and symbol2_commission_free) else f'{symbol1} is commission-free.' if symbol1_commission_free else f'{symbol2} is commission-free.' if symbol2_commission_free else 'Standard IBKR commission rates apply to both symbols.'}
                <br><br>
                Over {num_round_trips} round-trip trades, total IBKR fees of <strong>${total_fees_dollars:,.2f}</strong> reduced your gross return of {gross_return:.2%} to a net return of {net_return:.2%}.
                {'<br><br><span style="color: #F44336;">⚠️ High trading frequency is significantly impacting returns. Consider increasing the lookback period or entry threshold.</span>' if total_fees_dollars > (INITIAL_CAPITAL * 0.02) else ''}
            </div>
        </div>

        <div class="section">
            <h2 class="section-title">Charts & Visualizations</h2>

            <div class="chart-container">
                <div class="chart-title">Price Comparison</div>
                <img src="data:image/png;base64,{charts['price_comparison']}" alt="Price Comparison">
            </div>

            <div class="chart-container">
                <div class="chart-title">Spread with Bollinger Bands</div>
                <img src="data:image/png;base64,{charts['spread_bands']}" alt="Spread with Bands">
            </div>

            <div class="chart-container">
                <div class="chart-title">Portfolio Performance</div>
                <img src="data:image/png;base64,{charts['portfolio_performance']}" alt="Portfolio Performance">
            </div>

            <div class="chart-container">
                <div class="chart-title">Z-Score with Entry/Exit Signals</div>
                <img src="data:image/png;base64,{charts['zscore']}" alt="Z-Score">
            </div>

            <div class="chart-container">
                <div class="chart-title">Position Over Time</div>
                <img src="data:image/png;base64,{charts['position']}" alt="Position">
            </div>

            <div class="chart-container">
                <div class="chart-title">Strategy Drawdown</div>
                <img src="data:image/png;base64,{charts['drawdown']}" alt="Drawdown">
            </div>

            <div class="chart-container">
                <div class="chart-title">Distribution of Daily Returns</div>
                <img src="data:image/png;base64,{charts['returns_dist']}" alt="Returns Distribution">
            </div>
        </div>

        <div class="section">
            <h2 class="section-title">Strategy Explanation</h2>

            <div class="info-box">
                <strong>How the Pairs Trading Strategy Works:</strong>
                <ol style="margin-top: 10px; margin-left: 20px;">
                    <li><strong>Calculate the Spread:</strong> The price difference between {symbol1} and {symbol2}, adjusted by the hedge ratio.</li>
                    <li><strong>Identify Deviation:</strong> When the spread moves more than 1.5 standard deviations from its mean, a trading opportunity exists.</li>
                    <li><strong>Enter Position:</strong> Go long the spread when it's too low (buy {symbol1}, sell {symbol2}), go short when it's too high.</li>
                    <li><strong>Exit Position:</strong> Close the trade when the spread returns to its mean.</li>
                </ol>
            </div>

            <h3 style="margin: 20px 0 15px; color: #1a237e;">Strategy Parameters Used</h3>
            <table class="metrics-table">
                <tr><td>Lookback Period</td><td>{inputs['lookback_period']} days</td></tr>
                <tr><td>Entry Threshold</td><td>{DEFAULT_STD_DEV_THRESHOLD} standard deviations</td></tr>
                <tr><td>Exit Threshold</td><td>Mean (0 standard deviations)</td></tr>
                <tr><td>Initial Capital</td><td>${INITIAL_CAPITAL:,}</td></tr>
                <tr><td>Position Size per Leg</td><td>${POSITION_SIZE_PER_LEG:,}</td></tr>
                <tr><td>Fee Model</td><td>IBKR Realistic (commission + SEC + FINRA + spread)</td></tr>
            </table>
        </div>

        <div class="section">
            <h2 class="section-title">Disclaimer</h2>
            <div class="warning-box">
                <strong>Important:</strong> This report is for educational and research purposes only. Past performance does not guarantee future results.
                Trading involves substantial risk of loss and is not suitable for all investors. Always conduct your own research and consider
                consulting with a financial advisor before making investment decisions.
            </div>
        </div>

        <div class="footer">
            <p>Statistical Arbitrage Pairs Trading Backtester</p>
            <p>Report generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}</p>
            <p style="margin-top: 10px;"><em>To save as PDF: Press Ctrl+P (or Cmd+P on Mac) and select "Save as PDF"</em></p>
        </div>
    </div>
</body>
</html>'''

    return html_content

# =============================================================================
# GITHUB UPLOAD FUNCTIONS
# =============================================================================

def get_script_directory():
    """Get the directory where this script is located"""
    return os.path.dirname(os.path.abspath(__file__))

def ensure_reports_folder():
    """Create reports folder if it doesn't exist"""
    script_dir = get_script_directory()
    reports_dir = os.path.join(script_dir, 'reports')
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)
        print(f"Created reports folder: {reports_dir}")
    return reports_dir

def upload_to_github(report_path, ticker1, ticker2):
    """Commit and push the report to GitHub"""
    import subprocess

    script_dir = get_script_directory()
    repo_dir = os.path.dirname(script_dir)  # Go up one level to repo root

    try:
        # Get relative path for git
        rel_path = os.path.relpath(report_path, repo_dir)

        print("\nUploading to GitHub...")

        # Add the file
        result = subprocess.run(
            ['git', 'add', rel_path],
            cwd=repo_dir,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"Error adding file: {result.stderr}")
            return False

        # Commit
        commit_msg = f"Add backtest report: {ticker1} vs {ticker2}"
        result = subprocess.run(
            ['git', 'commit', '-m', commit_msg],
            cwd=repo_dir,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            if "nothing to commit" in result.stdout or "nothing to commit" in result.stderr:
                print("File already committed.")
            else:
                print(f"Error committing: {result.stderr}")
                return False

        # Push
        result = subprocess.run(
            ['git', 'push'],
            cwd=repo_dir,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            # Try with origin main
            result = subprocess.run(
                ['git', 'push', 'origin', 'main'],
                cwd=repo_dir,
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                print(f"Error pushing: {result.stderr}")
                print("\nYou can manually push later with: git push origin main")
                return False

        print("Successfully uploaded to GitHub!")
        return True

    except FileNotFoundError:
        print("Git is not installed or not in PATH.")
        print("Please install Git or manually commit and push the report.")
        return False
    except Exception as e:
        print(f"Error uploading to GitHub: {e}")
        return False

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_backtest():
    """Main function to run the backtest"""

    # Get user inputs
    inputs = get_user_inputs()
    if inputs is None:
        return

    print("\n" + "="*60)
    print("RUNNING BACKTEST...")
    print("="*60)

    # Download data
    df, error = download_pair_data(
        inputs['ticker1'],
        inputs['ticker2'],
        inputs['start_date'],
        inputs['end_date']
    )

    if df is None:
        print(f"\nERROR: {error}")
        return

    # Analyze pair
    print("Analyzing pair quality...")
    pair_analysis = analyze_pair_quality(df, inputs['ticker1'], inputs['ticker2'], USE_LOG_PRICES)

    print(f"  - Correlation: {pair_analysis['correlation']:.4f}")
    print(f"  - Cointegration p-value: {pair_analysis['cointegration_pvalue']:.4f}")
    print(f"  - Quality Score: {pair_analysis['quality_score']:.1f}/100")

    # Calculate strategy
    print(f"Calculating strategy returns (lookback: {inputs['lookback_period']} days)...")
    df_strategy = calculate_strategy_returns(df, inputs['ticker1'], inputs['ticker2'], pair_analysis, inputs['lookback_period'])

    # Calculate metrics
    print("Calculating performance metrics...")
    metrics = calculate_performance_metrics(df_strategy['strategy_returns'])

    if metrics is None:
        print("\nERROR: Could not calculate performance metrics.")
        return

    print(f"  - Total Return: {metrics['total_return']:.2%}")
    print(f"  - Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"  - Max Drawdown: {metrics['max_drawdown']:.2%}")

    # Generate charts
    print("Generating charts...")
    charts, final_value = create_charts(
        df, df_strategy, inputs['ticker1'], inputs['ticker2'],
        pair_analysis, metrics
    )

    # Generate HTML report
    print("Generating HTML report...")
    html_content = generate_html_report(
        inputs, df, df_strategy, pair_analysis, metrics, charts, final_value
    )

    # Ensure reports folder exists and save report there
    reports_dir = ensure_reports_folder()
    report_filename = f"Backtest_Report_{inputs['ticker1']}_{inputs['ticker2']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    report_path = os.path.join(reports_dir, report_filename)

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print("\n" + "="*60)
    print("BACKTEST COMPLETE!")
    print("="*60)
    print(f"\nReport saved to: {report_path}")
    print(f"\nTo view the report:")
    print(f"  1. Open the HTML file in any web browser")
    print(f"  2. To save as PDF: Press Ctrl+P (or Cmd+P) and select 'Save as PDF'")

    # Try to open the report automatically
    try:
        import webbrowser
        webbrowser.open('file://' + os.path.abspath(report_path))
        print("\nReport opened in your default web browser.")
    except:
        pass

    # Ask about GitHub upload
    print("\n" + "-"*60)
    upload_choice = input("Would you like to upload this report to GitHub? (y/n): ").strip().lower()

    if upload_choice == 'y':
        upload_to_github(report_path, inputs['ticker1'], inputs['ticker2'])
    else:
        print("\nReport saved locally. You can upload later by running:")
        print(f"  git add \"{report_path}\"")
        print(f"  git commit -m \"Add backtest report\"")
        print(f"  git push origin main")

    print("\n" + "="*60)
    return report_path

if __name__ == "__main__":
    run_backtest()
