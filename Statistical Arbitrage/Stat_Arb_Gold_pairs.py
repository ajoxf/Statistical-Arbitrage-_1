# -*- coding: utf-8 -*-
"""
Created on Sat Jun 28 00:22:49 2025

@author: AJ
"""
# -*- coding: utf-8 -*-
"""
Improved Pairs Trading Strategy for Gold/Mining Assets
@author: AJ
Enhanced version with better diagnostics and parameters
"""
import statsmodels.api as sm
import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import warnings
warnings.filterwarnings('ignore')
import sys

# =============================================================================
# IMPROVED CONFIGURATION - OPTIMIZED FOR GOLD/MINING PAIRS
# =============================================================================

# Better pairs for gold trading
PAIRS_TO_TEST = [
    ('GLD', 'GDX'),   # Physical gold vs miners (best for mean reversion)
    ('GLD', 'NEM'),   # Physical gold vs largest miner
    ('NEM', 'AEM'),   # Two major miners
    ('GDX', 'GDXJ'),  # Large vs junior miners
    ('GLD', 'GOLD'),  # Physical vs Barrick Gold
]

# Current pair (change this to test different pairs)
SYMBOL_1, SYMBOL_2 = PAIRS_TO_TEST[4]  # Start with GLD vs GDX

# Improved time period - longer for better statistics
START_DATE = '2018-01-01'
END_DATE = '2025-06-01'

# Optimized strategy parameters for commodities
LOOKBACK_PERIOD = 90      # Longer lookback for commodities
STD_DEV_THRESHOLD = 1   # Lower threshold for more trades
FORMATION_PERIOD = 120    # Longer formation period
USE_LOG_PRICES = True     # Use log transformation

# Portfolio parameters
INITIAL_CAPITAL = 100000

# =============================================================================
# ENHANCED DATA DOWNLOAD WITH PAIR VALIDATION
# =============================================================================

def download_and_validate_pair(symbol1, symbol2, start_date, end_date):
    """Download and validate a trading pair"""
    print(f"Testing pair: {symbol1} vs {symbol2}")
    print(f"Period: {start_date} to {end_date}")
    print("="*60)
    
    try:
        # Download data
        data1 = yf.download(symbol1, start=start_date, end=end_date, progress=False)
        data2 = yf.download(symbol2, start=start_date, end=end_date, progress=False)
        
        if data1.empty or data2.empty:
            raise ValueError(f"No data found for {symbol1} or {symbol2}")
        
        # Create combined dataframe
        df = pd.concat([data1['Close'], data2['Close']], axis=1)
        df.columns = [symbol1, symbol2]
        df = df.dropna()
        
        print(f"Data shape: {df.shape}")
        print(f"Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
        
        return df, True
        
    except Exception as e:
        print(f"Failed to download {symbol1} vs {symbol2}: {e}")
        return None, False

# Download data
df, success = download_and_validate_pair(SYMBOL_1, SYMBOL_2, START_DATE, END_DATE)

if not success:
    print("Trying alternative pairs...")
    for pair in PAIRS_TO_TEST[1:]:
        SYMBOL_1, SYMBOL_2 = pair
        df, success = download_and_validate_pair(SYMBOL_1, SYMBOL_2, START_DATE, END_DATE)
        if success:
            break
    
    if not success:
        print("All pairs failed. Using AAPL vs MSFT for demonstration.")
        SYMBOL_1, SYMBOL_2 = 'AAPL', 'MSFT'
        df, _ = download_and_validate_pair(SYMBOL_1, SYMBOL_2, START_DATE, END_DATE)

# =============================================================================
# ENHANCED COINTEGRATION AND RELATIONSHIP ANALYSIS
# =============================================================================

def analyze_pair_relationship(df, symbol1, symbol2, use_log=True):
    """Comprehensive analysis of pair relationship"""
    print(f"\n=== PAIR RELATIONSHIP ANALYSIS ===")
    
    # Transform prices if requested
    if use_log:
        prices1 = np.log(df[symbol1])
        prices2 = np.log(df[symbol2])
        print("Using log-transformed prices")
    else:
        prices1 = df[symbol1]
        prices2 = df[symbol2]
        print("Using level prices")
    
    # Basic correlation
    correlation = prices1.corr(prices2)
    print(f"Correlation: {correlation:.4f}")
    
    # Cointegration test (Engle-Granger)
    try:
        coint_score, coint_pvalue, _ = coint(prices1, prices2)
        print(f"Cointegration p-value: {coint_pvalue:.4f}")
        if coint_pvalue < 0.05:
            print("✓ Pair is cointegrated (good for pairs trading)")
            cointegrated = True
        else:
            print("✗ Pair is NOT cointegrated (may not mean revert)")
            cointegrated = False
    except:
        print("Could not perform cointegration test")
        cointegrated = False
    
    return prices1, prices2, correlation, cointegrated

# Analyze the pair
log_prices1, log_prices2, correlation, is_cointegrated = analyze_pair_relationship(
    df, SYMBOL_1, SYMBOL_2, USE_LOG_PRICES)

# Update dataframe with transformed prices
if USE_LOG_PRICES:
    df[f'{SYMBOL_1}_log'] = log_prices1
    df[f'{SYMBOL_2}_log'] = log_prices2
    price_col1 = f'{SYMBOL_1}_log'
    price_col2 = f'{SYMBOL_2}_log'
else:
    price_col1 = SYMBOL_1
    price_col2 = SYMBOL_2

# =============================================================================
# IMPROVED HEDGE RATIO AND SPREAD CALCULATION
# =============================================================================

# Adjust formation period if necessary
if len(df) < FORMATION_PERIOD:
    FORMATION_PERIOD = max(60, len(df) // 3)
    print(f"Adjusted formation period to {FORMATION_PERIOD}")

# Calculate hedge ratio using transformed prices
X = sm.add_constant(df[price_col2].iloc[:FORMATION_PERIOD])
model = sm.OLS(df[price_col1].iloc[:FORMATION_PERIOD], X)
model = model.fit()
hedge_ratio = model.params[price_col2]
intercept = model.params['const']

print(f'\n=== HEDGE RATIO ANALYSIS ===')
print(f'Hedge Ratio: {hedge_ratio:.4f}')
print(f'Intercept: {intercept:.4f}')
print(f'R-squared: {model.rsquared:.4f}')

# Calculate spread
df['spread'] = df[price_col1] - hedge_ratio * df[price_col2]

# Enhanced stationarity test
adf_result = adfuller(df['spread'].dropna(), maxlag=10)
print(f'\n=== STATIONARITY TEST (ADF) ===')
print(f'ADF Statistic: {adf_result[0]:.4f}')
print(f'p-value: {adf_result[1]:.4f}')

if adf_result[1] < 0.05:
    print("✓ Spread is stationary - Good for pairs trading!")
    stationary = True
else:
    print("✗ Spread is NOT stationary - Consider different parameters")
    stationary = False

# =============================================================================
# ADAPTIVE PARAMETER ADJUSTMENT
# =============================================================================

# Adjust parameters based on pair characteristics
if not is_cointegrated or not stationary:
    print(f"\n=== PARAMETER ADJUSTMENT ===")
    print("Pair shows weak mean reversion. Adjusting parameters...")
    STD_DEV_THRESHOLD = 1.0  # More aggressive
    LOOKBACK_PERIOD = min(120, len(df) // 4)  # Longer lookback
    print(f"New std dev threshold: {STD_DEV_THRESHOLD}")
    print(f"New lookback period: {LOOKBACK_PERIOD}")

# =============================================================================
# ENHANCED TRADING SIGNALS WITH MULTIPLE METHODS
# =============================================================================

# Method 1: Standard deviation bands
df['moving_average'] = df['spread'].rolling(LOOKBACK_PERIOD).mean()
df['moving_std'] = df['spread'].rolling(LOOKBACK_PERIOD).std()
df['upper_band'] = df['moving_average'] + STD_DEV_THRESHOLD * df['moving_std']
df['lower_band'] = df['moving_average'] - STD_DEV_THRESHOLD * df['moving_std']

# Method 2: Percentile bands (alternative approach)
df['upper_percentile'] = df['spread'].rolling(LOOKBACK_PERIOD).quantile(0.95)
df['lower_percentile'] = df['spread'].rolling(LOOKBACK_PERIOD).quantile(0.05)

# Method 3: Z-score
df['z_score'] = (df['spread'] - df['moving_average']) / df['moving_std']

# Initialize positions
df['position'] = 0.0
df['position_percentile'] = 0.0

# Apply trading logic (standard deviation method)
for i in range(LOOKBACK_PERIOD, len(df)):
    prev_pos = df['position'].iloc[i-1]
    current_spread = df['spread'].iloc[i]
    ma = df['moving_average'].iloc[i]
    upper = df['upper_band'].iloc[i]
    lower = df['lower_band'].iloc[i]
    
    # Exit conditions
    if prev_pos == 1 and current_spread >= ma:
        df.iloc[i, df.columns.get_loc('position')] = 0
    elif prev_pos == -1 and current_spread <= ma:
        df.iloc[i, df.columns.get_loc('position')] = 0
    # Entry conditions
    elif prev_pos == 0:
        if current_spread < lower:
            df.iloc[i, df.columns.get_loc('position')] = 1  # Long spread
        elif current_spread > upper:
            df.iloc[i, df.columns.get_loc('position')] = -1  # Short spread
        else:
            df.iloc[i, df.columns.get_loc('position')] = 0
    # Hold position
    else:
        df.iloc[i, df.columns.get_loc('position')] = prev_pos

# Apply percentile method
for i in range(LOOKBACK_PERIOD, len(df)):
    prev_pos = df['position_percentile'].iloc[i-1]
    current_spread = df['spread'].iloc[i]
    ma = df['moving_average'].iloc[i]
    upper = df['upper_percentile'].iloc[i]
    lower = df['lower_percentile'].iloc[i]
    
    # Similar logic with percentile bands
    if prev_pos == 1 and current_spread >= ma:
        df.iloc[i, df.columns.get_loc('position_percentile')] = 0
    elif prev_pos == -1 and current_spread <= ma:
        df.iloc[i, df.columns.get_loc('position_percentile')] = 0
    elif prev_pos == 0:
        if current_spread < lower:
            df.iloc[i, df.columns.get_loc('position_percentile')] = 1
        elif current_spread > upper:
            df.iloc[i, df.columns.get_loc('position_percentile')] = -1
        else:
            df.iloc[i, df.columns.get_loc('position_percentile')] = 0
    else:
        df.iloc[i, df.columns.get_loc('position_percentile')] = prev_pos

# =============================================================================
# RETURN CALCULATIONS FOR BOTH METHODS
# =============================================================================

# Calculate returns on original prices (not log prices)
df['returns_1'] = df[SYMBOL_1].pct_change()
df['returns_2'] = df[SYMBOL_2].pct_change()

# Position sizing
df['pos_1'] = df['position']
df['pos_2'] = -df['position'] * hedge_ratio

df['pos_1_pct'] = df['position_percentile']
df['pos_2_pct'] = -df['position_percentile'] * hedge_ratio

# Strategy returns
df['strategy_returns'] = (df['pos_1'].shift(1) * df['returns_1'] + 
                         df['pos_2'].shift(1) * df['returns_2'])

df['strategy_returns_pct'] = (df['pos_1_pct'].shift(1) * df['returns_1'] + 
                             df['pos_2_pct'].shift(1) * df['returns_2'])

# Cumulative returns
df['cum_strategy'] = (1 + df['strategy_returns'].fillna(0)).cumprod()
df['cum_strategy_pct'] = (1 + df['strategy_returns_pct'].fillna(0)).cumprod()
df['cum_buy_hold_1'] = (1 + df['returns_1'].fillna(0)).cumprod()
df['cum_buy_hold_2'] = (1 + df['returns_2'].fillna(0)).cumprod()

# Portfolio values
df['portfolio_std'] = INITIAL_CAPITAL * df['cum_strategy']
df['portfolio_pct'] = INITIAL_CAPITAL * df['cum_strategy_pct']
df['portfolio_bh1'] = INITIAL_CAPITAL * df['cum_buy_hold_1']
df['portfolio_bh2'] = INITIAL_CAPITAL * df['cum_buy_hold_2']

# =============================================================================
# COMPREHENSIVE PERFORMANCE ANALYSIS
# =============================================================================

def calculate_performance_metrics(returns, name):
    """Enhanced performance metrics calculation"""
    returns = returns.fillna(0)
    if len(returns) == 0:
        return {}
    
    total_return = (1 + returns).prod() - 1
    annual_return = (1 + total_return) ** (252 / len(returns)) - 1
    volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = annual_return / volatility if volatility > 0 else 0
    
    # Drawdown calculation
    cumulative = (1 + returns).cumprod()
    peak = cumulative.expanding().max()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min()
    
    # Win rate
    win_rate = (returns > 0).sum() / len(returns)
    
    # Calmar ratio
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0
    
    return {
        'name': name,
        'total_return': total_return,
        'annual_return': annual_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio,
        'win_rate': win_rate,
        'num_observations': len(returns)
    }

# Calculate metrics for all strategies
std_metrics = calculate_performance_metrics(df['strategy_returns'], 'Std Dev Strategy')
pct_metrics = calculate_performance_metrics(df['strategy_returns_pct'], 'Percentile Strategy')
bh1_metrics = calculate_performance_metrics(df['returns_1'], f'{SYMBOL_1} Buy & Hold')
bh2_metrics = calculate_performance_metrics(df['returns_2'], f'{SYMBOL_2} Buy & Hold')

# =============================================================================
# RESULTS DISPLAY
# =============================================================================

print(f"\n{'='*60}")
print(f"PAIRS TRADING RESULTS: {SYMBOL_1} vs {SYMBOL_2}")
print(f"{'='*60}")

print(f"\nPAIR QUALITY ASSESSMENT:")
print(f"  Correlation: {correlation:.4f}")
print(f"  Cointegrated: {'Yes' if is_cointegrated else 'No'}")
print(f"  Spread Stationary: {'Yes' if stationary else 'No'}")
print(f"  Hedge Ratio: {hedge_ratio:.4f}")
print(f"  R-squared: {model.rsquared:.4f}")

print(f"\nSTRATEGY COMPARISON:")
strategies = [std_metrics, pct_metrics, bh1_metrics, bh2_metrics]

for metrics in strategies:
    if metrics:
        print(f"\n{metrics['name']}:")
        if metrics['name'] in ['Std Dev Strategy', 'Percentile Strategy']:
            final_value = df[f"portfolio_{'std' if 'Std' in metrics['name'] else 'pct'}"].iloc[-1]
            print(f"  Final Value: ${final_value:,.0f}")
        else:
            final_value = INITIAL_CAPITAL * (1 + metrics['total_return'])
            print(f"  Final Value: ${final_value:,.0f}")
        
        print(f"  Total Return: {metrics['total_return']:.2%}")
        print(f"  Annual Return: {metrics['annual_return']:.2%}")
        print(f"  Volatility: {metrics['volatility']:.2%}")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"  Calmar Ratio: {metrics['calmar_ratio']:.2f}")
        print(f"  Win Rate: {metrics['win_rate']:.1%}")

# Trading summary
print(f"\nTRADING SUMMARY:")
print(f"  Total Days: {len(df)}")

# Standard deviation method
pos_summary = df['position'].value_counts().sort_index()
print(f"  Std Dev Method:")
for pos, count in pos_summary.items():
    pos_name = "No Position" if pos == 0 else "Long Spread" if pos == 1 else "Short Spread"
    print(f"    {pos_name}: {count} days ({count/len(df)*100:.1f}%)")

# Calculate number of trades
trades = (df['position'].diff() != 0).sum()
print(f"  Number of Trades: {trades}")

if trades > 0:
    avg_duration = len(df) / trades
    print(f"  Average Trade Duration: {avg_duration:.1f} days")

# =============================================================================
# ENHANCED VISUALIZATIONS
# =============================================================================

plt.style.use('default')
fig = plt.figure(figsize=(20, 16))

# Create a 3x2 subplot layout
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], hspace=0.3, wspace=0.3)

# 1. Price comparison
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(df.index, df[SYMBOL_1], label=SYMBOL_1, linewidth=2)
ax1_twin = ax1.twinx()
ax1_twin.plot(df.index, df[SYMBOL_2], label=SYMBOL_2, color='red', linewidth=2)
ax1.set_title(f'{SYMBOL_1} vs {SYMBOL_2} Price Comparison')
ax1.set_ylabel(f'{SYMBOL_1} Price', color='blue')
ax1_twin.set_ylabel(f'{SYMBOL_2} Price', color='red')
ax1.legend(loc='upper left')
ax1_twin.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# 2. Spread with bands
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(df.index, df['spread'], label='Spread', alpha=0.7)
ax2.plot(df.index, df['moving_average'], label='MA', color='black')
ax2.plot(df.index, df['upper_band'], label='Upper Band', color='red', linestyle='--')
ax2.plot(df.index, df['lower_band'], label='Lower Band', color='green', linestyle='--')
ax2.fill_between(df.index, df['upper_band'], df['lower_band'], alpha=0.1)

# Add entry signals
long_signals = df[df['position'] == 1].index
short_signals = df[df['position'] == -1].index
if len(long_signals) > 0:
    ax2.scatter(long_signals, df.loc[long_signals, 'spread'], 
               color='green', marker='^', s=30, alpha=0.7, label='Long Entry')
if len(short_signals) > 0:
    ax2.scatter(short_signals, df.loc[short_signals, 'spread'], 
               color='red', marker='v', s=30, alpha=0.7, label='Short Entry')

ax2.set_title('Spread with Trading Signals')
ax2.set_ylabel('Spread')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Portfolio performance comparison
ax3 = fig.add_subplot(gs[1, :])
ax3.plot(df.index, df['portfolio_std'], label='Std Dev Strategy', linewidth=3)
ax3.plot(df.index, df['portfolio_pct'], label='Percentile Strategy', linewidth=2, linestyle='--')
ax3.plot(df.index, df['portfolio_bh1'], label=f'{SYMBOL_1} Buy & Hold', alpha=0.7)
ax3.plot(df.index, df['portfolio_bh2'], label=f'{SYMBOL_2} Buy & Hold', alpha=0.7)
ax3.axhline(y=INITIAL_CAPITAL, color='black', linestyle=':', alpha=0.5)
ax3.set_title('Portfolio Performance Comparison')
ax3.set_ylabel('Portfolio Value ($)')
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Z-score of spread
ax4 = fig.add_subplot(gs[2, 0])
ax4.plot(df.index, df['z_score'], alpha=0.7)
ax4.axhline(y=2, color='red', linestyle='--', alpha=0.7, label='±2σ')
ax4.axhline(y=-2, color='red', linestyle='--', alpha=0.7)
ax4.axhline(y=1.5, color='orange', linestyle='--', alpha=0.7, label='±1.5σ')
ax4.axhline(y=-1.5, color='orange', linestyle='--', alpha=0.7)
ax4.axhline(y=0, color='black', alpha=0.3)
ax4.set_title('Spread Z-Score')
ax4.set_ylabel('Z-Score')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Position allocation
ax5 = fig.add_subplot(gs[2, 1])
ax5.plot(df.index, df['position'], linewidth=2)
ax5.fill_between(df.index, df['position'], 0, 
                where=(df['position'] > 0), alpha=0.3, color='green', label='Long Spread')
ax5.fill_between(df.index, df['position'], 0, 
                where=(df['position'] < 0), alpha=0.3, color='red', label='Short Spread')
ax5.set_title('Position Allocation Over Time')
ax5.set_ylabel('Position')
ax5.set_ylim(-1.1, 1.1)
ax5.legend()
ax5.grid(True, alpha=0.3)

plt.suptitle(f'Pairs Trading Analysis: {SYMBOL_1} vs {SYMBOL_2}', fontsize=16, fontweight='bold')
plt.show()

print(f"\n{'='*60}")
print(f"ANALYSIS COMPLETE")
print(f"{'='*60}")

# Recommendation
if is_cointegrated and stationary and std_metrics['sharpe_ratio'] > 0.5:
    print("✓ This pair shows good potential for pairs trading!")
elif std_metrics['sharpe_ratio'] > 0:
    print("⚠ Marginal pair - consider parameter optimization")
else:
    print("✗ Poor pair for current strategy - try different pairs or parameters")

print(f"\nNext steps:")
print(f"1. Try other pairs: {PAIRS_TO_TEST}")
print(f"2. Optimize parameters (lookback, threshold)")
print(f"3. Consider transaction costs")
print(f"4. Test on out-of-sample data")
