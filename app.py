# -*- coding: utf-8 -*-
"""
Statistical Arbitrage Pairs Trading Backtester - Web App
Streamlit-based web interface for non-technical users

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
from datetime import datetime, timedelta
from io import BytesIO
import base64
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Pairs Trading Backtester",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CONSTANTS
# =============================================================================

INITIAL_CAPITAL = 100000
POSITION_SIZE_PER_LEG = 50000
MIN_OBSERVATIONS = 100

IBKR_FEES = {
    'commission_per_share': 0.005,
    'min_commission': 1.00,
    'max_commission_pct': 0.01,
    'sec_fee_rate': 0.0000278,
    'finra_taf_per_share': 0.000166,
    'finra_taf_max': 8.30,
    'bid_ask_spread_bps': 2.5,
    'bid_ask_spread_bps_forex': 0.5,
    'forex_commission_bps': 2.0,
    'forex_min_commission': 2.00,
}

COMMISSION_FREE_ETFS = {
    'SPY', 'QQQ', 'IWM', 'GLD', 'SLV', 'TLT', 'HYG', 'LQD', 'EFA', 'EEM',
    'VTI', 'VEA', 'VWO', 'XLF', 'XLK', 'XLE', 'XLV', 'XLI', 'XLB', 'XLP',
    'XLY', 'XLU', 'GDX', 'GDXJ', 'USO', 'UNG', 'TIP', 'IEF', 'AGG', 'BND',
    'VNQ', 'ARKK', 'DIA', 'VOO', 'IVV', 'VTV', 'VUG', 'VIG', 'SCHD'
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def is_forex_symbol(symbol):
    """Check if symbol is a forex/currency pair"""
    symbol_upper = symbol.upper()
    if symbol_upper.endswith('=X'):
        return True
    forex_pairs = ['EUR', 'USD', 'GBP', 'JPY', 'CHF', 'AUD', 'CAD', 'NZD', 'CNY', 'HKD']
    for curr1 in forex_pairs:
        for curr2 in forex_pairs:
            if curr1 != curr2 and (f'{curr1}{curr2}' in symbol_upper or f'{curr2}{curr1}' in symbol_upper):
                return True
    return False


@st.cache_data(ttl=3600)
def download_pair_data(symbol1, symbol2, start_date, end_date):
    """Download and prepare data for a pair"""
    try:
        data1 = yf.download(symbol1, start=start_date, end=end_date, progress=False)
        data2 = yf.download(symbol2, start=start_date, end=end_date, progress=False)

        if data1.empty:
            return None, f"No data found for {symbol1}. Please check the ticker symbol."
        if data2.empty:
            return None, f"No data found for {symbol2}. Please check the ticker symbol."

        if isinstance(data1.columns, pd.MultiIndex):
            data1.columns = data1.columns.get_level_values(0)
        if isinstance(data2.columns, pd.MultiIndex):
            data2.columns = data2.columns.get_level_values(0)

        df = pd.concat([data1['Close'], data2['Close']], axis=1)
        df.columns = [symbol1, symbol2]
        df = df.dropna()

        if len(df) < MIN_OBSERVATIONS:
            return None, f"Insufficient data: only {len(df)} trading days. Need at least {MIN_OBSERVATIONS}."

        return df, None

    except Exception as e:
        return None, f"Error downloading data: {str(e)}"


def analyze_pair_quality(df, symbol1, symbol2, use_log=True):
    """Comprehensive pair quality analysis"""
    if use_log:
        prices1 = np.log(df[symbol1])
        prices2 = np.log(df[symbol2])
    else:
        prices1 = df[symbol1]
        prices2 = df[symbol2]

    correlation = prices1.corr(prices2)

    try:
        coint_score, coint_pvalue, _ = coint(prices1, prices2)
        cointegrated = coint_pvalue < 0.05
    except:
        coint_pvalue = 1.0
        cointegrated = False

    X = sm.add_constant(prices2)
    model = sm.OLS(prices1, X).fit()
    hedge_ratio = model.params.iloc[1] if hasattr(model.params, 'iloc') else model.params[1]
    r_squared = model.rsquared

    spread = prices1 - hedge_ratio * prices2
    try:
        adf_stat, adf_pvalue, _, _, _, _ = adfuller(spread.dropna(), maxlag=10)
        stationary = adf_pvalue < 0.05
    except:
        adf_pvalue = 1.0
        stationary = False

    return {
        'correlation': correlation,
        'cointegration_pvalue': coint_pvalue,
        'cointegrated': cointegrated,
        'hedge_ratio': hedge_ratio,
        'r_squared': r_squared,
        'adf_pvalue': adf_pvalue,
        'stationary': stationary,
        'prices1': prices1,
        'prices2': prices2,
        'spread': spread
    }


def calculate_ibkr_transaction_costs(symbol, price, shares_traded):
    """Calculate realistic IBKR transaction costs"""
    if shares_traded == 0 or price == 0:
        return 0.0, {}

    trade_value = abs(shares_traded * price)
    num_shares = abs(shares_traded)
    is_sell = shares_traded < 0
    is_forex = is_forex_symbol(symbol)

    if is_forex:
        commission = max(
            IBKR_FEES['forex_min_commission'],
            trade_value * (IBKR_FEES['forex_commission_bps'] / 10000)
        )
        sec_fee = 0
        finra_fee = 0
        spread_cost = trade_value * (IBKR_FEES['bid_ask_spread_bps_forex'] / 10000)
    else:
        if symbol.upper() in COMMISSION_FREE_ETFS:
            commission = 0.0
        else:
            commission = max(
                IBKR_FEES['min_commission'],
                min(num_shares * IBKR_FEES['commission_per_share'],
                    trade_value * IBKR_FEES['max_commission_pct'])
            )

        sec_fee = trade_value * IBKR_FEES['sec_fee_rate'] if is_sell else 0

        if is_sell:
            finra_fee = min(
                num_shares * IBKR_FEES['finra_taf_per_share'],
                IBKR_FEES['finra_taf_max']
            )
        else:
            finra_fee = 0

        spread_cost = trade_value * (IBKR_FEES['bid_ask_spread_bps'] / 10000)

    total_cost = commission + sec_fee + finra_fee + spread_cost

    breakdown = {
        'commission': commission,
        'sec_fee': sec_fee,
        'finra_fee': finra_fee,
        'spread_cost': spread_cost,
        'total': total_cost,
    }

    return total_cost / trade_value if trade_value > 0 else 0, breakdown


def calculate_strategy_returns(df, symbol1, symbol2, pair_analysis, lookback_period, std_threshold, stop_loss_threshold):
    """Calculate strategy returns with transaction costs and optional stop-loss"""
    spread = pair_analysis['spread']
    hedge_ratio = pair_analysis['hedge_ratio']
    use_stop_loss = stop_loss_threshold > 0

    df_strategy = df.copy()
    df_strategy['spread'] = spread
    df_strategy['ma'] = spread.rolling(lookback_period).mean()
    df_strategy['std'] = spread.rolling(lookback_period).std()
    df_strategy['upper_band'] = df_strategy['ma'] + std_threshold * df_strategy['std']
    df_strategy['lower_band'] = df_strategy['ma'] - std_threshold * df_strategy['std']
    df_strategy['z_score'] = (spread - df_strategy['ma']) / df_strategy['std']

    df_strategy['stop_loss_exit'] = False
    df_strategy['position'] = 0.0

    for i in range(lookback_period, len(df_strategy)):
        prev_pos = df_strategy['position'].iloc[i-1]
        current_zscore = df_strategy['z_score'].iloc[i]
        current_spread = df_strategy['spread'].iloc[i]
        ma = df_strategy['ma'].iloc[i]
        upper = df_strategy['upper_band'].iloc[i]
        lower = df_strategy['lower_band'].iloc[i]

        stop_loss_triggered = False
        if use_stop_loss and prev_pos != 0:
            if prev_pos == 1 and current_zscore < -stop_loss_threshold:
                stop_loss_triggered = True
            elif prev_pos == -1 and current_zscore > stop_loss_threshold:
                stop_loss_triggered = True

        if stop_loss_triggered:
            df_strategy.iloc[i, df_strategy.columns.get_loc('position')] = 0
            df_strategy.iloc[i, df_strategy.columns.get_loc('stop_loss_exit')] = True
        elif prev_pos == 1 and current_spread >= ma:
            df_strategy.iloc[i, df_strategy.columns.get_loc('position')] = 0
        elif prev_pos == -1 and current_spread <= ma:
            df_strategy.iloc[i, df_strategy.columns.get_loc('position')] = 0
        elif prev_pos == 0:
            if current_spread < lower:
                df_strategy.iloc[i, df_strategy.columns.get_loc('position')] = 1
            elif current_spread > upper:
                df_strategy.iloc[i, df_strategy.columns.get_loc('position')] = -1
            else:
                df_strategy.iloc[i, df_strategy.columns.get_loc('position')] = 0
        else:
            df_strategy.iloc[i, df_strategy.columns.get_loc('position')] = prev_pos

    df_strategy['returns_1'] = df[symbol1].pct_change()
    df_strategy['returns_2'] = df[symbol2].pct_change()

    df_strategy['shares_1'] = 0.0
    df_strategy['shares_2'] = 0.0

    current_shares_1 = 0.0
    current_shares_2 = 0.0

    for i in range(len(df_strategy)):
        current_pos = df_strategy['position'].iloc[i]
        prev_pos = df_strategy['position'].iloc[i-1] if i > 0 else 0

        if current_pos != prev_pos:
            if current_pos != 0:
                entry_price_1 = df[symbol1].iloc[i]
                entry_price_2 = df[symbol2].iloc[i]
                current_shares_1 = round(current_pos * POSITION_SIZE_PER_LEG / entry_price_1)
                current_shares_2 = round(-current_pos * hedge_ratio * POSITION_SIZE_PER_LEG / entry_price_2)
            else:
                current_shares_1 = 0.0
                current_shares_2 = 0.0

        df_strategy.iloc[i, df_strategy.columns.get_loc('shares_1')] = current_shares_1
        df_strategy.iloc[i, df_strategy.columns.get_loc('shares_2')] = current_shares_2

    df_strategy['shares_1_change'] = df_strategy['shares_1'].diff().fillna(0)
    df_strategy['shares_2_change'] = df_strategy['shares_2'].diff().fillna(0)

    df_strategy['transaction_costs_dollars'] = 0.0
    df_strategy['commission_1'] = 0.0
    df_strategy['commission_2'] = 0.0
    df_strategy['sec_fee'] = 0.0
    df_strategy['finra_fee'] = 0.0
    df_strategy['spread_cost'] = 0.0

    for i in range(len(df_strategy)):
        total_cost = 0.0
        if df_strategy['shares_1_change'].iloc[i] != 0:
            _, breakdown = calculate_ibkr_transaction_costs(
                symbol1, df[symbol1].iloc[i], df_strategy['shares_1_change'].iloc[i]
            )
            df_strategy.iloc[i, df_strategy.columns.get_loc('commission_1')] = breakdown.get('commission', 0)
            df_strategy.iloc[i, df_strategy.columns.get_loc('sec_fee')] += breakdown.get('sec_fee', 0)
            df_strategy.iloc[i, df_strategy.columns.get_loc('finra_fee')] += breakdown.get('finra_fee', 0)
            df_strategy.iloc[i, df_strategy.columns.get_loc('spread_cost')] += breakdown.get('spread_cost', 0)
            total_cost += breakdown.get('total', 0)

        if df_strategy['shares_2_change'].iloc[i] != 0:
            _, breakdown = calculate_ibkr_transaction_costs(
                symbol2, df[symbol2].iloc[i], df_strategy['shares_2_change'].iloc[i]
            )
            df_strategy.iloc[i, df_strategy.columns.get_loc('commission_2')] = breakdown.get('commission', 0)
            df_strategy.iloc[i, df_strategy.columns.get_loc('sec_fee')] += breakdown.get('sec_fee', 0)
            df_strategy.iloc[i, df_strategy.columns.get_loc('finra_fee')] += breakdown.get('finra_fee', 0)
            df_strategy.iloc[i, df_strategy.columns.get_loc('spread_cost')] += breakdown.get('spread_cost', 0)
            total_cost += breakdown.get('total', 0)

        df_strategy.iloc[i, df_strategy.columns.get_loc('transaction_costs_dollars')] = total_cost

    price_1_prev = df[symbol1].shift(1)
    price_2_prev = df[symbol2].shift(1)
    shares_1_prev = df_strategy['shares_1'].shift(1).fillna(0)
    shares_2_prev = df_strategy['shares_2'].shift(1).fillna(0)

    df_strategy['daily_pnl_1'] = shares_1_prev * price_1_prev * df_strategy['returns_1']
    df_strategy['daily_pnl_2'] = shares_2_prev * price_2_prev * df_strategy['returns_2']
    df_strategy['daily_pnl_gross'] = df_strategy['daily_pnl_1'] + df_strategy['daily_pnl_2']
    df_strategy['strategy_returns_gross'] = df_strategy['daily_pnl_gross'] / INITIAL_CAPITAL
    df_strategy['strategy_returns'] = (
        df_strategy['strategy_returns_gross'] - (df_strategy['transaction_costs_dollars'] / INITIAL_CAPITAL)
    )

    return df_strategy


def extract_trade_log(df, df_strategy, symbol1, symbol2, hedge_ratio):
    """Extract detailed P&L for each individual trade"""
    trades = []
    position_changes = df_strategy['position'].diff().fillna(0)

    in_trade = False
    trade_start_idx = None
    trade_direction = 0

    for i in range(len(df_strategy)):
        current_pos = df_strategy['position'].iloc[i]
        pos_change = position_changes.iloc[i]

        if not in_trade and pos_change != 0 and current_pos != 0:
            in_trade = True
            trade_start_idx = i
            trade_direction = int(current_pos)

        elif in_trade and current_pos == 0 and pos_change != 0:
            trade_end_idx = i

            entry_date = df_strategy.index[trade_start_idx]
            exit_date = df_strategy.index[trade_end_idx]
            holding_days = (exit_date - entry_date).days

            is_stop_loss = df_strategy['stop_loss_exit'].iloc[trade_end_idx] if 'stop_loss_exit' in df_strategy.columns else False

            entry_price_1 = df[symbol1].iloc[trade_start_idx]
            entry_price_2 = df[symbol2].iloc[trade_start_idx]
            exit_price_1 = df[symbol1].iloc[trade_end_idx]
            exit_price_2 = df[symbol2].iloc[trade_end_idx]

            shares_1 = abs(df_strategy['shares_1'].iloc[trade_start_idx])
            shares_2 = abs(df_strategy['shares_2'].iloc[trade_start_idx])

            if trade_direction == 1:
                pnl_1 = shares_1 * (exit_price_1 - entry_price_1)
                pnl_2 = shares_2 * (entry_price_2 - exit_price_2)
            else:
                pnl_1 = shares_1 * (entry_price_1 - exit_price_1)
                pnl_2 = shares_2 * (exit_price_2 - entry_price_2)

            gross_pnl = pnl_1 + pnl_2

            entry_fees = (
                df_strategy['commission_1'].iloc[trade_start_idx] +
                df_strategy['commission_2'].iloc[trade_start_idx] +
                df_strategy['sec_fee'].iloc[trade_start_idx] +
                df_strategy['finra_fee'].iloc[trade_start_idx] +
                df_strategy['spread_cost'].iloc[trade_start_idx]
            )
            exit_fees = (
                df_strategy['commission_1'].iloc[trade_end_idx] +
                df_strategy['commission_2'].iloc[trade_end_idx] +
                df_strategy['sec_fee'].iloc[trade_end_idx] +
                df_strategy['finra_fee'].iloc[trade_end_idx] +
                df_strategy['spread_cost'].iloc[trade_end_idx]
            )
            total_fees = entry_fees + exit_fees
            net_pnl = gross_pnl - total_fees

            entry_zscore = df_strategy['z_score'].iloc[trade_start_idx]
            exit_zscore = df_strategy['z_score'].iloc[trade_end_idx]

            capital_used = shares_1 * entry_price_1 + shares_2 * entry_price_2
            return_pct = (net_pnl / capital_used * 100) if capital_used > 0 else 0

            trades.append({
                'trade_num': len(trades) + 1,
                'direction': 'Long Spread' if trade_direction == 1 else 'Short Spread',
                'entry_date': entry_date.strftime('%Y-%m-%d'),
                'exit_date': exit_date.strftime('%Y-%m-%d'),
                'holding_days': holding_days,
                'entry_zscore': entry_zscore,
                'exit_zscore': exit_zscore,
                'gross_pnl': gross_pnl,
                'total_fees': total_fees,
                'net_pnl': net_pnl,
                'return_pct': return_pct,
                'is_stop_loss': is_stop_loss,
                'is_open': False
            })

            in_trade = False
            trade_start_idx = None
            trade_direction = 0

    # Handle open position at end
    if in_trade and trade_start_idx is not None:
        trade_end_idx = len(df_strategy) - 1

        entry_date = df_strategy.index[trade_start_idx]
        exit_date = df_strategy.index[trade_end_idx]
        holding_days = (exit_date - entry_date).days

        entry_price_1 = df[symbol1].iloc[trade_start_idx]
        entry_price_2 = df[symbol2].iloc[trade_start_idx]
        current_price_1 = df[symbol1].iloc[trade_end_idx]
        current_price_2 = df[symbol2].iloc[trade_end_idx]

        shares_1 = abs(df_strategy['shares_1'].iloc[trade_start_idx])
        shares_2 = abs(df_strategy['shares_2'].iloc[trade_start_idx])

        if trade_direction == 1:
            pnl_1 = shares_1 * (current_price_1 - entry_price_1)
            pnl_2 = shares_2 * (entry_price_2 - current_price_2)
        else:
            pnl_1 = shares_1 * (entry_price_1 - current_price_1)
            pnl_2 = shares_2 * (current_price_2 - entry_price_2)

        gross_pnl = pnl_1 + pnl_2

        entry_fees = (
            df_strategy['commission_1'].iloc[trade_start_idx] +
            df_strategy['commission_2'].iloc[trade_start_idx] +
            df_strategy['sec_fee'].iloc[trade_start_idx] +
            df_strategy['finra_fee'].iloc[trade_start_idx] +
            df_strategy['spread_cost'].iloc[trade_start_idx]
        )
        total_fees = entry_fees
        net_pnl = gross_pnl - total_fees

        entry_zscore = df_strategy['z_score'].iloc[trade_start_idx]
        current_zscore = df_strategy['z_score'].iloc[trade_end_idx]

        capital_used = shares_1 * entry_price_1 + shares_2 * entry_price_2
        return_pct = (net_pnl / capital_used * 100) if capital_used > 0 else 0

        trades.append({
            'trade_num': len(trades) + 1,
            'direction': 'Long Spread' if trade_direction == 1 else 'Short Spread',
            'entry_date': entry_date.strftime('%Y-%m-%d'),
            'exit_date': 'OPEN',
            'holding_days': holding_days,
            'entry_zscore': entry_zscore,
            'exit_zscore': current_zscore,
            'gross_pnl': gross_pnl,
            'total_fees': total_fees,
            'net_pnl': net_pnl,
            'return_pct': return_pct,
            'is_stop_loss': False,
            'is_open': True
        })

    return trades


def calculate_performance_metrics(returns_series):
    """Calculate comprehensive performance metrics"""
    returns_series = returns_series.fillna(0)
    if len(returns_series) == 0 or returns_series.std() == 0:
        return None

    total_return = returns_series.sum()
    num_years = len(returns_series) / 252
    annual_return = total_return / num_years if num_years > 0 else 0
    volatility = returns_series.std() * np.sqrt(252)
    sharpe_ratio = annual_return / volatility if volatility > 0 else 0

    cumulative = returns_series.cumsum()
    peak = cumulative.expanding().max()
    drawdown = cumulative - peak
    max_drawdown = drawdown.min()

    positive_returns = returns_series[returns_series > 0]
    negative_returns = returns_series[returns_series < 0]

    avg_win = positive_returns.mean() if len(positive_returns) > 0 else 0
    avg_loss = abs(negative_returns.mean()) if len(negative_returns) > 0 else 0
    profit_factor = (positive_returns.sum() / abs(negative_returns.sum())) if len(negative_returns) > 0 and negative_returns.sum() != 0 else 0

    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0

    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'num_years': num_years
    }


# =============================================================================
# HTML REPORT GENERATION
# =============================================================================

def generate_html_report(symbol1, symbol2, start_date, end_date, lookback_period,
                         std_threshold, stop_loss_threshold, df, df_strategy,
                         pair_analysis, metrics, trade_log, final_value, total_fees):
    """Generate a comprehensive HTML report for download"""

    # Calculate additional metrics
    gross_return = df_strategy['strategy_returns_gross'].fillna(0).sum()
    winning_trades = len([t for t in trade_log if t['net_pnl'] > 0 and not t['is_open']])
    total_completed = len([t for t in trade_log if not t['is_open']])
    trade_win_rate = winning_trades / total_completed if total_completed > 0 else 0

    # Trade log HTML
    trade_rows = ""
    for t in trade_log:
        pnl_color = '#4CAF50' if t['net_pnl'] > 0 else '#F44336'
        direction_icon = '&#x1F4C8;' if t['direction'] == 'Long Spread' else '&#x1F4C9;'
        stop_marker = ' &#x1F6D1;' if t.get('is_stop_loss', False) else ''
        open_marker = ' &#x23F3;' if t.get('is_open', False) else ''
        exit_display = f"<span style='color: #FF9800;'>{t['exit_date']}</span>" if t.get('is_open') else t['exit_date']

        trade_rows += f"""
        <tr>
            <td>{t['trade_num']}</td>
            <td>{direction_icon} {t['direction']}{stop_marker}{open_marker}</td>
            <td>{t['entry_date']}</td>
            <td>{exit_display}</td>
            <td>{t['holding_days']}</td>
            <td>{t['entry_zscore']:.2f}</td>
            <td>{t['exit_zscore']:.2f}</td>
            <td style="color: {'#4CAF50' if t['gross_pnl'] > 0 else '#F44336'};">${t['gross_pnl']:,.2f}</td>
            <td style="color: #F44336;">-${t['total_fees']:,.2f}</td>
            <td style="color: {pnl_color}; font-weight: bold;">${t['net_pnl']:,.2f}</td>
            <td style="color: {pnl_color};">{t['return_pct']:.2f}%</td>
        </tr>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pairs Trading Backtest Report: {symbol1} vs {symbol2}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; background: #f5f5f5; padding: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .header {{ background: linear-gradient(135deg, #1a237e 0%, #3949ab 100%); color: white; padding: 30px; text-align: center; }}
        .header h1 {{ font-size: 2.5em; margin-bottom: 10px; }}
        .section {{ padding: 30px; border-bottom: 1px solid #eee; }}
        .section-title {{ font-size: 1.5em; color: #1a237e; margin-bottom: 20px; padding-bottom: 10px; border-bottom: 2px solid #3949ab; }}
        .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 20px; }}
        .summary-card {{ background: #f8f9fa; border-radius: 8px; padding: 20px; text-align: center; border-left: 4px solid #3949ab; }}
        .summary-card.positive {{ border-left-color: #4CAF50; }}
        .summary-card.negative {{ border-left-color: #F44336; }}
        .summary-card .label {{ font-size: 0.9em; color: #666; margin-bottom: 5px; }}
        .summary-card .value {{ font-size: 1.8em; font-weight: bold; color: #1a237e; }}
        .summary-card .value.positive {{ color: #4CAF50; }}
        .summary-card .value.negative {{ color: #F44336; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #f8f9fa; font-weight: 600; color: #1a237e; }}
        tr:hover {{ background: #f8f9fa; }}
        .info-box {{ background: #E3F2FD; border-left: 4px solid #2196F3; padding: 15px 20px; margin: 20px 0; border-radius: 0 8px 8px 0; }}
        .warning-box {{ background: #FFF3E0; border-left: 4px solid #FF9800; padding: 15px 20px; margin: 20px 0; border-radius: 0 8px 8px 0; }}
        .success-box {{ background: #E8F5E9; border-left: 4px solid #4CAF50; padding: 15px 20px; margin: 20px 0; border-radius: 0 8px 8px 0; }}
        .two-column {{ display: grid; grid-template-columns: 1fr 1fr; gap: 30px; }}
        @media (max-width: 768px) {{ .two-column {{ grid-template-columns: 1fr; }} }}
        @media print {{ body {{ background: white; padding: 0; }} .container {{ box-shadow: none; }} }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Pairs Trading Backtest Report</h1>
            <div style="font-size: 1.2em; opacity: 0.9;">{symbol1} vs {symbol2}</div>
            <div style="margin-top: 15px; font-size: 0.9em; opacity: 0.8;">
                Backtest Period: {start_date} to {end_date}<br>
                Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </div>
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
                    <div class="label">Trade Win Rate</div>
                    <div class="value">{trade_win_rate:.1%} ({winning_trades}/{total_completed})</div>
                </div>
                <div class="summary-card">
                    <div class="label">Trading Days</div>
                    <div class="value">{len(df):,}</div>
                </div>
            </div>
            {'<div class="success-box"><strong>Good News!</strong> This pair shows statistical properties suitable for pairs trading.</div>' if pair_analysis['cointegrated'] and pair_analysis['stationary'] else '<div class="warning-box"><strong>Caution:</strong> This pair may not be ideal for pairs trading. Consider testing other pairs.</div>'}
        </div>

        <div class="section">
            <h2 class="section-title">Strategy Parameters</h2>
            <table>
                <tr><th>Parameter</th><th>Value</th></tr>
                <tr><td>Lookback Period</td><td>{lookback_period} days</td></tr>
                <tr><td>Entry Threshold</td><td>{std_threshold} standard deviations</td></tr>
                <tr><td>Stop-Loss</td><td>{'Disabled' if stop_loss_threshold == 0 else f'{stop_loss_threshold} standard deviations'}</td></tr>
                <tr><td>Position Size per Leg</td><td>${POSITION_SIZE_PER_LEG:,}</td></tr>
            </table>
        </div>

        <div class="section">
            <h2 class="section-title">Pair Statistical Analysis</h2>
            <table>
                <tr><th>Metric</th><th>Value</th><th>Interpretation</th></tr>
                <tr><td>Correlation</td><td>{pair_analysis['correlation']:.4f}</td><td>{'Strong' if abs(pair_analysis['correlation']) > 0.7 else 'Moderate' if abs(pair_analysis['correlation']) > 0.5 else 'Weak'} correlation</td></tr>
                <tr><td>Cointegration p-value</td><td>{pair_analysis['cointegration_pvalue']:.4f}</td><td>{'Cointegrated (Good)' if pair_analysis['cointegrated'] else 'Not Cointegrated (Caution)'}</td></tr>
                <tr><td>Hedge Ratio</td><td>{pair_analysis['hedge_ratio']:.4f}</td><td>For every 1 unit of {symbol1}, trade {abs(pair_analysis['hedge_ratio']):.2f} units of {symbol2}</td></tr>
                <tr><td>R-Squared</td><td>{pair_analysis['r_squared']:.4f}</td><td>{pair_analysis['r_squared']*100:.1f}% of variance explained</td></tr>
                <tr><td>Spread Stationarity (ADF p-value)</td><td>{pair_analysis['adf_pvalue']:.4f}</td><td>{'Stationary (Mean-Reverting)' if pair_analysis['stationary'] else 'Non-Stationary (Caution)'}</td></tr>
            </table>
        </div>

        <div class="section">
            <h2 class="section-title">Performance Metrics</h2>
            <div class="two-column">
                <div>
                    <h3 style="margin-bottom: 15px; color: #1a237e;">Return Metrics</h3>
                    <table>
                        <tr><td>Total Return (Net)</td><td><strong>{metrics['total_return']:.2%}</strong></td></tr>
                        <tr><td>Gross Return (Before Fees)</td><td>{gross_return:.2%}</td></tr>
                        <tr><td>Annual Return</td><td>{metrics['annual_return']:.2%}</td></tr>
                        <tr><td>Volatility (Annual)</td><td>{metrics['volatility']:.2%}</td></tr>
                        <tr><td>Sharpe Ratio</td><td><strong>{metrics['sharpe_ratio']:.2f}</strong></td></tr>
                        <tr><td>Calmar Ratio</td><td>{metrics['calmar_ratio']:.2f}</td></tr>
                    </table>
                </div>
                <div>
                    <h3 style="margin-bottom: 15px; color: #1a237e;">Risk Metrics</h3>
                    <table>
                        <tr><td>Maximum Drawdown</td><td><strong style="color: #F44336;">{metrics['max_drawdown']:.2%}</strong></td></tr>
                        <tr><td>Trade Win Rate</td><td>{trade_win_rate:.1%} ({winning_trades}/{total_completed})</td></tr>
                        <tr><td>Profit Factor</td><td>{metrics['profit_factor']:.2f}</td></tr>
                        <tr><td>Average Win</td><td style="color: #4CAF50;">{metrics['avg_win']*100:.3f}%</td></tr>
                        <tr><td>Average Loss</td><td style="color: #F44336;">-{metrics['avg_loss']*100:.3f}%</td></tr>
                        <tr><td>Fee Drag</td><td style="color: #F44336;">{(gross_return - metrics['total_return']):.2%}</td></tr>
                    </table>
                </div>
            </div>
        </div>

        <div class="section">
            <h2 class="section-title">IBKR Fee Breakdown</h2>
            <table>
                <tr><th>Fee Type</th><th>Amount</th></tr>
                <tr><td>Commissions</td><td>${df_strategy['commission_1'].sum() + df_strategy['commission_2'].sum():,.2f}</td></tr>
                <tr><td>SEC Fee</td><td>${df_strategy['sec_fee'].sum():,.2f}</td></tr>
                <tr><td>FINRA TAF</td><td>${df_strategy['finra_fee'].sum():,.2f}</td></tr>
                <tr><td>Bid-Ask Spread</td><td>${df_strategy['spread_cost'].sum():,.2f}</td></tr>
                <tr style="font-weight: bold; background: #f8f9fa;"><td>Total Fees</td><td style="color: #F44336;">${total_fees:,.2f}</td></tr>
            </table>
        </div>

        <div class="section">
            <h2 class="section-title">Trade Log</h2>
            <div style="overflow-x: auto;">
                <table style="font-size: 0.9em;">
                    <thead>
                        <tr style="background: #1a237e; color: white;">
                            <th>#</th><th>Direction</th><th>Entry</th><th>Exit</th><th>Days</th>
                            <th>Entry Z</th><th>Exit Z</th><th>Gross P&L</th><th>Fees</th><th>Net P&L</th><th>Return</th>
                        </tr>
                    </thead>
                    <tbody>
                        {trade_rows}
                    </tbody>
                </table>
            </div>
            <div class="info-box" style="margin-top: 20px;">
                <strong>Legend:</strong><br>
                &#x1F4C8; Long Spread: Bought {symbol1}, Sold {symbol2}<br>
                &#x1F4C9; Short Spread: Sold {symbol1}, Bought {symbol2}<br>
                &#x1F6D1; Stop-Loss Exit | &#x23F3; Open Position (unrealized P&L)
            </div>
        </div>

        <div style="background: #f8f9fa; padding: 20px 30px; text-align: center; color: #666; font-size: 0.9em;">
            Generated by Pairs Trading Backtester | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
</body>
</html>"""

    return html


# =============================================================================
# STREAMLIT UI
# =============================================================================

def main():
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a237e;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border-left: 4px solid #1a237e;
    }
    .positive {
        color: #2e7d32;
    }
    .negative {
        color: #c62828;
    }
    .info-box {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown('<h1 class="main-header">Pairs Trading Backtester</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Test statistical arbitrage strategies on any two assets</p>', unsafe_allow_html=True)

    # Sidebar for inputs
    with st.sidebar:
        st.header("Strategy Settings")

        st.subheader("Select Your Assets")

        col1, col2 = st.columns(2)
        with col1:
            symbol1 = st.text_input(
                "First Asset",
                value="GLD",
                help="Enter any Yahoo Finance ticker (e.g., AAPL, MSFT, GLD)"
            ).upper().strip()
        with col2:
            symbol2 = st.text_input(
                "Second Asset",
                value="SLV",
                help="Enter a related asset to trade against"
            ).upper().strip()

        st.markdown("---")
        st.subheader("Backtest Period")

        today = datetime.now().date()
        default_start = today - timedelta(days=365*2)

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=default_start,
                help="When to start the backtest"
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=today,
                help="When to end the backtest"
            )

        st.markdown("---")
        st.subheader("Strategy Parameters")

        lookback_period = st.slider(
            "Lookback Period (days)",
            min_value=20,
            max_value=200,
            value=90,
            step=10,
            help="""
            How many days to use for calculating the 'normal' spread.
            - 30-60: More trades, higher risk
            - 90: Balanced (recommended)
            - 120+: Fewer trades, more stable
            """
        )

        std_threshold = st.slider(
            "Entry Threshold (standard deviations)",
            min_value=1.0,
            max_value=3.0,
            value=1.5,
            step=0.1,
            help="""
            How far the spread must deviate to trigger a trade.
            - 1.0-1.5: More aggressive, more trades
            - 2.0: Conservative, fewer but stronger signals
            """
        )

        stop_loss_enabled = st.checkbox("Enable Stop-Loss", value=True, help="Exit losing trades to limit losses")

        if stop_loss_enabled:
            stop_loss_threshold = st.slider(
                "Stop-Loss Threshold",
                min_value=2.0,
                max_value=5.0,
                value=3.0,
                step=0.5,
                help="Exit if z-score exceeds this level against your position"
            )
        else:
            stop_loss_threshold = 0

        st.markdown("---")

        run_backtest = st.button("Run Backtest", type="primary", use_container_width=True)

    # Main content area
    if run_backtest:
        if not symbol1 or not symbol2:
            st.error("Please enter both ticker symbols.")
            return

        if symbol1 == symbol2:
            st.error("Please enter two different tickers.")
            return

        if start_date >= end_date:
            st.error("End date must be after start date.")
            return

        # Progress indicator
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Step 1: Download data
        status_text.text("Downloading market data...")
        progress_bar.progress(20)

        df, error = download_pair_data(symbol1, symbol2, str(start_date), str(end_date))

        if error:
            st.error(error)
            progress_bar.empty()
            status_text.empty()
            return

        # Step 2: Analyze pair
        status_text.text("Analyzing pair relationship...")
        progress_bar.progress(40)

        pair_analysis = analyze_pair_quality(df, symbol1, symbol2)

        # Step 3: Run backtest
        status_text.text("Running backtest...")
        progress_bar.progress(60)

        df_strategy = calculate_strategy_returns(
            df, symbol1, symbol2, pair_analysis,
            lookback_period, std_threshold, stop_loss_threshold
        )

        # Step 4: Extract trades
        status_text.text("Analyzing trades...")
        progress_bar.progress(80)

        trade_log = extract_trade_log(df, df_strategy, symbol1, symbol2, pair_analysis['hedge_ratio'])
        metrics = calculate_performance_metrics(df_strategy['strategy_returns'])

        progress_bar.progress(100)
        status_text.empty()
        progress_bar.empty()

        if metrics is None:
            st.error("Could not calculate metrics. The strategy may not have generated any trades.")
            return

        # Calculate final values
        final_value = INITIAL_CAPITAL * (1 + df_strategy['strategy_returns'].sum())
        total_fees = df_strategy['transaction_costs_dollars'].sum()

        # Results Section
        st.markdown("---")
        st.header("Backtest Results")

        # Summary Cards
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_ret = metrics['total_return']
            color = "positive" if total_ret > 0 else "negative"
            st.metric(
                "Total Return",
                f"{total_ret:.1%}",
                delta=f"${final_value - INITIAL_CAPITAL:,.0f}"
            )

        with col2:
            st.metric(
                "Sharpe Ratio",
                f"{metrics['sharpe_ratio']:.2f}",
                delta="Good" if metrics['sharpe_ratio'] > 1 else "Low"
            )

        with col3:
            st.metric(
                "Max Drawdown",
                f"{metrics['max_drawdown']:.1%}",
            )

        with col4:
            st.metric(
                "Final Value",
                f"${final_value:,.0f}",
                delta=f"from ${INITIAL_CAPITAL:,}"
            )

        # Pair Quality Section
        st.markdown("---")
        st.subheader("Pair Quality Analysis")

        is_suitable = pair_analysis['cointegrated'] and pair_analysis['stationary']

        if is_suitable:
            st.success("This pair shows good statistical properties for pairs trading.")
        else:
            st.warning("This pair may not be ideal for pairs trading. Consider testing other pairs.")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Correlation", f"{pair_analysis['correlation']:.3f}")
        with col2:
            coint_status = "Yes" if pair_analysis['cointegrated'] else "No"
            st.metric("Cointegrated", coint_status)
        with col3:
            st.metric("Hedge Ratio", f"{pair_analysis['hedge_ratio']:.3f}")
        with col4:
            stat_status = "Yes" if pair_analysis['stationary'] else "No"
            st.metric("Mean-Reverting", stat_status)

        # Charts Section
        st.markdown("---")
        st.subheader("Performance Charts")

        # Portfolio Performance Chart
        fig, ax = plt.subplots(figsize=(12, 5))
        portfolio_value = INITIAL_CAPITAL * (1 + df_strategy['strategy_returns'].cumsum())
        ax.plot(df_strategy.index, portfolio_value, label='Pairs Strategy', linewidth=2, color='#1a237e')
        ax.axhline(y=INITIAL_CAPITAL, color='gray', linestyle=':', alpha=0.7, label='Initial Capital')
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value ($)')
        ax.set_title('Portfolio Performance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        st.pyplot(fig)
        plt.close()

        # Z-Score Chart
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df_strategy.index, df_strategy['z_score'], alpha=0.8, linewidth=1, color='#1a237e')
        ax.axhline(y=std_threshold, color='#c62828', linestyle='--', linewidth=1.5, label=f'Short Entry (+{std_threshold})')
        ax.axhline(y=-std_threshold, color='#2e7d32', linestyle='--', linewidth=1.5, label=f'Long Entry (-{std_threshold})')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1, label='Mean (Exit)')
        if stop_loss_threshold > 0:
            ax.axhline(y=stop_loss_threshold, color='#9c27b0', linestyle=':', linewidth=2, label=f'Stop-Loss (+{stop_loss_threshold})')
            ax.axhline(y=-stop_loss_threshold, color='#9c27b0', linestyle=':', linewidth=2, label=f'Stop-Loss (-{stop_loss_threshold})')
        ax.fill_between(df_strategy.index, -std_threshold, std_threshold, alpha=0.1, color='gray')
        ax.set_xlabel('Date')
        ax.set_ylabel('Z-Score')
        ax.set_title('Z-Score with Entry/Exit Levels')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        y_limit = max(4, (stop_loss_threshold + 1) if stop_loss_threshold > 0 else 4)
        ax.set_ylim(-y_limit, y_limit)
        st.pyplot(fig)
        plt.close()

        # Trade Log Section
        st.markdown("---")
        st.subheader("Trade Log")

        if trade_log:
            winning_trades = len([t for t in trade_log if t['net_pnl'] > 0 and not t['is_open']])
            losing_trades = len([t for t in trade_log if t['net_pnl'] <= 0 and not t['is_open']])
            total_trades = len([t for t in trade_log if not t['is_open']])

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Trades", len(trade_log))
            with col2:
                win_rate = winning_trades / total_trades if total_trades > 0 else 0
                st.metric("Win Rate", f"{win_rate:.1%}")
            with col3:
                total_net = sum(t['net_pnl'] for t in trade_log)
                st.metric("Net P&L", f"${total_net:,.2f}")
            with col4:
                st.metric("Total Fees", f"${total_fees:,.2f}")

            # Convert to DataFrame for display
            trade_df = pd.DataFrame(trade_log)
            trade_df = trade_df[['trade_num', 'direction', 'entry_date', 'exit_date', 'holding_days',
                                  'entry_zscore', 'exit_zscore', 'gross_pnl', 'total_fees', 'net_pnl', 'return_pct']]
            trade_df.columns = ['#', 'Direction', 'Entry', 'Exit', 'Days', 'Entry Z', 'Exit Z',
                                'Gross P&L', 'Fees', 'Net P&L', 'Return %']

            # Format columns
            trade_df['Gross P&L'] = trade_df['Gross P&L'].apply(lambda x: f"${x:,.2f}")
            trade_df['Fees'] = trade_df['Fees'].apply(lambda x: f"${x:,.2f}")
            trade_df['Net P&L'] = trade_df['Net P&L'].apply(lambda x: f"${x:,.2f}")
            trade_df['Entry Z'] = trade_df['Entry Z'].apply(lambda x: f"{x:.2f}")
            trade_df['Exit Z'] = trade_df['Exit Z'].apply(lambda x: f"{x:.2f}")
            trade_df['Return %'] = trade_df['Return %'].apply(lambda x: f"{x:.2f}%")

            st.dataframe(trade_df, use_container_width=True, hide_index=True)
        else:
            st.info("No trades were executed during this period.")

        # Fee Breakdown
        st.markdown("---")
        st.subheader("IBKR Fee Breakdown")

        total_commissions = df_strategy['commission_1'].sum() + df_strategy['commission_2'].sum()
        total_sec = df_strategy['sec_fee'].sum()
        total_finra = df_strategy['finra_fee'].sum()
        total_spread = df_strategy['spread_cost'].sum()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Commissions", f"${total_commissions:,.2f}")
        with col2:
            st.metric("SEC Fee", f"${total_sec:,.2f}")
        with col3:
            st.metric("FINRA TAF", f"${total_finra:,.2f}")
        with col4:
            st.metric("Bid-Ask Spread", f"${total_spread:,.2f}")

        # Detailed Performance Metrics
        st.markdown("---")
        st.subheader("Detailed Performance Metrics")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Return Metrics**")
            gross_return = df_strategy['strategy_returns_gross'].fillna(0).sum()
            return_data = {
                "Metric": ["Total Return (Net)", "Gross Return (Before Fees)", "Annual Return", "Volatility (Annual)", "Sharpe Ratio", "Calmar Ratio"],
                "Value": [
                    f"{metrics['total_return']:.2%}",
                    f"{gross_return:.2%}",
                    f"{metrics['annual_return']:.2%}",
                    f"{metrics['volatility']:.2%}",
                    f"{metrics['sharpe_ratio']:.2f}",
                    f"{metrics['calmar_ratio']:.2f}"
                ]
            }
            st.dataframe(pd.DataFrame(return_data), use_container_width=True, hide_index=True)

        with col2:
            st.markdown("**Risk Metrics**")
            winning_trades = len([t for t in trade_log if t['net_pnl'] > 0 and not t['is_open']])
            total_completed = len([t for t in trade_log if not t['is_open']])
            win_rate = winning_trades / total_completed if total_completed > 0 else 0
            risk_data = {
                "Metric": ["Maximum Drawdown", "Trade Win Rate", "Profit Factor", "Average Win", "Average Loss", "Fee Drag on Returns"],
                "Value": [
                    f"{metrics['max_drawdown']:.2%}",
                    f"{win_rate:.1%} ({winning_trades}/{total_completed})",
                    f"{metrics['profit_factor']:.2f}",
                    f"{metrics['avg_win']*100:.3f}%",
                    f"-{metrics['avg_loss']*100:.3f}%",
                    f"{(gross_return - metrics['total_return']):.2%}"
                ]
            }
            st.dataframe(pd.DataFrame(risk_data), use_container_width=True, hide_index=True)

        # Detailed Pair Statistics
        st.markdown("---")
        st.subheader("Detailed Pair Statistics")

        pair_stats = {
            "Metric": ["Correlation", "Cointegration p-value", "Cointegrated?", "Hedge Ratio", "R-Squared", "Spread Stationarity (ADF p-value)", "Stationary?"],
            "Value": [
                f"{pair_analysis['correlation']:.4f}",
                f"{pair_analysis['cointegration_pvalue']:.4f}",
                "Yes (Good)" if pair_analysis['cointegrated'] else "No (Caution)",
                f"{pair_analysis['hedge_ratio']:.4f}",
                f"{pair_analysis['r_squared']:.4f}",
                f"{pair_analysis['adf_pvalue']:.4f}",
                "Yes (Mean-Reverting)" if pair_analysis['stationary'] else "No (Caution)"
            ],
            "Interpretation": [
                "Strong" if abs(pair_analysis['correlation']) > 0.7 else "Moderate" if abs(pair_analysis['correlation']) > 0.5 else "Weak",
                "Cointegrated" if pair_analysis['cointegrated'] else "Not Cointegrated",
                "Suitable for pairs trading" if pair_analysis['cointegrated'] else "May not be suitable",
                f"For every 1 unit of {symbol1}, trade {abs(pair_analysis['hedge_ratio']):.2f} units of {symbol2}",
                f"{pair_analysis['r_squared']*100:.1f}% of variance explained",
                "Stationary" if pair_analysis['stationary'] else "Non-Stationary",
                "Spread tends to revert to mean" if pair_analysis['stationary'] else "Spread may drift"
            ]
        }
        st.dataframe(pd.DataFrame(pair_stats), use_container_width=True, hide_index=True)

        # Additional Charts
        st.markdown("---")
        st.subheader("Additional Charts")

        # Spread with Bollinger Bands
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df_strategy.index, df_strategy['spread'], alpha=0.8, label='Spread', linewidth=1.5, color='#2196F3')
        ax.plot(df_strategy.index, df_strategy['ma'], color='black', label='Moving Average', linewidth=2)
        ax.plot(df_strategy.index, df_strategy['upper_band'], color='#F44336', linestyle='--', label=f'Upper Band (+{std_threshold}σ)', linewidth=1.5)
        ax.plot(df_strategy.index, df_strategy['lower_band'], color='#4CAF50', linestyle='--', label=f'Lower Band (-{std_threshold}σ)', linewidth=1.5)
        ax.fill_between(df_strategy.index, df_strategy['lower_band'], df_strategy['upper_band'], alpha=0.1, color='gray')
        ax.set_xlabel('Date')
        ax.set_ylabel('Spread Value')
        ax.set_title('Spread with Bollinger Bands')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()

        # Drawdown Chart
        fig, ax = plt.subplots(figsize=(12, 4))
        cumulative_return = df_strategy['strategy_returns'].cumsum()
        peak = cumulative_return.expanding().max()
        drawdown = (cumulative_return - peak) * 100
        ax.fill_between(df_strategy.index, drawdown, 0, alpha=0.5, color='#F44336')
        ax.plot(df_strategy.index, drawdown, linewidth=1, color='#B71C1C')
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')
        ax.set_title('Strategy Drawdown')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()

        # Position Over Time
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.fill_between(df_strategy.index, df_strategy['position'], 0,
                        where=(df_strategy['position'] > 0), alpha=0.5, color='#4CAF50', label='Long')
        ax.fill_between(df_strategy.index, df_strategy['position'], 0,
                        where=(df_strategy['position'] < 0), alpha=0.5, color='#F44336', label='Short')
        ax.plot(df_strategy.index, df_strategy['position'], linewidth=0.5, color='black')
        ax.set_xlabel('Date')
        ax.set_ylabel('Position')
        ax.set_title('Position Over Time')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1.5, 1.5)
        st.pyplot(fig)
        plt.close()

        # Download Report Section
        st.markdown("---")
        st.subheader("Download Report")

        # Generate HTML Report
        html_report = generate_html_report(
            symbol1, symbol2, str(start_date), str(end_date),
            lookback_period, std_threshold, stop_loss_threshold,
            df, df_strategy, pair_analysis, metrics, trade_log,
            final_value, total_fees
        )

        report_filename = f"Backtest_Report_{symbol1}_{symbol2}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"

        st.download_button(
            label="Download Full HTML Report",
            data=html_report,
            file_name=report_filename,
            mime="text/html",
            use_container_width=True
        )

        st.info("The HTML report contains all charts and detailed analysis. You can open it in any web browser and save as PDF.")

        # Strategy Explanation
        st.markdown("---")
        st.subheader("How This Strategy Works")

        st.markdown("""
        <div class="info-box">
        <strong>Pairs Trading Explained:</strong><br><br>

        1. <strong>Find Related Assets:</strong> Two assets that historically move together (like GLD and SLV - both precious metals)<br><br>

        2. <strong>Watch the Spread:</strong> The difference in their prices usually stays within a range<br><br>

        3. <strong>Trade Mean Reversion:</strong><br>
           - When spread goes TOO LOW: Buy Asset 1, Sell Asset 2 (Long Spread)<br>
           - When spread goes TOO HIGH: Sell Asset 1, Buy Asset 2 (Short Spread)<br>
           - When spread returns to normal: Close the trade<br><br>

        4. <strong>The Z-Score</strong> measures how far the spread is from "normal" in standard deviations
        </div>
        """, unsafe_allow_html=True)

    else:
        # Show welcome message when no backtest is running
        st.markdown("""
        <div class="info-box">
        <strong>Welcome!</strong> This tool helps you test pairs trading strategies.<br><br>

        <strong>Quick Start:</strong><br>
        1. Enter two related assets in the sidebar (e.g., GLD and SLV)<br>
        2. Set your date range<br>
        3. Adjust strategy parameters (or use defaults)<br>
        4. Click "Run Backtest"<br><br>

        <strong>Popular Pairs to Try:</strong><br>
        - GLD/SLV (Gold vs Silver)<br>
        - XOM/CVX (Exxon vs Chevron)<br>
        - KO/PEP (Coca-Cola vs Pepsi)<br>
        - JPM/BAC (JPMorgan vs Bank of America)
        </div>
        """, unsafe_allow_html=True)

        # Show example pairs
        st.subheader("Example Asset Pairs")

        example_pairs = [
            {"Pair": "GLD / SLV", "Sector": "Precious Metals", "Description": "Gold and Silver ETFs"},
            {"Pair": "XOM / CVX", "Sector": "Energy", "Description": "Major oil companies"},
            {"Pair": "KO / PEP", "Sector": "Consumer Staples", "Description": "Beverage giants"},
            {"Pair": "JPM / BAC", "Sector": "Financials", "Description": "Major US banks"},
            {"Pair": "MSFT / AAPL", "Sector": "Technology", "Description": "Tech giants"},
            {"Pair": "HD / LOW", "Sector": "Retail", "Description": "Home improvement stores"},
        ]

        st.dataframe(pd.DataFrame(example_pairs), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
