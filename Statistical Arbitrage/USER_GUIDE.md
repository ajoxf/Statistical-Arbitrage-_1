# Statistical Arbitrage - Pairs Trading Backtester

**Created by Ankit Jhaveri**

## Table of Contents
1. [Introduction](#introduction)
2. [Web App (Easiest Option)](#web-app-easiest-option)
3. [What is Pairs Trading?](#what-is-pairs-trading)
4. [Requirements](#requirements)
5. [Installation](#installation)
6. [How to Run the Backtester](#how-to-run-the-backtester)
7. [Understanding the Report](#understanding-the-report)
8. [Choosing Good Pairs](#choosing-good-pairs)
9. [Troubleshooting](#troubleshooting)
10. [Files in This Repository](#files-in-this-repository)
11. [Glossary](#glossary)

---

## Introduction

This tool allows you to test a **pairs trading strategy** on any two stocks or ETFs available on Yahoo Finance. It generates a professional HTML report with charts and performance metrics that you can save as a PDF.

**No coding required!** Just use the web app or run the program locally.

---

## Web App (Easiest Option)

**Don't want to install anything?** Use the online web app:

### [Launch Pairs Trading Backtester](https://statarb-pairstrading.streamlit.app)

The web app offers:
- **No installation required** - runs in your browser
- **Same features** as the command-line version
- **Interactive interface** - just fill in the form and click "Run Backtest"
- **Download reports** - save detailed HTML reports to your computer

### How to Use the Web App

1. Go to the web app link above
2. Enter two ticker symbols (e.g., GLD and SLV)
3. Set your date range
4. Adjust strategy parameters (or use defaults)
5. Click **"Run Backtest"**
6. View results and download the HTML report

---

## What is Pairs Trading?

Pairs trading is a market-neutral strategy that involves:

1. **Finding two related assets** (e.g., Coca-Cola and Pepsi)
2. **Monitoring their price relationship** (the "spread")
3. **Trading when they diverge** from their normal relationship
4. **Profiting when they converge** back to normal

### Simple Example

Imagine Coca-Cola (KO) and Pepsi (PEP) usually trade at similar prices. If KO suddenly drops while PEP stays the same:
- **Buy** KO (it's "cheap")
- **Sell** PEP (it's relatively "expensive")
- **Wait** for them to return to their normal relationship
- **Close both positions** and collect the profit

---

## Requirements

### Software Needed

| Software | Purpose | How to Get It |
|----------|---------|---------------|
| **Python 3.8+** | Runs the program | [python.org/downloads](https://www.python.org/downloads/) |
| **Git** (optional) | Upload reports to GitHub | [git-scm.com](https://git-scm.com/) |

### Python Packages

The program needs these packages (install once):

```bash
pip install pandas numpy yfinance matplotlib seaborn statsmodels
```

Or on Windows:
```cmd
py -m pip install pandas numpy yfinance matplotlib seaborn statsmodels
```

---

## Installation

### Step 1: Download the Repository

**Option A: Using Git (Recommended)**
```bash
git clone https://github.com/YOUR_USERNAME/Statistical-Arbitrage-_1.git
cd Statistical-Arbitrage-_1
```

**Option B: Download ZIP**
1. Go to the GitHub repository
2. Click the green "Code" button
3. Select "Download ZIP"
4. Extract the ZIP file

### Step 2: Install Python Packages

Open a terminal/command prompt and run:

```bash
pip install pandas numpy yfinance matplotlib seaborn statsmodels
```

### Step 3: Verify Installation

```bash
cd "Statistical Arbitrage"
python Stat_Arb_User_Backtest.py
```

If you see the welcome message, you're ready!

---

## How to Run the Backtester

### Step 1: Open Terminal/Command Prompt

**Windows:**
- Press `Win + R`
- Type `cmd` and press Enter

**Mac:**
- Press `Cmd + Space`
- Type `Terminal` and press Enter

### Step 2: Navigate to the Folder

```bash
cd "path/to/Statistical-Arbitrage-_1/Statistical Arbitrage"
```

### Step 3: Run the Program

```bash
python Stat_Arb_User_Backtest.py
```

Or on Windows:
```cmd
py Stat_Arb_User_Backtest.py
```

### Step 4: Enter Your Inputs

The program will ask you for:

```
============================================================
   STATISTICAL ARBITRAGE PAIRS TRADING BACKTESTER
============================================================

Enter FIRST ticker symbol: AAPL
Enter SECOND ticker symbol: MSFT

Start date (YYYY-MM-DD): 2020-01-01
End date (YYYY-MM-DD): 2024-12-31

Proceed with backtest? (y/n): y
```

### Step 5: View Results

- The HTML report opens automatically in your browser
- Reports are saved in the `reports/` folder
- You can optionally upload to GitHub when prompted

### Step 6: Save as PDF (Optional)

1. With the report open in your browser
2. Press `Ctrl + P` (Windows) or `Cmd + P` (Mac)
3. Select "Save as PDF" as the destination
4. Click Save

---

## Understanding the Report

### Executive Summary

| Metric | What It Means | Good Value |
|--------|---------------|------------|
| **Total Return** | Overall profit/loss | Positive is good |
| **Annual Return** | Yearly average return | > 10% is good |
| **Sharpe Ratio** | Risk-adjusted return | > 1.0 is good, > 2.0 is excellent |
| **Max Drawdown** | Largest peak-to-trough loss | < -20% is concerning |

### Pair Quality Analysis

| Metric | What It Means | Ideal Value |
|--------|---------------|-------------|
| **Correlation** | How closely prices move together | > 0.7 |
| **Cointegration p-value** | Statistical test for mean reversion | < 0.05 |
| **Quality Score** | Overall pair rating | > 60/100 |

### Charts Explained

1. **Price Comparison** - Shows both assets' prices over time
2. **Spread with Bands** - The price difference with entry/exit zones
3. **Portfolio Performance** - Your strategy vs. buy-and-hold
4. **Z-Score** - How far the spread is from "normal"
5. **Position Over Time** - When you're long, short, or flat
6. **Drawdown** - Visualization of losses from peak
7. **Returns Distribution** - Histogram of daily returns

---

## Choosing Good Pairs

### Best Types of Pairs

| Category | Examples | Why They Work |
|----------|----------|---------------|
| **Same Industry** | KO & PEP, HD & LOW | Similar business drivers |
| **ETF Variants** | SPY & IVV, GLD & IAU | Track the same index |
| **Sector ETFs** | XLF & VFH | Same sector exposure |
| **Related Commodities** | GLD & GDX | Gold price drives both |

### Popular Pairs to Test

```
Classic Pairs:
- KO vs PEP (Coca-Cola vs Pepsi)
- HD vs LOW (Home Depot vs Lowe's)
- V vs MA (Visa vs Mastercard)
- XOM vs CVX (Exxon vs Chevron)

ETF Pairs:
- SPY vs IVV (S&P 500 ETFs)
- GLD vs IAU (Gold ETFs)
- QQQ vs TQQQ (NASDAQ, different leverage)

Sector Pairs:
- JPM vs BAC (Major banks)
- AAPL vs MSFT (Big tech)
- UPS vs FDX (Shipping)
```

### What Makes a Good Pair?

1. **High Correlation (> 0.7)** - Prices move together
2. **Cointegrated (p-value < 0.05)** - Spread is mean-reverting
3. **Same Industry/Sector** - Affected by same market forces
4. **Similar Market Cap** - Comparable size companies
5. **Liquid (High Volume)** - Easy to trade

---

## Troubleshooting

### Common Errors and Solutions

#### "No module named 'yfinance'"
```bash
pip install yfinance
```

#### "'pip' is not recognized"
Use this instead:
```bash
py -m pip install yfinance
```

#### "No data found for TICKER"
- Check the ticker symbol is correct
- Verify it exists on [Yahoo Finance](https://finance.yahoo.com)
- The ticker might be delisted or too new

#### "Insufficient data"
- Your date range is too short
- Try extending the date range (at least 1 year recommended)

#### Report won't open automatically
- Find the report in the `reports/` folder
- Open the `.html` file manually with any web browser

#### Git push fails
- Make sure you have Git installed
- Ensure you have write access to the repository
- Try pushing manually:
  ```bash
  git add .
  git commit -m "Add report"
  git push origin main
  ```

---

## Files in This Repository

```
Statistical-Arbitrage-_1/
│
├── Statistical Arbitrage/
│   │
│   ├── Stat_Arb_User_Backtest.py    # Main user-friendly backtester
│   │
│   ├── Stat_Arb_Gold_pairs.py       # Specialized gold pairs analysis
│   ├── Stat_Arb_Multiple_pairs.py   # Multi-asset screening tool
│   ├── Stat_Arb_Multiple_pairs_with_fees.py  # With transaction costs
│   ├── Stat_Arb_Multiple_pairs_wo_fees.py    # Without transaction costs
│   │
│   ├── untitled76.py                # Live trading system (advanced)
│   │
│   ├── reports/                     # Generated HTML reports
│   │   └── Backtest_Report_XXX.html
│   │
│   └── USER_GUIDE.md                # This guide
│
└── README.md
```

### Which File Should I Use?

| Your Goal | Use This File |
|-----------|---------------|
| Quick backtest of any pair | `Stat_Arb_User_Backtest.py` |
| Test many pairs automatically | `Stat_Arb_Multiple_pairs.py` |
| Gold/mining pairs specifically | `Stat_Arb_Gold_pairs.py` |
| Include realistic trading fees | `Stat_Arb_Multiple_pairs_with_fees.py` |

---

## Glossary

| Term | Definition |
|------|------------|
| **Backtest** | Testing a strategy on historical data |
| **Cointegration** | Statistical property where two series move together over time |
| **Correlation** | Measure of how closely two assets move together (-1 to +1) |
| **Drawdown** | Decline from peak to trough in portfolio value |
| **Hedge Ratio** | The ratio of positions in each asset to maintain neutrality |
| **Mean Reversion** | Tendency of prices to return to their average |
| **Pairs Trading** | Strategy of trading two correlated assets |
| **Sharpe Ratio** | Risk-adjusted return (higher is better) |
| **Spread** | The price difference between two assets |
| **Ticker** | Symbol used to identify a stock (e.g., AAPL for Apple) |
| **Z-Score** | Number of standard deviations from the mean |

---

## Quick Reference Card

### Run the Backtester
```bash
cd "Statistical Arbitrage"
python Stat_Arb_User_Backtest.py
```

### Find Ticker Symbols
Visit [finance.yahoo.com](https://finance.yahoo.com) and search for any stock

### Date Format
Always use: `YYYY-MM-DD` (e.g., `2020-01-01`)

### Good Pair Criteria
- Correlation > 0.7
- Cointegration p-value < 0.05
- Quality Score > 60

### Save Report as PDF
`Ctrl + P` → "Save as PDF"

---

## Need Help?

1. Check the [Troubleshooting](#troubleshooting) section
2. Verify your Python packages are installed
3. Make sure ticker symbols are correct (check Yahoo Finance)
4. Try a well-known pair first (e.g., KO vs PEP)

---

*Last Updated: January 2026*
*Author: AJ*
