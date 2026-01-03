# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 18:15:40 2025

@author: AJ
"""

import asyncio
import logging
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import deque
import aiohttp
import boto3
from botocore.exceptions import ClientError
import yaml
import os
from enum import Enum

# Configuration Management
@dataclass
class TradingConfig:
    # Instrument Configuration
    instrument: str = "XAUUSD"
    spot_symbol: str = "XAUUSD"
    future_symbol: str = "XAUUSD_FUT"
    
    # Strategy Parameters
    basis_threshold_std: float = 1.5  # Standard deviation threshold
    lookback_window: int = 100  # Rolling window for statistics
    position_size: float = 1000.0  # Position size in USD
    
    # Risk Management
    max_position_value: float = 100000.0
    stop_loss_pct: float = 0.02  # 2% stop loss
    
    # Market Data
    price_update_interval: int = 1  # seconds
    risk_free_rate: float = 0.05  # 5% annual risk-free rate
    storage_cost: float = 0.001  # 0.1% annual storage cost for gold
    
    # MTS Connection
    mts_api_url: str = "https://api.mts.com/v1"
    mts_api_key: str = ""
    mts_secret: str = ""
    
    # AWS Configuration
    aws_region: str = "us-east-1"
    cloudwatch_namespace: str = "BasisTrading"
    
    # Logging
    log_level: str = "INFO"
    log_to_cloudwatch: bool = True

class TradeAction(Enum):
    BUY_SPOT_SELL_FUTURE = "buy_spot_sell_future"
    SELL_SPOT_BUY_FUTURE = "sell_spot_buy_future"
    CLOSE_POSITION = "close_position"
    HOLD = "hold"

@dataclass
class MarketData:
    timestamp: datetime
    spot_price: float
    future_price: float
    basis: float
    cost_of_carry: float
    time_to_expiry: float

@dataclass
class Position:
    instrument: str
    spot_quantity: float = 0.0
    future_quantity: float = 0.0
    entry_basis: float = 0.0
    entry_time: datetime = None
    unrealized_pnl: float = 0.0

class MTSConnector:
    """MTS Platform API Connector"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.session = None
        self.authenticated = False
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        await self.authenticate()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def authenticate(self):
        """Authenticate with MTS platform"""
        try:
            auth_payload = {
                "api_key": self.config.mts_api_key,
                "secret": self.config.mts_secret
            }
            
            async with self.session.post(
                f"{self.config.mts_api_url}/auth",
                json=auth_payload
            ) as response:
                if response.status == 200:
                    auth_data = await response.json()
                    self.session.headers.update({
                        "Authorization": f"Bearer {auth_data.get('access_token')}"
                    })
                    self.authenticated = True
                    logging.info("Successfully authenticated with MTS")
                else:
                    raise Exception(f"Authentication failed: {response.status}")
        
        except Exception as e:
            logging.error(f"MTS authentication error: {e}")
            raise
    
    async def get_market_data(self, symbol: str) -> Dict:
        """Fetch real-time market data"""
        try:
            async with self.session.get(
                f"{self.config.mts_api_url}/market-data/{symbol}"
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise Exception(f"Market data fetch failed: {response.status}")
        
        except Exception as e:
            logging.error(f"Market data error for {symbol}: {e}")
            raise
    
    async def place_order(self, symbol: str, side: str, quantity: float) -> Dict:
        """Place trading order"""
        try:
            order_payload = {
                "symbol": symbol,
                "side": side,  # "buy" or "sell"
                "quantity": quantity,
                "type": "market"
            }
            
            async with self.session.post(
                f"{self.config.mts_api_url}/orders",
                json=order_payload
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise Exception(f"Order placement failed: {response.status}")
        
        except Exception as e:
            logging.error(f"Order placement error: {e}")
            raise

class BasisTradingEngine:
    """Main basis trading engine"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self.load_config(config_path)
        self.setup_logging()
        
        # Data storage
        self.price_history = deque(maxlen=self.config.lookback_window * 2)
        self.position = Position(instrument=self.config.instrument)
        
        # AWS clients
        self.cloudwatch = boto3.client('cloudwatch', region_name=self.config.aws_region)
        
        # Statistics
        self.basis_stats = {
            'mean': 0.0,
            'std': 0.0,
            'last_update': datetime.now()
        }
        
        self.running = False
    
    def load_config(self, config_path: str) -> TradingConfig:
        """Load configuration from YAML file"""
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as file:
                    config_dict = yaml.safe_load(file)
                return TradingConfig(**config_dict)
            else:
                logging.warning(f"Config file {config_path} not found, using defaults")
                return TradingConfig()
        except Exception as e:
            logging.error(f"Error loading config: {e}")
            return TradingConfig()
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        if self.config.log_to_cloudwatch:
            # Add CloudWatch logging handler here if needed
            pass
    
    def calculate_cost_of_carry(self, spot_price: float, time_to_expiry: float) -> float:
        """Calculate theoretical cost of carry"""
        # Cost of carry = Spot * (risk_free_rate + storage_cost) * time_to_expiry
        return spot_price * (self.config.risk_free_rate + self.config.storage_cost) * time_to_expiry
    
    def calculate_time_to_expiry(self, expiry_date: datetime) -> float:
        """Calculate time to expiry in years"""
        now = datetime.now()
        time_diff = expiry_date - now
        return max(0, time_diff.days / 365.25)
    
    def update_basis_statistics(self, basis_values: List[float]):
        """Update rolling basis statistics"""
        if len(basis_values) >= 30:  # Minimum samples for reliable statistics
            self.basis_stats['mean'] = np.mean(basis_values)
            self.basis_stats['std'] = np.std(basis_values)
            self.basis_stats['last_update'] = datetime.now()
    
    def generate_signal(self, market_data: MarketData) -> TradeAction:
        """Generate trading signal based on basis analysis"""
        if len(self.price_history) < self.config.lookback_window:
            return TradeAction.HOLD
        
        # Extract recent basis values
        recent_basis = [data.basis for data in list(self.price_history)[-self.config.lookback_window:]]
        self.update_basis_statistics(recent_basis)
        
        # Calculate z-score of current basis relative to cost of carry
        basis_vs_carry = market_data.basis - market_data.cost_of_carry
        
        if self.basis_stats['std'] == 0:
            return TradeAction.HOLD
        
        z_score = (basis_vs_carry - np.mean([data.basis - data.cost_of_carry 
                                          for data in list(self.price_history)[-self.config.lookback_window:]])) / self.basis_stats['std']
        
        logging.info(f"Basis: {market_data.basis:.4f}, Cost of Carry: {market_data.cost_of_carry:.4f}, Z-Score: {z_score:.2f}")
        
        # Generate signals
        if self.position.spot_quantity == 0:  # No position
            if z_score > self.config.basis_threshold_std:
                return TradeAction.BUY_SPOT_SELL_FUTURE
            elif z_score < -self.config.basis_threshold_std:
                return TradeAction.SELL_SPOT_BUY_FUTURE
        else:  # Has position
            # Check for convergence or stop loss
            if abs(z_score) < 0.5:  # Basis has converged
                return TradeAction.CLOSE_POSITION
            elif market_data.time_to_expiry < 0.01:  # Close to expiry
                return TradeAction.CLOSE_POSITION
        
        return TradeAction.HOLD
    
    async def execute_trade(self, action: TradeAction, market_data: MarketData, mts_connector: MTSConnector):
        """Execute trading action"""
        try:
            if action == TradeAction.BUY_SPOT_SELL_FUTURE:
                # Buy spot, sell future
                spot_qty = self.config.position_size / market_data.spot_price
                future_qty = self.config.position_size / market_data.future_price
                
                await mts_connector.place_order(self.config.spot_symbol, "buy", spot_qty)
                await mts_connector.place_order(self.config.future_symbol, "sell", future_qty)
                
                self.position.spot_quantity = spot_qty
                self.position.future_quantity = -future_qty
                self.position.entry_basis = market_data.basis
                self.position.entry_time = market_data.timestamp
                
                logging.info(f"Executed: Buy {spot_qty:.4f} spot, Sell {future_qty:.4f} future")
            
            elif action == TradeAction.SELL_SPOT_BUY_FUTURE:
                # Sell spot, buy future
                spot_qty = self.config.position_size / market_data.spot_price
                future_qty = self.config.position_size / market_data.future_price
                
                await mts_connector.place_order(self.config.spot_symbol, "sell", spot_qty)
                await mts_connector.place_order(self.config.future_symbol, "buy", future_qty)
                
                self.position.spot_quantity = -spot_qty
                self.position.future_quantity = future_qty
                self.position.entry_basis = market_data.basis
                self.position.entry_time = market_data.timestamp
                
                logging.info(f"Executed: Sell {spot_qty:.4f} spot, Buy {future_qty:.4f} future")
            
            elif action == TradeAction.CLOSE_POSITION:
                # Close existing positions
                if self.position.spot_quantity != 0:
                    side = "sell" if self.position.spot_quantity > 0 else "buy"
                    await mts_connector.place_order(self.config.spot_symbol, side, abs(self.position.spot_quantity))
                
                if self.position.future_quantity != 0:
                    side = "buy" if self.position.future_quantity < 0 else "sell"
                    await mts_connector.place_order(self.config.future_symbol, side, abs(self.position.future_quantity))
                
                # Calculate PnL
                pnl = self.calculate_pnl(market_data)
                logging.info(f"Position closed. PnL: ${pnl:.2f}")
                
                # Reset position
                self.position = Position(instrument=self.config.instrument)
        
        except Exception as e:
            logging.error(f"Trade execution error: {e}")
    
    def calculate_pnl(self, market_data: MarketData) -> float:
        """Calculate current PnL"""
        spot_pnl = self.position.spot_quantity * market_data.spot_price
        future_pnl = self.position.future_quantity * market_data.future_price
        return spot_pnl + future_pnl
    
    async def send_metrics_to_cloudwatch(self, market_data: MarketData):
        """Send metrics to AWS CloudWatch"""
        try:
            metrics = [
                {
                    'MetricName': 'SpotPrice',
                    'Value': market_data.spot_price,
                    'Unit': 'None',
                    'Dimensions': [{'Name': 'Instrument', 'Value': self.config.instrument}]
                },
                {
                    'MetricName': 'FuturePrice',
                    'Value': market_data.future_price,
                    'Unit': 'None',
                    'Dimensions': [{'Name': 'Instrument', 'Value': self.config.instrument}]
                },
                {
                    'MetricName': 'Basis',
                    'Value': market_data.basis,
                    'Unit': 'None',
                    'Dimensions': [{'Name': 'Instrument', 'Value': self.config.instrument}]
                },
                {
                    'MetricName': 'UnrealizedPnL',
                    'Value': self.calculate_pnl(market_data),
                    'Unit': 'None',
                    'Dimensions': [{'Name': 'Instrument', 'Value': self.config.instrument}]
                }
            ]
            
            self.cloudwatch.put_metric_data(
                Namespace=self.config.cloudwatch_namespace,
                MetricData=metrics
            )
        
        except Exception as e:
            logging.error(f"CloudWatch metrics error: {e}")
    
    async def run_trading_loop(self):
        """Main trading loop"""
        self.running = True
        
        async with MTSConnector(self.config) as mts_connector:
            while self.running:
                try:
                    # Fetch market data
                    spot_data = await mts_connector.get_market_data(self.config.spot_symbol)
                    future_data = await mts_connector.get_market_data(self.config.future_symbol)
                    
                    # Extract prices
                    spot_price = spot_data.get('price', 0.0)
                    future_price = future_data.get('price', 0.0)
                    
                    # Assume future expires in 3 months (configurable)
                    expiry_date = datetime.now() + timedelta(days=90)
                    time_to_expiry = self.calculate_time_to_expiry(expiry_date)
                    
                    # Calculate basis and cost of carry
                    basis = future_price - spot_price
                    cost_of_carry = self.calculate_cost_of_carry(spot_price, time_to_expiry)
                    
                    # Create market data object
                    market_data = MarketData(
                        timestamp=datetime.now(),
                        spot_price=spot_price,
                        future_price=future_price,
                        basis=basis,
                        cost_of_carry=cost_of_carry,
                        time_to_expiry=time_to_expiry
                    )
                    
                    # Store historical data
                    self.price_history.append(market_data)
                    
                    # Generate and execute trading signal
                    action = self.generate_signal(market_data)
                    if action != TradeAction.HOLD:
                        await self.execute_trade(action, market_data, mts_connector)
                    
                    # Send metrics to CloudWatch
                    await self.send_metrics_to_cloudwatch(market_data)
                    
                    # Update position PnL
                    self.position.unrealized_pnl = self.calculate_pnl(market_data)
                    
                    # Log current status
                    logging.info(f"Spot: ${spot_price:.2f}, Future: ${future_price:.2f}, "
                               f"Basis: ${basis:.2f}, Action: {action.value}")
                    
                except Exception as e:
                    logging.error(f"Trading loop error: {e}")
                
                # Wait for next iteration
                await asyncio.sleep(self.config.price_update_interval)
    
    def stop(self):
        """Stop the trading engine"""
        self.running = False
        logging.info("Trading engine stopped")

async def main():
    """Main entry point"""
    # Load configuration
    config_path = os.getenv('CONFIG_PATH', 'config.yaml')
    
    # Initialize trading engine
    engine = BasisTradingEngine(config_path)
    
    try:
        logging.info("Starting XAUUSD Basis Trading System...")
        await engine.run_trading_loop()
    except KeyboardInterrupt:
        logging.info("Shutting down...")
    except Exception as e:
        logging.error(f"Critical error: {e}")
    finally:
        engine.stop()

if __name__ == "__main__":
    # Run the trading system
    asyncio.run(main())