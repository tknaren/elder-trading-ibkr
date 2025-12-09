"""
Elder Trading System - Backtesting Engine
Provides historical backtesting with configurable indicators
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from models.database import get_database
from services.indicators import calculate_all_indicators


class BacktestEngine:
    """Backtesting engine for strategy validation"""

    def __init__(self, symbol: str, market: str = 'US', lookback_days: int = 90):
        self.symbol = symbol
        self.market = market
        self.lookback_days = lookback_days
        self.db = get_database().get_connection()
        self.trades = []
        self.stats = {}

    def fetch_historical_data(self) -> List[Dict]:
        """Fetch lookback period of historical OHLCV data"""
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=self.lookback_days)

        data = self.db.execute('''
            SELECT date, open, high, low, close, volume
            FROM stock_historical_data
            WHERE symbol = ? AND date BETWEEN ? AND ?
            ORDER BY date ASC
        ''', (self.symbol, start_date.isoformat(), end_date.isoformat())).fetchall()

        return [dict(row) for row in data]

    def calculate_daily_indicators(self, history_slice: List[Dict]) -> Dict:
        """Calculate indicators using pandas DataFrame"""
        try:
            if len(history_slice) < 5:
                return {}
            
            # Create DataFrame for indicator calculation
            df = pd.DataFrame(history_slice)
            df['Open'] = df['open'].astype(float)
            df['High'] = df['high'].astype(float)
            df['Low'] = df['low'].astype(float)
            df['Close'] = df['close'].astype(float)
            df['Volume'] = df['volume'].astype(float)
            
            # Calculate indicators
            indicators = calculate_all_indicators(
                df['High'], df['Low'], df['Close'], df['Volume']
            )
            
            # Return latest indicators as dict
            if len(indicators) > 0:
                latest = indicators.iloc[-1]
                return latest.to_dict()
            return {}
        except Exception as e:
            print(f"Error calculating indicators: {e}")
            return {}

    def score_stock_simple(self, indicators: Dict, current: Dict, previous: Dict) -> Dict:
        """Simple scoring based on key indicators for backtesting"""
        try:
            score = 0
            is_a_trade = False
            
            # EMA alignment check
            ema_22 = float(indicators.get('ema_22', 0))
            ema_50 = float(indicators.get('ema_50', 0))
            ema_100 = float(indicators.get('ema_100', 0))
            
            # Check EMA alignment: 22>50>100 for bullish A-trade
            if ema_22 > ema_50 > ema_100:
                score += 3
                is_a_trade = True
            
            # MACD histogram rising
            macd_h = float(indicators.get('macd_histogram', 0))
            if macd_h > 0:
                score += 1
            
            # Force Index
            force_idx = float(indicators.get('force_index_2', 0))
            if force_idx > 0:
                score += 1
            
            # RSI
            rsi = float(indicators.get('rsi', 50))
            if 30 < rsi < 70:
                score += 1
            
            return {
                'signal_strength': score,
                'is_a_trade': is_a_trade,
                'score_breakdown': {
                    'ema_alignment': 3 if is_a_trade else 0,
                    'macd': 1 if macd_h > 0 else 0,
                    'force_index': 1 if force_idx > 0 else 0,
                    'rsi': 1 if 30 < rsi < 70 else 0
                }
            }
        except Exception as e:
            print(f"Error scoring: {e}")
            return {'signal_strength': 0, 'is_a_trade': False}

    def check_entry_signal(self, current: Dict, previous: Dict) -> Tuple[bool, float]:
        """
        Check if entry signal is triggered
        Entry: Breakout of previous day candle high

        Returns: (signal_triggered, entry_price)
        """
        prev_high = float(previous.get('high', 0))
        current_close = float(current.get('close', 0))
        current_open = float(current.get('open', 0))
        
        # Entry when today closes above yesterday's high
        if current_close > prev_high:
            # Entry price is max of open or previous high
            entry_price = max(prev_high + 0.01, current_open)
            return True, round(entry_price, 2)
        
        return False, 0

    def calculate_sl_and_target(self, entry: float, prev_candle: Dict) -> Tuple[float, float]:
        """
        Calculate stop loss and target
        SL: Previous day low
        Target: 1.5 R:R ratio
        """
        prev_low = float(prev_candle.get('low', 0))
        risk = entry - prev_low
        
        if risk <= 0:
            return 0, 0
        
        target = entry + (1.5 * risk)
        return round(prev_low, 2), round(target, 2)

    def run_backtest(self, config: Optional[Dict] = None) -> Dict:
        """
        Run complete backtest for the symbol
        """
        history = self.fetch_historical_data()
        
        if len(history) < 2:
            return {
                'symbol': self.symbol,
                'error': 'Insufficient historical data',
                'trades': [],
                'stats': {}
            }
        
        self.trades = []
        
        # Process each candle starting from index 1 (need previous candle)
        for i in range(1, len(history)):
            current = history[i]
            previous = history[i-1]
            
            # Calculate indicators for current candle (use up to current index)
            indicators = self.calculate_daily_indicators(history[:i+1])
            
            if not indicators:
                continue
            
            # Score the stock
            score_data = self.score_stock_simple(indicators, current, previous)
            
            # Check if A-Trade or score > 7
            signal_strength = score_data.get('signal_strength', 0)
            is_a_trade = score_data.get('is_a_trade', False)
            
            if not (is_a_trade or signal_strength > 7):
                continue
            
            # Check entry signal
            entry_triggered, entry_price = self.check_entry_signal(current, previous)
            
            if not entry_triggered:
                continue
            
            # Calculate SL and Target
            sl_price, target_price = self.calculate_sl_and_target(entry_price, previous)
            
            if sl_price <= 0 or target_price <= 0:
                continue
            
            # Create trade record
            trade = {
                'date': current['date'],
                'entry_date': current['date'],
                'entry_price': round(entry_price, 2),
                'stop_loss': round(sl_price, 2),
                'target': round(target_price, 2),
                'risk': round(entry_price - sl_price, 2),
                'reward': round(target_price - entry_price, 2),
                'rr_ratio': round((target_price - entry_price) / (entry_price - sl_price), 2) if entry_price != sl_price else 0,
                'signal_strength': signal_strength,
                'is_a_trade': is_a_trade,
                'status': 'open',
                'exit_price': None,
                'exit_date': None,
                'pnl': None,
                'pnl_pct': None
            }
            
            # Check if trade hits target or SL in subsequent candles
            trade = self._simulate_trade_exit(trade, history[i+1:])
            
            self.trades.append(trade)
        
        # Calculate statistics
        self.stats = self._calculate_statistics()
        
        return {
            'symbol': self.symbol,
            'market': self.market,
            'period': f'{self.lookback_days} days',
            'trades': self.trades,
            'stats': self.stats
        }
    
    def _simulate_trade_exit(self, trade: Dict, remaining_history: List[Dict]) -> Dict:
        """Simulate trade exit - check if hit SL or target"""
        for future_candle in remaining_history:
            high = float(future_candle.get('high', 0))
            low = float(future_candle.get('low', 0))
            
            # Check if hit target
            if high >= trade['target']:
                trade['exit_date'] = future_candle['date']
                trade['exit_price'] = trade['target']
                trade['pnl'] = round(trade['reward'], 2)
                trade['pnl_pct'] = round((trade['reward'] / trade['risk']) * 100, 2) if trade['risk'] > 0 else 0
                trade['status'] = 'win'
                return trade
            
            # Check if hit stop loss
            if low <= trade['stop_loss']:
                trade['exit_date'] = future_candle['date']
                trade['exit_price'] = trade['stop_loss']
                trade['pnl'] = round(-trade['risk'], 2)
                trade['pnl_pct'] = -100.0
                trade['status'] = 'loss'
                return trade
        
        # Trade still open
        trade['status'] = 'open'
        return trade

    def _calculate_statistics(self) -> Dict:
        """Calculate backtest statistics"""
        if not self.trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'open_trades': 0,
                'win_percentage': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'total_pnl': 0,
                'expectancy': 0,
                'max_win_streak': 0
            }

        closed_trades = [t for t in self.trades if t['status'] != 'open']
        winning_trades = [t for t in closed_trades if t['status'] == 'win']
        losing_trades = [t for t in closed_trades if t['status'] == 'loss']
        open_trades = [t for t in self.trades if t['status'] == 'open']

        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        
        total_wins = sum([t['pnl'] for t in winning_trades]) if winning_trades else 0
        total_losses = sum([t['pnl'] for t in losing_trades]) if losing_trades else 0
        total_pnl = total_wins + total_losses
        
        avg_win = (total_wins / win_count) if win_count > 0 else 0
        avg_loss = abs(total_losses / loss_count) if loss_count > 0 else 0
        
        profit_factor = abs(total_wins / total_losses) if total_losses != 0 else 0
        win_pct = (win_count / len(closed_trades) * 100) if closed_trades else 0
        
        expectancy = (win_pct/100 * avg_win) - ((100-win_pct)/100 * avg_loss)
        
        # Calculate max win streak
        max_streak = 0
        current_streak = 0
        for trade in closed_trades:
            if trade['status'] == 'win':
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return {
            'total_trades': len(self.trades),
            'winning_trades': win_count,
            'losing_trades': loss_count,
            'open_trades': len(open_trades),
            'win_percentage': round(win_pct, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'profit_factor': round(profit_factor, 2),
            'total_pnl': round(total_pnl, 2),
            'expectancy': round(expectancy, 2),
            'max_win_streak': max_streak
        }


def run_backtest_for_symbol(symbol: str, market: str = 'US', lookback_days: int = 90, config: Optional[Dict] = None) -> Dict:
    """Helper function to run backtest"""
    engine = BacktestEngine(symbol, market, lookback_days)
    return engine.run_backtest(config)
