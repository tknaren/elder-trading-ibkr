"""
Elder Trading System - Backtesting Engine v2.3
==============================================

Reliable backtesting using the EXACT same scoring system as the live screener.

KEY FEATURES:
1. Uses v2.3 scoring system (same as screener)
2. Works with cached OHLCV data from database
3. Falls back to IBKR fetch if no cache available
4. Simulates Elder's entry/stop/target methodology
5. Accurate trade simulation with slippage consideration

SCORING SYSTEM (v2.3):
SCREEN 1 (Weekly) - Max 6 points:
  - MACD-H Rising: Spring (+2) or Summer (+1)
  - MACD Line < Signal: Both below 0 (+2) or just below (+1)
  - EMA Alignment: 20 > 50 > 100 (+2)

SCREEN 2 (Daily) - Max 5 points:
  - Price in KC pullback zone: Deep (+2) or Normal (+1)
  - Force Index < 0: (+1)
  - Stochastic < 50: (+1)
  - Bullish Pattern: (+1)

GRADES: A (≥7 + all weekly filters), B (5-6), C (1-4), AVOID (0)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict

# Import the actual scoring functions from screener
from services.indicators import (
    calculate_all_indicators,
    calculate_ema,
    calculate_macd,
    calculate_keltner_channel,
    calculate_atr
)
from services.candlestick_patterns import scan_patterns, get_bullish_patterns


@dataclass
class BacktestTrade:
    """Represents a single backtest trade"""
    signal_date: str
    entry_date: str
    entry_price: float
    stop_loss: float
    target: float
    quantity: int
    risk_per_share: float
    reward_per_share: float
    rr_ratio: float
    signal_strength: int
    grade: str
    
    # Exit info
    exit_date: Optional[str] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    
    # P&L
    pnl: float = 0.0
    pnl_percent: float = 0.0
    status: str = 'open'
    
    # Days held
    days_held: int = 0


@dataclass
class BacktestResult:
    """Complete backtest result"""
    symbol: str
    market: str
    period_days: int
    start_date: str
    end_date: str
    
    # Trade stats
    total_trades: int
    winning_trades: int
    losing_trades: int
    open_trades: int
    
    # Performance
    win_rate: float
    total_pnl: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    expectancy: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    max_drawdown: float
    avg_days_held: float
    
    # All trades
    trades: List[Dict]
    
    # Equity curve
    equity_curve: List[Dict]


class BacktestEngine:
    """
    Backtesting engine using v2.3 scoring system
    """
    
    def __init__(
        self,
        symbol: str,
        market: str = 'US',
        lookback_days: int = 180,
        initial_capital: float = 100000,
        risk_per_trade_pct: float = 1.0,
        rr_target: float = 1.5
    ):
        self.symbol = symbol
        self.market = market
        self.lookback_days = lookback_days
        self.initial_capital = initial_capital
        self.risk_per_trade_pct = risk_per_trade_pct
        self.rr_target = rr_target
        
        self.trades: List[BacktestTrade] = []
        self.equity_curve: List[Dict] = []
        self.current_capital = initial_capital
    
    def fetch_historical_data(self) -> Optional[pd.DataFrame]:
        """
        Fetch historical data - tries cache first, then IBKR
        """
        try:
            from models.database import get_database
            db = get_database().get_connection()
            
            # Calculate date range
            end_date = datetime.now().date()
            # Get extra data for indicator warm-up
            start_date = end_date - timedelta(days=self.lookback_days + 200)
            
            # Try to get from cache first
            rows = db.execute('''
                SELECT date, open, high, low, close, volume
                FROM stock_historical_data
                WHERE symbol = ? AND date >= ? AND date <= ?
                ORDER BY date ASC
            ''', (self.symbol, start_date.isoformat(), end_date.isoformat())).fetchall()
            
            if rows and len(rows) >= 100:
                df = pd.DataFrame([
                    {
                        'Date': row['date'],
                        'Open': float(row['open']),
                        'High': float(row['high']),
                        'Low': float(row['low']),
                        'Close': float(row['close']),
                        'Volume': int(row['volume'])
                    }
                    for row in rows
                ])
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                print(f"📦 {self.symbol}: Loaded {len(df)} cached bars for backtest")
                return df
            
            # Try IBKR if cache is insufficient
            print(f"🔄 {self.symbol}: Fetching from IBKR for backtest...")
            from services.ibkr_client import fetch_stock_data
            data = fetch_stock_data(self.symbol, period='2y')
            
            if data and 'history' in data:
                return data['history']
            
            return None
            
        except Exception as e:
            print(f"❌ {self.symbol}: Error fetching data: {e}")
            return None
    
    def analyze_weekly_trend(self, hist: pd.DataFrame) -> Dict:
        """
        Screen 1: Weekly Trend Analysis - EXACT same as screener v2.3
        """
        weekly = hist.resample('W').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min',
            'Close': 'last', 'Volume': 'sum'
        }).dropna()
        
        if len(weekly) < 20:
            return {
                'screen1_score': 0,
                'macd_h_score': 0,
                'macd_line_score': 0,
                'ema_alignment_score': 0,
                'weekly_bullish': False
            }
        
        closes = weekly['Close']
        
        # Calculate MACD
        macd = calculate_macd(closes)
        current_macd_h = macd['histogram'].iloc[-1]
        prev_macd_h = macd['histogram'].iloc[-2] if len(macd['histogram']) > 1 else current_macd_h
        current_macd_line = macd['macd_line'].iloc[-1]
        current_signal = macd['signal_line'].iloc[-1]
        
        macd_h_rising = current_macd_h > prev_macd_h
        
        # 1. MACD-H Rising Score
        macd_h_score = 0
        if macd_h_rising:
            if prev_macd_h < 0:  # Spring
                macd_h_score = 2
            else:  # Summer
                macd_h_score = 1
        
        # 2. MACD Line vs Signal Score
        macd_line_score = 0
        if current_macd_line < current_signal:
            if current_macd_line < 0 and current_signal < 0:
                macd_line_score = 2
            else:
                macd_line_score = 1
        
        # 3. EMA Alignment (20 > 50 > 100)
        data_len = len(closes)
        ema_20 = calculate_ema(closes, min(data_len, 20)).iloc[-1]
        ema_50 = calculate_ema(closes, min(data_len, 50)).iloc[-1]
        ema_100 = calculate_ema(closes, min(data_len, 100)).iloc[-1]
        
        ema_alignment_score = 0
        if ema_20 > ema_50 and ema_50 > ema_100:
            ema_alignment_score = 2
        
        screen1_score = macd_h_score + macd_line_score + ema_alignment_score
        
        return {
            'screen1_score': screen1_score,
            'macd_h_score': macd_h_score,
            'macd_line_score': macd_line_score,
            'ema_alignment_score': ema_alignment_score,
            'weekly_bullish': screen1_score >= 3
        }
    
    def calculate_daily_score(
        self,
        hist_slice: pd.DataFrame,
        weekly: Dict
    ) -> Dict:
        """
        Calculate signal strength using v2.3 scoring - EXACT same as screener
        """
        if len(hist_slice) < 50:
            return {
                'signal_strength': 0,
                'grade': 'AVOID',
                'is_a_trade': False,
                'screen1_score': 0,
                'screen2_score': 0
            }
        
        # Get indicators
        indicators = calculate_all_indicators(
            hist_slice['High'],
            hist_slice['Low'],
            hist_slice['Close'],
            hist_slice['Volume']
        )
        
        score = 0
        
        # ══════════════════════════════════════════════════════
        # SCREEN 1: Weekly scoring (from weekly dict)
        # ══════════════════════════════════════════════════════
        macd_h_score = weekly.get('macd_h_score', 0)
        macd_line_score = weekly.get('macd_line_score', 0)
        ema_alignment_score = weekly.get('ema_alignment_score', 0)
        
        score += macd_h_score + macd_line_score + ema_alignment_score
        screen1_score = macd_h_score + macd_line_score + ema_alignment_score
        
        # ══════════════════════════════════════════════════════
        # SCREEN 2: Daily scoring
        # ══════════════════════════════════════════════════════
        
        # 1. Price vs Keltner Channel
        price = indicators.get('price', hist_slice['Close'].iloc[-1])
        kc_middle = indicators.get('kc_middle', price)
        kc_lower = indicators.get('kc_lower', price * 0.97)
        atr = indicators.get('atr', price * 0.02)
        
        kc_lower_1 = kc_middle - atr
        kc_lower_3 = kc_middle - 3 * atr
        
        kc_score = 0
        if price >= kc_lower_3 and price < kc_lower_1:
            kc_score = 2  # Deep pullback
        elif price >= kc_lower_1 and price < kc_middle:
            kc_score = 1  # Normal pullback
        score += kc_score
        
        # 2. Force Index < 0
        force_index = indicators.get('force_index_2', 0)
        if force_index < 0:
            score += 1
        
        # 3. Stochastic < 50
        stochastic = indicators.get('stochastic_k', 50)
        if stochastic < 50:
            score += 1
        
        # 4. Bullish Pattern
        patterns = scan_patterns(hist_slice.tail(10))
        bullish_patterns = get_bullish_patterns(patterns)
        if bullish_patterns:
            score += 1
        
        screen2_score = score - screen1_score
        
        # ══════════════════════════════════════════════════════
        # GRADE DETERMINATION (v2.3 rules)
        # ══════════════════════════════════════════════════════
        all_weekly_filters_pass = (
            macd_h_score > 0 and 
            macd_line_score > 0 and 
            ema_alignment_score > 0
        )
        
        if all_weekly_filters_pass and score >= 7:
            grade = 'A'
        elif all_weekly_filters_pass and score >= 5:
            grade = 'B'
        elif score >= 7:
            grade = 'B'  # Downgraded from A due to missing weekly filter
        elif score >= 5:
            grade = 'B'
        elif score >= 1:
            grade = 'C'
        else:
            grade = 'AVOID'
        
        is_a_trade = grade == 'A'
        
        return {
            'signal_strength': score,
            'grade': grade,
            'is_a_trade': is_a_trade,
            'screen1_score': screen1_score,
            'screen2_score': screen2_score,
            'all_weekly_filters_pass': all_weekly_filters_pass,
            'kc_lower_1': kc_lower_1,
            'kc_lower_3': kc_lower_3,
            'indicators': indicators
        }
    
    def calculate_trade_levels(
        self,
        hist_slice: pd.DataFrame,
        indicators: Dict
    ) -> Dict:
        """
        Calculate Entry/Stop/Target using Elder's methodology
        """
        price = hist_slice['Close'].iloc[-1]
        prev_high = hist_slice['High'].iloc[-1]
        
        # Entry: Buy-stop above previous day's high
        entry = round(float(prev_high * 1.001), 2)  # 0.1% buffer
        
        # Stop: Below recent swing low or 2x ATR below entry
        atr = indicators.get('atr', price * 0.02)
        recent_low = hist_slice['Low'].tail(5).min()
        
        # Use whichever is tighter but not too tight
        atr_stop = entry - (2 * atr)
        swing_stop = recent_low * 0.998  # 0.2% buffer below swing low
        
        stop = round(float(max(atr_stop, swing_stop)), 2)
        
        # Target: Based on R:R ratio
        risk = entry - stop
        if risk <= 0:
            risk = entry * 0.02  # Fallback to 2%
            stop = round(entry - risk, 2)
        
        target = round(float(entry + (risk * self.rr_target)), 2)
        
        return {
            'entry': entry,
            'stop': stop,
            'target': target,
            'risk': round(risk, 2),
            'reward': round(target - entry, 2),
            'rr_ratio': round((target - entry) / risk, 2) if risk > 0 else 0
        }
    
    def simulate_trade(
        self,
        trade: BacktestTrade,
        future_bars: pd.DataFrame
    ) -> BacktestTrade:
        """
        Simulate trade execution on future bars
        """
        entry_triggered = False
        
        for i, (date, bar) in enumerate(future_bars.iterrows()):
            # Check entry trigger first (buy-stop)
            if not entry_triggered:
                if bar['High'] >= trade.entry_price:
                    entry_triggered = True
                    trade.entry_date = date.strftime('%Y-%m-%d')
                    # Entry at the entry price (buy-stop)
                    continue
                else:
                    # Entry not triggered, check if setup invalidated
                    if bar['Low'] < trade.stop_loss:
                        # Setup invalidated before entry
                        trade.status = 'cancelled'
                        trade.exit_reason = 'Setup invalidated before entry'
                        return trade
            
            # If entry triggered, check exit conditions
            if entry_triggered:
                trade.days_held += 1
                
                # Check stop loss first (conservative)
                if bar['Low'] <= trade.stop_loss:
                    trade.exit_date = date.strftime('%Y-%m-%d')
                    trade.exit_price = trade.stop_loss
                    trade.pnl = -trade.risk_per_share * trade.quantity
                    trade.pnl_percent = -100.0 * (trade.risk_per_share / trade.entry_price)
                    trade.status = 'loss'
                    trade.exit_reason = 'Stop Loss hit'
                    return trade
                
                # Check target
                if bar['High'] >= trade.target:
                    trade.exit_date = date.strftime('%Y-%m-%d')
                    trade.exit_price = trade.target
                    trade.pnl = trade.reward_per_share * trade.quantity
                    trade.pnl_percent = 100.0 * (trade.reward_per_share / trade.entry_price)
                    trade.status = 'win'
                    trade.exit_reason = 'Target hit'
                    return trade
        
        # Trade still open at end of data
        if entry_triggered:
            last_bar = future_bars.iloc[-1]
            trade.exit_date = future_bars.index[-1].strftime('%Y-%m-%d')
            trade.exit_price = last_bar['Close']
            trade.pnl = (trade.exit_price - trade.entry_price) * trade.quantity
            trade.pnl_percent = 100.0 * ((trade.exit_price - trade.entry_price) / trade.entry_price)
            trade.status = 'open'
            trade.exit_reason = 'End of data'
        else:
            trade.status = 'no_entry'
            trade.exit_reason = 'Entry never triggered'
        
        return trade
    
    def run(self, min_grade: str = 'B') -> Optional[BacktestResult]:
        """
        Run the backtest
        
        Args:
            min_grade: Minimum grade to take trades ('A', 'B', or 'C')
        """
        # Fetch data
        hist = self.fetch_historical_data()
        if hist is None or len(hist) < 100:
            print(f"❌ {self.symbol}: Insufficient data for backtest")
            return None
        
        print(f"📊 {self.symbol}: Running backtest on {len(hist)} bars...")
        
        # Determine backtest window
        total_bars = len(hist)
        warmup_bars = 100  # For indicator calculation
        
        # Start from warmup_bars index
        backtest_start_idx = warmup_bars
        
        # Calculate risk per trade
        risk_amount = self.initial_capital * (self.risk_per_trade_pct / 100)
        
        # Track active trade to avoid overlapping
        active_trade = None
        
        # Iterate through each day
        for i in range(backtest_start_idx, total_bars - 1):
            current_date = hist.index[i]
            
            # Skip if we have an active trade
            if active_trade is not None:
                if active_trade.status in ['win', 'loss', 'cancelled', 'no_entry', 'open']:
                    active_trade = None
                else:
                    continue
            
            # Get history up to current day
            hist_slice = hist.iloc[:i+1]
            
            # Analyze weekly trend
            weekly = self.analyze_weekly_trend(hist_slice)
            
            # Calculate daily score
            scoring = self.calculate_daily_score(hist_slice, weekly)
            
            # Check if signal meets minimum grade
            grade = scoring['grade']
            
            grade_order = {'A': 1, 'B': 2, 'C': 3, 'AVOID': 4}
            min_grade_order = grade_order.get(min_grade, 2)
            current_grade_order = grade_order.get(grade, 4)
            
            if current_grade_order > min_grade_order:
                continue
            
            # Calculate trade levels
            levels = self.calculate_trade_levels(
                hist_slice,
                scoring.get('indicators', {})
            )
            
            # Skip if invalid levels
            if levels['risk'] <= 0:
                continue
            
            # Calculate position size
            quantity = int(risk_amount / levels['risk'])
            if quantity <= 0:
                continue
            
            # Create trade
            trade = BacktestTrade(
                signal_date=current_date.strftime('%Y-%m-%d'),
                entry_date='',
                entry_price=levels['entry'],
                stop_loss=levels['stop'],
                target=levels['target'],
                quantity=quantity,
                risk_per_share=levels['risk'],
                reward_per_share=levels['reward'],
                rr_ratio=levels['rr_ratio'],
                signal_strength=scoring['signal_strength'],
                grade=grade
            )
            
            # Simulate trade on future bars
            future_bars = hist.iloc[i+1:]
            trade = self.simulate_trade(trade, future_bars)
            
            # Add to trades list
            if trade.status != 'no_entry':
                self.trades.append(trade)
                
                # Update equity
                self.current_capital += trade.pnl
                self.equity_curve.append({
                    'date': trade.exit_date or trade.signal_date,
                    'equity': round(self.current_capital, 2),
                    'trade_pnl': round(trade.pnl, 2)
                })
            
            # Set active trade if still running
            if trade.status == 'open':
                active_trade = trade
        
        # Calculate statistics
        return self._calculate_results(hist)
    
    def _calculate_results(self, hist: pd.DataFrame) -> BacktestResult:
        """Calculate backtest statistics"""
        closed_trades = [t for t in self.trades if t.status in ['win', 'loss']]
        winning_trades = [t for t in closed_trades if t.status == 'win']
        losing_trades = [t for t in closed_trades if t.status == 'loss']
        open_trades = [t for t in self.trades if t.status == 'open']
        
        # Basic stats
        total_trades = len(self.trades)
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        
        # P&L
        total_wins = sum(t.pnl for t in winning_trades)
        total_losses = sum(t.pnl for t in losing_trades)
        total_pnl = total_wins + total_losses
        
        # Averages
        avg_win = total_wins / win_count if win_count > 0 else 0
        avg_loss = abs(total_losses / loss_count) if loss_count > 0 else 0
        
        # Win rate
        win_rate = (win_count / len(closed_trades) * 100) if closed_trades else 0
        
        # Profit factor
        profit_factor = abs(total_wins / total_losses) if total_losses != 0 else 0
        
        # Expectancy
        expectancy = (win_rate/100 * avg_win) - ((100-win_rate)/100 * avg_loss)
        
        # Consecutive wins/losses
        max_wins, max_losses = self._calculate_streaks(closed_trades)
        
        # Max drawdown
        max_dd = self._calculate_max_drawdown()
        
        # Average days held
        days_list = [t.days_held for t in closed_trades if t.days_held > 0]
        avg_days = sum(days_list) / len(days_list) if days_list else 0
        
        return BacktestResult(
            symbol=self.symbol,
            market=self.market,
            period_days=self.lookback_days,
            start_date=hist.index[0].strftime('%Y-%m-%d'),
            end_date=hist.index[-1].strftime('%Y-%m-%d'),
            total_trades=total_trades,
            winning_trades=win_count,
            losing_trades=loss_count,
            open_trades=len(open_trades),
            win_rate=round(win_rate, 2),
            total_pnl=round(total_pnl, 2),
            avg_win=round(avg_win, 2),
            avg_loss=round(avg_loss, 2),
            profit_factor=round(profit_factor, 2),
            expectancy=round(expectancy, 2),
            max_consecutive_wins=max_wins,
            max_consecutive_losses=max_losses,
            max_drawdown=round(max_dd, 2),
            avg_days_held=round(avg_days, 1),
            trades=[asdict(t) for t in self.trades],
            equity_curve=self.equity_curve
        )
    
    def _calculate_streaks(self, closed_trades: List[BacktestTrade]) -> Tuple[int, int]:
        """Calculate max consecutive wins and losses"""
        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0
        
        for trade in closed_trades:
            if trade.status == 'win':
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
        
        return max_wins, max_losses
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown percentage"""
        if not self.equity_curve:
            return 0.0
        
        peak = self.initial_capital
        max_dd = 0.0
        
        for point in self.equity_curve:
            equity = point['equity']
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100
            max_dd = max(max_dd, dd)
        
        return max_dd


def run_backtest(
    symbol: str,
    market: str = 'US',
    lookback_days: int = 180,
    initial_capital: float = 100000,
    risk_per_trade_pct: float = 1.0,
    rr_target: float = 1.5,
    min_grade: str = 'B'
) -> Optional[Dict]:
    """
    Run backtest for a single symbol
    
    Args:
        symbol: Stock symbol
        market: 'US' or 'INDIA'
        lookback_days: Days of history to backtest
        initial_capital: Starting capital
        risk_per_trade_pct: Risk per trade as percentage
        rr_target: Risk:Reward target ratio
        min_grade: Minimum grade to take trades ('A', 'B', 'C')
    
    Returns:
        Backtest result dictionary
    """
    engine = BacktestEngine(
        symbol=symbol,
        market=market,
        lookback_days=lookback_days,
        initial_capital=initial_capital,
        risk_per_trade_pct=risk_per_trade_pct,
        rr_target=rr_target
    )
    
    result = engine.run(min_grade=min_grade)
    
    if result:
        return asdict(result)
    return None


def run_portfolio_backtest(
    symbols: List[str],
    market: str = 'US',
    lookback_days: int = 180,
    initial_capital: float = 100000,
    risk_per_trade_pct: float = 1.0,
    rr_target: float = 1.5,
    min_grade: str = 'B'
) -> Dict:
    """
    Run backtest across multiple symbols
    
    Returns aggregated results
    """
    results = []
    all_trades = []
    total_pnl = 0
    
    for symbol in symbols:
        result = run_backtest(
            symbol=symbol,
            market=market,
            lookback_days=lookback_days,
            initial_capital=initial_capital,
            risk_per_trade_pct=risk_per_trade_pct,
            rr_target=rr_target,
            min_grade=min_grade
        )
        
        if result:
            results.append(result)
            all_trades.extend(result['trades'])
            total_pnl += result['total_pnl']
    
    # Aggregate stats
    total_trades = sum(r['total_trades'] for r in results)
    winning = sum(r['winning_trades'] for r in results)
    losing = sum(r['losing_trades'] for r in results)
    
    return {
        'symbols_tested': len(symbols),
        'symbols_with_trades': len([r for r in results if r['total_trades'] > 0]),
        'total_trades': total_trades,
        'winning_trades': winning,
        'losing_trades': losing,
        'win_rate': round(winning / total_trades * 100, 2) if total_trades > 0 else 0,
        'total_pnl': round(total_pnl, 2),
        'avg_pnl_per_symbol': round(total_pnl / len(results), 2) if results else 0,
        'individual_results': results,
        'all_trades': sorted(all_trades, key=lambda x: x.get('signal_date', ''))
    }
