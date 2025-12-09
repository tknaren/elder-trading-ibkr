"""
Elder Trading System - Practical Backtesting Engine
====================================================

A PRACTICAL backtesting engine that follows Elder's CORE principle:
"Weekly UP + Daily DOWN = Go Long"

The v2.3 scoring system is too strict for backtesting (generates 0 trades).
This module uses Elder's essential methodology without excessive filtering.

CORE LOGIC (from Elder's Trading Table):
┌─────────────┬─────────────┬───────────────┐
│ Weekly Trend│ Daily Trend │ Action        │
├─────────────┼─────────────┼───────────────┤
│ UP          │ DOWN        │ GO LONG ✓     │
│ UP          │ UP          │ Stand Aside   │
│ DOWN        │ Any         │ Stand Aside   │
└─────────────┴─────────────┴───────────────┘

WEEKLY UP = EMA-22 rising OR MACD-H rising
DAILY DOWN = Force Index < 0 OR Stochastic < 50 OR Price below EMA

ENTRY: Buy-stop above previous day's high
STOP: Below recent swing low (5 bars) or 2×ATR
TARGET: Based on configurable R:R ratio (default 1.5)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field

from services.indicators import (
    calculate_all_indicators,
    calculate_ema,
    calculate_macd,
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
    
    # Signal details
    weekly_trend: str = ''
    daily_pullback: str = ''
    
    # Exit info
    exit_date: Optional[str] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    
    # P&L
    pnl: float = 0.0
    pnl_percent: float = 0.0
    status: str = 'pending'
    
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
    data_bars: int
    
    # Trade stats
    total_signals: int
    total_trades: int
    winning_trades: int
    losing_trades: int
    cancelled_trades: int
    open_trades: int
    
    # Performance
    win_rate: float
    total_pnl: float
    total_pnl_percent: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    expectancy: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    max_drawdown: float
    avg_days_held: float
    
    # Capital tracking
    initial_capital: float
    final_capital: float
    return_percent: float
    
    # All trades
    trades: List[Dict] = field(default_factory=list)
    
    # Equity curve
    equity_curve: List[Dict] = field(default_factory=list)


class PracticalBacktestEngine:
    """
    Practical backtesting engine using Elder's core methodology
    """
    
    def __init__(
        self,
        symbol: str,
        market: str = 'US',
        lookback_days: int = 365,
        initial_capital: float = 100000,
        risk_per_trade_pct: float = 2.0,
        rr_target: float = 1.5,
        max_concurrent_trades: int = 1
    ):
        self.symbol = symbol
        self.market = market
        self.lookback_days = lookback_days
        self.initial_capital = initial_capital
        self.risk_per_trade_pct = risk_per_trade_pct
        self.rr_target = rr_target
        self.max_concurrent_trades = max_concurrent_trades
        
        self.trades: List[BacktestTrade] = []
        self.equity_curve: List[Dict] = []
        self.current_capital = initial_capital
        self.total_signals = 0
    
    def fetch_historical_data(self) -> Optional[pd.DataFrame]:
        """Fetch historical data from cache or IBKR"""
        try:
            from models.database import get_database
            db = get_database().get_connection()
            
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=self.lookback_days + 200)
            
            rows = db.execute('''
                SELECT date, open, high, low, close, volume
                FROM stock_historical_data
                WHERE symbol = ? AND date >= ? AND date <= ?
                ORDER BY date ASC
            ''', (self.symbol, start_date.isoformat(), end_date.isoformat())).fetchall()
            
            if rows and len(rows) >= 100:
                df = pd.DataFrame([{
                    'Date': row['date'],
                    'Open': float(row['open']),
                    'High': float(row['high']),
                    'Low': float(row['low']),
                    'Close': float(row['close']),
                    'Volume': int(row['volume'])
                } for row in rows])
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                return df
            
            # Try IBKR
            from services.ibkr_client import fetch_stock_data
            data = fetch_stock_data(self.symbol, period='2y')
            if data and 'history' in data:
                return data['history']
            
            return None
            
        except Exception as e:
            print(f"❌ {self.symbol}: Error fetching data: {e}")
            return None
    
    def check_weekly_uptrend(self, hist_slice: pd.DataFrame) -> Tuple[bool, str, int]:
        """
        Check if weekly trend is UP
        
        Elder's Weekly UP criteria (ANY of these):
        1. Weekly EMA-22 is rising (slope positive)
        2. Weekly MACD-Histogram is rising
        3. Price above weekly EMA-22
        
        Returns: (is_uptrend, reason, score)
        """
        # Resample to weekly
        weekly = hist_slice.resample('W').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min',
            'Close': 'last', 'Volume': 'sum'
        }).dropna()
        
        if len(weekly) < 10:
            return False, "Insufficient weekly data", 0
        
        closes = weekly['Close']
        score = 0
        reasons = []
        
        # 1. EMA-22 rising
        ema_22 = calculate_ema(closes, min(len(closes), 22))
        if len(ema_22) >= 2:
            ema_slope = ema_22.iloc[-1] - ema_22.iloc[-2]
            if ema_slope > 0:
                score += 1
                reasons.append("EMA-22 rising")
        
        # 2. MACD-H rising
        if len(closes) >= 26:
            macd = calculate_macd(closes)
            if len(macd['histogram']) >= 2:
                macd_h_slope = macd['histogram'].iloc[-1] - macd['histogram'].iloc[-2]
                if macd_h_slope > 0:
                    score += 1
                    if macd['histogram'].iloc[-2] < 0:
                        reasons.append("MACD-H Spring (rising from below 0)")
                        score += 1  # Bonus for Spring
                    else:
                        reasons.append("MACD-H rising")
        
        # 3. Price above EMA-22
        if len(ema_22) > 0:
            if closes.iloc[-1] > ema_22.iloc[-1]:
                score += 1
                reasons.append("Price > EMA-22")
        
        is_uptrend = score >= 1
        reason = " + ".join(reasons) if reasons else "No uptrend signals"
        
        return is_uptrend, reason, score
    
    def check_daily_pullback(self, hist_slice: pd.DataFrame) -> Tuple[bool, str, int]:
        """
        Check if daily is in pullback (DOWN in uptrend = BUY opportunity)
        
        Elder's Daily DOWN criteria (ANY of these):
        1. Force Index 2-EMA < 0
        2. Stochastic < 50
        3. Price at or below daily EMA-22 (buying value)
        4. RSI < 50
        
        Returns: (is_pullback, reason, score)
        """
        if len(hist_slice) < 30:
            return False, "Insufficient daily data", 0
        
        # Calculate indicators
        indicators = calculate_all_indicators(
            hist_slice['High'],
            hist_slice['Low'], 
            hist_slice['Close'],
            hist_slice['Volume']
        )
        
        score = 0
        reasons = []
        
        # 1. Force Index < 0
        force_index = indicators.get('force_index_2', 0)
        if force_index < 0:
            score += 2  # Strong signal per Elder
            reasons.append(f"Force Index < 0 ({force_index:.0f})")
        
        # 2. Stochastic < 50
        stochastic = indicators.get('stochastic_k', 50)
        if stochastic < 50:
            score += 1
            if stochastic < 30:
                score += 1  # Bonus for oversold
                reasons.append(f"Stochastic oversold ({stochastic:.1f})")
            else:
                reasons.append(f"Stochastic < 50 ({stochastic:.1f})")
        
        # 3. Price at/below EMA
        price = hist_slice['Close'].iloc[-1]
        ema_22 = indicators.get('ema_22', price)
        if price <= ema_22:
            score += 1
            reasons.append("Price at/below EMA-22 (value zone)")
        
        # 4. RSI < 50
        rsi = indicators.get('rsi', 50)
        if rsi < 50:
            score += 1
            if rsi < 30:
                score += 1
                reasons.append(f"RSI oversold ({rsi:.1f})")
            else:
                reasons.append(f"RSI < 50 ({rsi:.1f})")
        
        # 5. Bullish candlestick pattern
        patterns = scan_patterns(hist_slice.tail(5))
        bullish = get_bullish_patterns(patterns)
        if bullish:
            score += 1
            reasons.append(f"Bullish pattern: {bullish[0].get('name', 'detected')}")
        
        is_pullback = score >= 2
        reason = " + ".join(reasons) if reasons else "No pullback signals"
        
        return is_pullback, reason, score
    
    def calculate_trade_levels(
        self,
        hist_slice: pd.DataFrame,
        entry_bar_idx: int
    ) -> Dict:
        """Calculate Entry/Stop/Target levels"""
        price = hist_slice['Close'].iloc[-1]
        prev_high = hist_slice['High'].iloc[-1]
        
        # Entry: Buy-stop above previous high
        entry = round(float(prev_high * 1.002), 2)  # 0.2% buffer
        
        # Stop: Below recent swing low
        recent_lows = hist_slice['Low'].tail(5)
        swing_low = recent_lows.min()
        
        # ATR-based stop as alternative
        atr = calculate_atr(
            hist_slice['High'],
            hist_slice['Low'],
            hist_slice['Close']
        ).iloc[-1]
        
        atr_stop = entry - (2 * atr)
        
        # Use the tighter stop but ensure minimum 1% risk
        stop = max(float(swing_low * 0.998), float(atr_stop))
        min_stop = entry * 0.99  # At least 1% risk
        if stop > min_stop:
            stop = min_stop
        
        stop = round(stop, 2)
        
        # Risk calculation
        risk = entry - stop
        if risk <= 0:
            risk = entry * 0.02
            stop = round(entry - risk, 2)
        
        # Target based on R:R
        target = round(float(entry + (risk * self.rr_target)), 2)
        
        return {
            'entry': entry,
            'stop': stop,
            'target': target,
            'risk': round(risk, 2),
            'reward': round(target - entry, 2),
            'rr_ratio': round((target - entry) / risk, 2) if risk > 0 else 0,
            'atr': round(float(atr), 2)
        }
    
    def simulate_trade(
        self,
        trade: BacktestTrade,
        future_bars: pd.DataFrame,
        max_days: int = 20
    ) -> BacktestTrade:
        """Simulate trade execution"""
        entry_triggered = False
        days_waiting = 0
        
        for i, (date, bar) in enumerate(future_bars.iterrows()):
            if i >= max_days and not entry_triggered:
                trade.status = 'expired'
                trade.exit_reason = f'Entry not triggered in {max_days} days'
                return trade
            
            # Check entry trigger
            if not entry_triggered:
                days_waiting += 1
                
                # Entry triggers when high >= entry price
                if bar['High'] >= trade.entry_price:
                    entry_triggered = True
                    trade.entry_date = date.strftime('%Y-%m-%d')
                    trade.status = 'open'
                    continue
                
                # Check if setup invalidated
                if bar['Low'] < trade.stop_loss:
                    trade.status = 'cancelled'
                    trade.exit_reason = 'Setup invalidated (hit stop before entry)'
                    return trade
            
            # If entry triggered, check exits
            if entry_triggered:
                trade.days_held += 1
                
                # Check stop loss FIRST (conservative)
                if bar['Low'] <= trade.stop_loss:
                    trade.exit_date = date.strftime('%Y-%m-%d')
                    trade.exit_price = trade.stop_loss
                    trade.pnl = round(-trade.risk_per_share * trade.quantity, 2)
                    trade.pnl_percent = round(-100 * trade.risk_per_share / trade.entry_price, 2)
                    trade.status = 'loss'
                    trade.exit_reason = 'Stop loss hit'
                    return trade
                
                # Check target
                if bar['High'] >= trade.target:
                    trade.exit_date = date.strftime('%Y-%m-%d')
                    trade.exit_price = trade.target
                    trade.pnl = round(trade.reward_per_share * trade.quantity, 2)
                    trade.pnl_percent = round(100 * trade.reward_per_share / trade.entry_price, 2)
                    trade.status = 'win'
                    trade.exit_reason = 'Target hit'
                    return trade
        
        # End of data
        if entry_triggered:
            last_close = future_bars['Close'].iloc[-1]
            trade.exit_date = future_bars.index[-1].strftime('%Y-%m-%d')
            trade.exit_price = round(float(last_close), 2)
            trade.pnl = round((last_close - trade.entry_price) * trade.quantity, 2)
            trade.pnl_percent = round(100 * (last_close - trade.entry_price) / trade.entry_price, 2)
            trade.status = 'open'
            trade.exit_reason = 'End of backtest period'
        else:
            trade.status = 'no_entry'
            trade.exit_reason = 'Entry never triggered'
        
        return trade
    
    def run(self, min_score: int = 3) -> Optional[BacktestResult]:
        """
        Run the backtest
        
        Args:
            min_score: Minimum combined score (weekly + daily) to take trade
        """
        hist = self.fetch_historical_data()
        if hist is None or len(hist) < 100:
            print(f"❌ {self.symbol}: Insufficient data")
            return None
        
        print(f"📊 {self.symbol}: Running backtest on {len(hist)} bars...")
        
        warmup_bars = 100
        risk_amount = self.initial_capital * (self.risk_per_trade_pct / 100)
        
        active_trade = None
        
        for i in range(warmup_bars, len(hist) - 1):
            current_date = hist.index[i]
            
            # Skip if active trade
            if active_trade and active_trade.status == 'open':
                continue
            
            # Reset active trade if completed
            if active_trade and active_trade.status in ['win', 'loss', 'cancelled', 'expired', 'no_entry']:
                active_trade = None
            
            # Get history slice
            hist_slice = hist.iloc[:i+1]
            
            # Check weekly uptrend
            weekly_up, weekly_reason, weekly_score = self.check_weekly_uptrend(hist_slice)
            
            if not weekly_up:
                continue
            
            # Check daily pullback
            daily_down, daily_reason, daily_score = self.check_daily_pullback(hist_slice)
            
            if not daily_down:
                continue
            
            # Combined score
            total_score = weekly_score + daily_score
            
            if total_score < min_score:
                continue
            
            self.total_signals += 1
            
            # Calculate trade levels
            levels = self.calculate_trade_levels(hist_slice, i)
            
            if levels['risk'] <= 0:
                continue
            
            # Position sizing
            quantity = int(risk_amount / levels['risk'])
            if quantity <= 0:
                continue
            
            # Determine grade
            if total_score >= 6:
                grade = 'A'
            elif total_score >= 4:
                grade = 'B'
            else:
                grade = 'C'
            
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
                signal_strength=total_score,
                grade=grade,
                weekly_trend=weekly_reason,
                daily_pullback=daily_reason
            )
            
            # Simulate
            future_bars = hist.iloc[i+1:]
            trade = self.simulate_trade(trade, future_bars)
            
            # Record trade
            if trade.status not in ['no_entry', 'expired']:
                self.trades.append(trade)
                self.current_capital += trade.pnl
                self.equity_curve.append({
                    'date': trade.exit_date or trade.signal_date,
                    'equity': round(self.current_capital, 2),
                    'trade_pnl': trade.pnl
                })
            
            if trade.status == 'open':
                active_trade = trade
        
        return self._calculate_results(hist)
    
    def _calculate_results(self, hist: pd.DataFrame) -> BacktestResult:
        """Calculate final statistics"""
        closed = [t for t in self.trades if t.status in ['win', 'loss']]
        wins = [t for t in closed if t.status == 'win']
        losses = [t for t in closed if t.status == 'loss']
        cancelled = [t for t in self.trades if t.status == 'cancelled']
        open_trades = [t for t in self.trades if t.status == 'open']
        
        total_wins = sum(t.pnl for t in wins)
        total_losses = sum(t.pnl for t in losses)
        total_pnl = total_wins + total_losses
        
        win_count = len(wins)
        loss_count = len(losses)
        
        avg_win = total_wins / win_count if win_count > 0 else 0
        avg_loss = abs(total_losses / loss_count) if loss_count > 0 else 0
        win_rate = (win_count / len(closed) * 100) if closed else 0
        profit_factor = abs(total_wins / total_losses) if total_losses != 0 else (999 if total_wins > 0 else 0)
        expectancy = (win_rate/100 * avg_win) - ((100-win_rate)/100 * avg_loss)
        
        max_wins, max_losses = self._calc_streaks(closed)
        max_dd = self._calc_max_drawdown()
        
        days_list = [t.days_held for t in closed if t.days_held > 0]
        avg_days = sum(days_list) / len(days_list) if days_list else 0
        
        return_pct = ((self.current_capital - self.initial_capital) / self.initial_capital) * 100
        
        return BacktestResult(
            symbol=self.symbol,
            market=self.market,
            period_days=self.lookback_days,
            start_date=hist.index[0].strftime('%Y-%m-%d'),
            end_date=hist.index[-1].strftime('%Y-%m-%d'),
            data_bars=len(hist),
            total_signals=self.total_signals,
            total_trades=len(self.trades),
            winning_trades=win_count,
            losing_trades=loss_count,
            cancelled_trades=len(cancelled),
            open_trades=len(open_trades),
            win_rate=round(win_rate, 2),
            total_pnl=round(total_pnl, 2),
            total_pnl_percent=round(return_pct, 2),
            avg_win=round(avg_win, 2),
            avg_loss=round(avg_loss, 2),
            profit_factor=round(profit_factor, 2),
            expectancy=round(expectancy, 2),
            max_consecutive_wins=max_wins,
            max_consecutive_losses=max_losses,
            max_drawdown=round(max_dd, 2),
            avg_days_held=round(avg_days, 1),
            initial_capital=self.initial_capital,
            final_capital=round(self.current_capital, 2),
            return_percent=round(return_pct, 2),
            trades=[asdict(t) for t in self.trades],
            equity_curve=self.equity_curve
        )
    
    def _calc_streaks(self, closed: List[BacktestTrade]) -> Tuple[int, int]:
        max_w, max_l, cur_w, cur_l = 0, 0, 0, 0
        for t in closed:
            if t.status == 'win':
                cur_w += 1
                cur_l = 0
                max_w = max(max_w, cur_w)
            else:
                cur_l += 1
                cur_w = 0
                max_l = max(max_l, cur_l)
        return max_w, max_l
    
    def _calc_max_drawdown(self) -> float:
        if not self.equity_curve:
            return 0.0
        peak = self.initial_capital
        max_dd = 0.0
        for pt in self.equity_curve:
            eq = pt['equity']
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100
            max_dd = max(max_dd, dd)
        return max_dd


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def run_backtest(
    symbol: str,
    market: str = 'US',
    lookback_days: int = 365,
    initial_capital: float = 100000,
    risk_per_trade_pct: float = 2.0,
    rr_target: float = 1.5,
    min_score: int = 3
) -> Optional[Dict]:
    """
    Run backtest for a single symbol
    
    Args:
        symbol: Stock symbol
        market: 'US' or 'INDIA'
        lookback_days: Days of history (default 365)
        initial_capital: Starting capital
        risk_per_trade_pct: Risk per trade % (default 2%)
        rr_target: Risk:Reward ratio target (default 1.5)
        min_score: Minimum score to take trade (default 3)
    """
    engine = PracticalBacktestEngine(
        symbol=symbol,
        market=market,
        lookback_days=lookback_days,
        initial_capital=initial_capital,
        risk_per_trade_pct=risk_per_trade_pct,
        rr_target=rr_target
    )
    
    result = engine.run(min_score=min_score)
    return asdict(result) if result else None


def run_portfolio_backtest(
    symbols: List[str],
    market: str = 'US',
    lookback_days: int = 365,
    initial_capital: float = 100000,
    risk_per_trade_pct: float = 2.0,
    rr_target: float = 1.5,
    min_score: int = 3
) -> Dict:
    """Run backtest across multiple symbols"""
    results = []
    all_trades = []
    
    for symbol in symbols:
        result = run_backtest(
            symbol=symbol,
            market=market,
            lookback_days=lookback_days,
            initial_capital=initial_capital,
            risk_per_trade_pct=risk_per_trade_pct,
            rr_target=rr_target,
            min_score=min_score
        )
        
        if result:
            results.append(result)
            all_trades.extend(result.get('trades', []))
    
    # Aggregate
    total_trades = sum(r['total_trades'] for r in results)
    winning = sum(r['winning_trades'] for r in results)
    losing = sum(r['losing_trades'] for r in results)
    total_pnl = sum(r['total_pnl'] for r in results)
    
    return {
        'summary': {
            'symbols_tested': len(symbols),
            'symbols_with_trades': len([r for r in results if r['total_trades'] > 0]),
            'total_signals': sum(r['total_signals'] for r in results),
            'total_trades': total_trades,
            'winning_trades': winning,
            'losing_trades': losing,
            'win_rate': round(winning / total_trades * 100, 2) if total_trades > 0 else 0,
            'total_pnl': round(total_pnl, 2),
            'avg_pnl_per_symbol': round(total_pnl / len(results), 2) if results else 0
        },
        'individual_results': results,
        'all_trades': sorted(all_trades, key=lambda x: x.get('signal_date', ''))
    }
