"""
Elder Trading System - Screener Service
Implements the Triple Screen methodology for stock screening

NOTE: This module now uses the enhanced v2 functions with all validation fixes.

Features v2:
- Screen 1 as MANDATORY GATE (not just scoring)
- Impulse RED blocks trades entirely
- Correct daily_ready logic (AND/OR)
- New high-scoring rules (divergence, false breakout, kangaroo tail)
- Elder Entry/Stop/Target calculations

VALIDATION FIXES APPLIED:
1. Screen 1 must pass before any scoring occurs
2. Impulse RED disqualifies trade (not just penalty)
3. daily_ready = screen1_passed AND impulse_ok AND (force_index < 0 OR stochastic < 30)
4. is_a_trade includes weekly_bullish check
5. Stochastic oversold threshold: 30 (was 50)

NEW HIGH-SCORING RULES:
+3: MACD divergence (strongest signal per Elder)
+3: Price near lower channel (short-term oversold)
+3: False downside breakout
+3: Pullback to value (Weekly EMA↑, Daily EMA↑, price < fast EMA)
+2: Kangaroo tail pattern
+2: Force Index down spike

ENTRY/STOP/TARGET (Elder Method):
- Entry: Daily EMA-22 (buy at value)
- Target: Keltner Channel Upper Band
- Stop: Below deepest historical EMA-22 penetration

Data Source: IBKR Client Portal API
"""

# Import v2 functions as the default implementation
from services.screener_v2 import (
    analyze_weekly_trend,
    calculate_signal_strength_v2 as calculate_signal_strength,
    calculate_elder_trade_levels as calculate_trade_levels,
    scan_stock_v2 as scan_stock,
    run_weekly_screen_v2 as run_weekly_screen,
    run_daily_screen_v2 as run_daily_screen,
    detect_kangaroo_tail,
    detect_false_breakout,
    detect_force_index_spike,
    calculate_ema_penetration_history,
    NASDAQ_100_TOP,
    NIFTY_50
)

# Re-export for backward compatibility
__all__ = [
    'analyze_weekly_trend',
    'calculate_signal_strength',
    'calculate_trade_levels',
    'scan_stock',
    'run_weekly_screen',
    'run_daily_screen',
    'detect_kangaroo_tail',
    'detect_false_breakout',
    'detect_force_index_spike',
    'calculate_ema_penetration_history',
    'NASDAQ_100_TOP',
    'NIFTY_50'
]
