"""
Elder Trading System - Indicator Configuration
Configurable indicator selection with clear categories

INDICATOR CATEGORIES:
1. TREND - Identify market direction
2. MOMENTUM - Measure speed/strength of move
3. OSCILLATOR - Identify overbought/oversold
4. VOLUME - Confirm moves with volume
5. VOLATILITY - Measure price range for stops
6. IMPULSE - Combined trend+momentum (Elder's)

Each category has multiple indicators - you choose ONE from each.
"""

# ============================================================
# INDICATOR CATALOG - All available indicators by category
# ============================================================

INDICATOR_CATALOG = {
    "TREND": {
        "description": "Identifies the primary market direction",
        "usage": "Screen 1 - Weekly trend determination",
        "indicators": {
            "EMA_22": {
                "name": "22-Period EMA",
                "description": "Exponential Moving Average - Elder's preferred trend indicator",
                "params": {"period": 22},
                "interpretation": {
                    "bullish": "Price above EMA AND EMA slope rising",
                    "bearish": "Price below EMA AND EMA slope falling"
                },
                "recommended": True,
                "elder_original": True
            },
            "EMA_13": {
                "name": "13-Period EMA",
                "description": "Faster EMA for shorter-term trends",
                "params": {"period": 13},
                "interpretation": {
                    "bullish": "Price above EMA AND EMA slope rising",
                    "bearish": "Price below EMA AND EMA slope falling"
                },
                "recommended": False,
                "elder_original": True
            },
            "SMA_50": {
                "name": "50-Period SMA",
                "description": "Simple Moving Average - classic trend indicator",
                "params": {"period": 50},
                "interpretation": {
                    "bullish": "Price above SMA",
                    "bearish": "Price below SMA"
                },
                "recommended": False,
                "elder_original": False
            },
            "SMA_200": {
                "name": "200-Period SMA",
                "description": "Long-term trend - institutional favorite",
                "params": {"period": 200},
                "interpretation": {
                    "bullish": "Price above SMA (bull market)",
                    "bearish": "Price below SMA (bear market)"
                },
                "recommended": False,
                "elder_original": False
            }
        }
    },
    
    "MOMENTUM": {
        "description": "Measures the speed and strength of price movement",
        "usage": "Screen 1 - Confirm trend strength",
        "indicators": {
            "MACD_HISTOGRAM": {
                "name": "MACD Histogram",
                "description": "Elder's primary momentum indicator - histogram SLOPE matters most",
                "params": {"fast": 12, "slow": 26, "signal": 9},
                "interpretation": {
                    "bullish": "Histogram rising (slope positive)",
                    "bearish": "Histogram falling (slope negative)",
                    "powerful": "Divergence between price and histogram"
                },
                "recommended": True,
                "elder_original": True
            },
            "MACD_LINE": {
                "name": "MACD Line",
                "description": "Traditional MACD line crossovers",
                "params": {"fast": 12, "slow": 26, "signal": 9},
                "interpretation": {
                    "bullish": "MACD crosses above signal line",
                    "bearish": "MACD crosses below signal line"
                },
                "recommended": False,
                "elder_original": False
            },
            "ROC": {
                "name": "Rate of Change",
                "description": "Simple momentum - percentage change over period",
                "params": {"period": 12},
                "interpretation": {
                    "bullish": "ROC positive and rising",
                    "bearish": "ROC negative and falling"
                },
                "recommended": False,
                "elder_original": False
            }
        }
    },
    
    "OSCILLATOR": {
        "description": "Identifies overbought/oversold conditions",
        "usage": "Screen 2 - Entry timing",
        "indicators": {
            "FORCE_INDEX": {
                "name": "Force Index (2-EMA)",
                "description": "Elder's own indicator - combines price change with volume",
                "params": {"period": 2},
                "interpretation": {
                    "buy_signal": "Force Index < 0 in uptrend (pullback)",
                    "sell_signal": "Force Index > 0 in downtrend (rally)",
                    "exhaustion": "Extreme spike = possible reversal"
                },
                "recommended": True,
                "elder_original": True
            },
            "STOCHASTIC": {
                "name": "Stochastic Oscillator",
                "description": "Classic overbought/oversold indicator",
                "params": {"period": 14, "smooth_k": 3},
                "interpretation": {
                    "buy_signal": "Below 30 (oversold)",
                    "sell_signal": "Above 70 (overbought)",
                    "divergence": "Price vs Stochastic divergence"
                },
                "recommended": True,
                "elder_original": False
            },
            "RSI": {
                "name": "RSI (Relative Strength Index)",
                "description": "Momentum oscillator measuring speed of price changes",
                "params": {"period": 14},
                "interpretation": {
                    "buy_signal": "Below 30 (oversold)",
                    "sell_signal": "Above 70 (overbought)",
                    "divergence": "Price vs RSI divergence"
                },
                "recommended": False,
                "elder_original": False
            },
            "WILLIAMS_R": {
                "name": "Williams %R",
                "description": "Similar to Stochastic, inverted scale",
                "params": {"period": 14},
                "interpretation": {
                    "buy_signal": "Below -80 (oversold)",
                    "sell_signal": "Above -20 (overbought)"
                },
                "recommended": False,
                "elder_original": False
            },
            "CCI": {
                "name": "CCI (Commodity Channel Index)",
                "description": "Measures deviation from average price",
                "params": {"period": 20},
                "interpretation": {
                    "buy_signal": "Below -100 (oversold)",
                    "sell_signal": "Above +100 (overbought)"
                },
                "recommended": False,
                "elder_original": False
            }
        }
    },
    
    "VOLUME": {
        "description": "Confirms price moves with volume analysis",
        "usage": "Screen 2 - Validate breakouts and trends",
        "indicators": {
            "FORCE_INDEX_13": {
                "name": "Force Index (13-EMA)",
                "description": "Intermediate-term Force Index for trend confirmation",
                "params": {"period": 13},
                "interpretation": {
                    "bullish": "Rising Force Index confirms uptrend",
                    "bearish": "Falling Force Index confirms downtrend"
                },
                "recommended": True,
                "elder_original": True
            },
            "OBV": {
                "name": "On-Balance Volume",
                "description": "Cumulative volume based on price direction",
                "params": {},
                "interpretation": {
                    "bullish": "OBV rising with price",
                    "bearish": "OBV falling with price",
                    "divergence": "OBV diverges from price = warning"
                },
                "recommended": False,
                "elder_original": False
            },
            "VOLUME_SMA": {
                "name": "Volume vs Average",
                "description": "Compare current volume to average",
                "params": {"period": 20},
                "interpretation": {
                    "confirmation": "Volume > 150% average confirms move",
                    "weak": "Volume < 50% average = weak move"
                },
                "recommended": False,
                "elder_original": False
            }
        }
    },
    
    "VOLATILITY": {
        "description": "Measures price range for stop-loss placement",
        "usage": "Screen 3 - Position sizing and stops",
        "indicators": {
            "ATR": {
                "name": "ATR (Average True Range)",
                "description": "Elder's choice for stop-loss calculation",
                "params": {"period": 14},
                "interpretation": {
                    "stop_loss": "Place stop 2× ATR from entry",
                    "position_size": "Higher ATR = smaller position",
                    "breakout": "ATR expansion = volatility increasing"
                },
                "recommended": True,
                "elder_original": True
            },
            "BOLLINGER_WIDTH": {
                "name": "Bollinger Band Width",
                "description": "Measures band expansion/contraction",
                "params": {"period": 20, "std": 2},
                "interpretation": {
                    "squeeze": "Narrow bands = breakout coming",
                    "expansion": "Wide bands = trend in progress"
                },
                "recommended": False,
                "elder_original": False
            },
            "KELTNER_CHANNEL": {
                "name": "Keltner Channel",
                "description": "ATR-based channels around EMA",
                "params": {"ema_period": 20, "atr_period": 10, "multiplier": 2},
                "interpretation": {
                    "buy": "Price at lower channel in uptrend",
                    "sell": "Price at upper channel"
                },
                "recommended": False,
                "elder_original": False
            }
        }
    },
    
    "IMPULSE": {
        "description": "Combined trend + momentum system",
        "usage": "Trade permission - GREEN/BLUE/RED",
        "indicators": {
            "ELDER_IMPULSE": {
                "name": "Elder Impulse System",
                "description": "EMA slope + MACD-H slope combined",
                "params": {"ema_period": 13},
                "interpretation": {
                    "GREEN": "EMA rising + MACD-H rising = BUY allowed",
                    "RED": "EMA falling + MACD-H falling = NO buying",
                    "BLUE": "Mixed signals = Neutral, caution"
                },
                "recommended": True,
                "elder_original": True
            }
        }
    }
}


# ============================================================
# DEFAULT CONFIGURATION - Elder's Original Setup
# ============================================================

DEFAULT_INDICATOR_CONFIG = {
    "name": "Elder Triple Screen (Original)",
    "description": "Dr. Alexander Elder's original indicator combination",
    
    "screen1_weekly": {
        "trend": "EMA_22",
        "momentum": "MACD_HISTOGRAM"
    },
    
    "screen2_daily": {
        "oscillator_primary": "FORCE_INDEX",
        "oscillator_secondary": "STOCHASTIC",  # Optional confirmation
        "volume": "FORCE_INDEX_13",
        "impulse": "ELDER_IMPULSE"
    },
    
    "screen3_entry": {
        "volatility": "ATR",
        "trend_fast": "EMA_13"
    },
    
    "scoring_weights": {
        "weekly_trend": 2,
        "weekly_momentum": 1,
        "oscillator_oversold": 2,
        "oscillator_uptick": 1,
        "price_vs_ema": 1,
        "divergence": 2,
        "impulse_green": 1,
        "impulse_red_penalty": -2
    }
}


# ============================================================
# ALTERNATIVE CONFIGURATIONS
# ============================================================

ALTERNATIVE_CONFIGS = {
    "elder_conservative": {
        "name": "Elder Conservative",
        "description": "Stricter criteria - fewer but higher quality signals",
        "screen1_weekly": {
            "trend": "EMA_22",
            "momentum": "MACD_HISTOGRAM"
        },
        "screen2_daily": {
            "oscillator_primary": "FORCE_INDEX",
            "oscillator_secondary": "RSI",
            "volume": "FORCE_INDEX_13",
            "impulse": "ELDER_IMPULSE"
        },
        "screen3_entry": {
            "volatility": "ATR",
            "trend_fast": "EMA_13"
        },
        "scoring_weights": {
            "weekly_trend": 3,  # Higher weight on trend
            "weekly_momentum": 2,
            "oscillator_oversold": 2,
            "oscillator_uptick": 1,
            "price_vs_ema": 2,  # Must be at value
            "divergence": 3,  # Divergence more important
            "impulse_green": 2,
            "impulse_red_penalty": -5  # Stricter penalty
        }
    },
    
    "swing_trader": {
        "name": "Swing Trader",
        "description": "RSI + Stochastic focus for swing trades",
        "screen1_weekly": {
            "trend": "SMA_50",
            "momentum": "MACD_HISTOGRAM"
        },
        "screen2_daily": {
            "oscillator_primary": "RSI",
            "oscillator_secondary": "STOCHASTIC",
            "volume": "VOLUME_SMA",
            "impulse": "ELDER_IMPULSE"
        },
        "screen3_entry": {
            "volatility": "ATR",
            "trend_fast": "EMA_13"
        }
    },
    
    "volume_focused": {
        "name": "Volume Focused",
        "description": "Emphasizes volume confirmation",
        "screen1_weekly": {
            "trend": "EMA_22",
            "momentum": "MACD_HISTOGRAM"
        },
        "screen2_daily": {
            "oscillator_primary": "FORCE_INDEX",
            "oscillator_secondary": "FORCE_INDEX",
            "volume": "OBV",
            "impulse": "ELDER_IMPULSE"
        },
        "screen3_entry": {
            "volatility": "ATR",
            "trend_fast": "EMA_13"
        }
    }
}


def get_indicator_info(indicator_id: str) -> dict:
    """Get detailed info about a specific indicator"""
    for category, data in INDICATOR_CATALOG.items():
        if indicator_id in data["indicators"]:
            info = data["indicators"][indicator_id].copy()
            info["category"] = category
            info["id"] = indicator_id
            return info
    return None


def get_recommended_indicators() -> dict:
    """Get all recommended (Elder's original) indicators"""
    recommended = {}
    for category, data in INDICATOR_CATALOG.items():
        for ind_id, ind_data in data["indicators"].items():
            if ind_data.get("recommended"):
                recommended[ind_id] = {
                    "category": category,
                    **ind_data
                }
    return recommended


def get_indicators_by_category(category: str) -> dict:
    """Get all indicators in a category"""
    if category in INDICATOR_CATALOG:
        return INDICATOR_CATALOG[category]["indicators"]
    return {}


def validate_config(config: dict) -> tuple:
    """Validate an indicator configuration"""
    errors = []
    warnings = []
    
    required_sections = ["screen1_weekly", "screen2_daily", "screen3_entry"]
    for section in required_sections:
        if section not in config:
            errors.append(f"Missing required section: {section}")
    
    # Check indicators exist
    for section, indicators in config.items():
        if isinstance(indicators, dict):
            for role, ind_id in indicators.items():
                if not get_indicator_info(ind_id):
                    errors.append(f"Unknown indicator: {ind_id} in {section}.{role}")
    
    return errors, warnings


# ============================================================
# CONFIGURATION DISPLAY HELPERS
# ============================================================

def get_config_summary(config: dict) -> str:
    """Generate human-readable summary of indicator configuration"""
    lines = [
        "═" * 60,
        f"  INDICATOR CONFIGURATION: {config.get('name', 'Custom')}",
        "═" * 60,
        "",
        "SCREEN 1 - WEEKLY (Strategic Direction)",
        "─" * 40,
    ]
    
    s1 = config.get("screen1_weekly", {})
    for role, ind_id in s1.items():
        info = get_indicator_info(ind_id)
        if info:
            lines.append(f"  {role.upper()}: {info['name']}")
            lines.append(f"    → {info['description']}")
    
    lines.extend([
        "",
        "SCREEN 2 - DAILY (Entry Timing)",
        "─" * 40,
    ])
    
    s2 = config.get("screen2_daily", {})
    for role, ind_id in s2.items():
        info = get_indicator_info(ind_id)
        if info:
            lines.append(f"  {role.upper()}: {info['name']}")
    
    lines.extend([
        "",
        "SCREEN 3 - ENTRY (Execution)",
        "─" * 40,
    ])
    
    s3 = config.get("screen3_entry", {})
    for role, ind_id in s3.items():
        info = get_indicator_info(ind_id)
        if info:
            lines.append(f"  {role.upper()}: {info['name']}")
    
    lines.append("═" * 60)
    
    return "\n".join(lines)
