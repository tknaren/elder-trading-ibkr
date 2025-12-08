"""
Elder Trading System - Database Models
SQLite database operations and schema management
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import Optional, List, Dict


class Database:
    """Database connection manager"""

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.environ.get('DATABASE_PATH', '')
            if not db_path:
                if os.path.exists('/home'):
                    db_path = '/home/data/elder_trading.db'
                else:
                    db_path = 'elder_trading.db'

        self.db_path = db_path
        self._ensure_directory()
        self._init_db()

    def _ensure_directory(self):
        """Ensure database directory exists"""
        try:
            db_dir = os.path.dirname(self.db_path)
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)
        except Exception as e:
            print(f"Warning: Could not create db directory: {e}")

    def get_connection(self):
        """Get database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initialize database schema"""
        conn = self.get_connection()
        conn.executescript('''
            -- Users table
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- Account settings
            CREATE TABLE IF NOT EXISTS account_settings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                account_name TEXT NOT NULL,
                market TEXT NOT NULL,
                trading_capital REAL NOT NULL,
                risk_per_trade REAL DEFAULT 2.0,
                max_monthly_drawdown REAL DEFAULT 6.0,
                target_rr REAL DEFAULT 2.0,
                max_open_positions INTEGER DEFAULT 5,
                currency TEXT DEFAULT 'USD',
                broker TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
            
            -- Strategies
            CREATE TABLE IF NOT EXISTS strategies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                is_active BOOLEAN DEFAULT 1,
                config JSON NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
            
            -- APGAR parameters
            CREATE TABLE IF NOT EXISTS apgar_parameters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_id INTEGER NOT NULL,
                parameter_name TEXT NOT NULL,
                parameter_label TEXT NOT NULL,
                options JSON NOT NULL,
                display_order INTEGER DEFAULT 0,
                FOREIGN KEY (strategy_id) REFERENCES strategies(id)
            );
            
            -- Weekly scans
            CREATE TABLE IF NOT EXISTS weekly_scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                market TEXT NOT NULL,
                scan_date DATE NOT NULL,
                week_start DATE NOT NULL,
                week_end DATE NOT NULL,
                results JSON NOT NULL,
                summary JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
            
            -- Daily scans
            CREATE TABLE IF NOT EXISTS daily_scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                weekly_scan_id INTEGER NOT NULL,
                market TEXT NOT NULL,
                scan_date DATE NOT NULL,
                results JSON NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id),
                FOREIGN KEY (weekly_scan_id) REFERENCES weekly_scans(id)
            );
            
            -- Trade setups
            CREATE TABLE IF NOT EXISTS trade_setups (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                daily_scan_id INTEGER,
                symbol TEXT NOT NULL,
                market TEXT NOT NULL,
                strategy_id INTEGER,
                apgar_score INTEGER,
                apgar_details JSON,
                entry_price REAL,
                stop_loss REAL,
                target_price REAL,
                position_size INTEGER,
                risk_amount REAL,
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
            
            -- Trade journal
            CREATE TABLE IF NOT EXISTS trade_journal (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                trade_setup_id INTEGER,
                symbol TEXT NOT NULL,
                market TEXT NOT NULL,
                direction TEXT DEFAULT 'LONG',
                entry_date DATE,
                entry_price REAL,
                exit_date DATE,
                exit_price REAL,
                position_size INTEGER,
                stop_loss REAL,
                target_price REAL,
                pnl REAL,
                pnl_percent REAL,
                fees REAL DEFAULT 0,
                strategy_id INTEGER,
                apgar_score INTEGER,
                notes TEXT,
                lessons_learned TEXT,
                grade TEXT,
                status TEXT DEFAULT 'open',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
            
            -- Daily checklist
            CREATE TABLE IF NOT EXISTS daily_checklist (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                checklist_date DATE NOT NULL,
                items JSON NOT NULL,
                completed_at TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id),
                UNIQUE(user_id, checklist_date)
            );
            
            -- Watchlists
            CREATE TABLE IF NOT EXISTS watchlists (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                market TEXT NOT NULL,
                symbols JSON NOT NULL,
                is_default BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
            
            -- Trade Bills
            CREATE TABLE IF NOT EXISTS trade_bills (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                ticker TEXT NOT NULL,
                current_market_price REAL,
                entry_price REAL,
                stop_loss REAL,
                target_price REAL,
                quantity REAL,
                upper_channel REAL,
                lower_channel REAL,
                target_pips REAL,
                stop_loss_pips REAL,
                max_qty_for_risk REAL,
                overnight_charges REAL DEFAULT 0,
                risk_per_share REAL,
                position_size REAL,
                risk_percent REAL,
                channel_height REAL,
                potential_gain REAL,
                target_1_1_c REAL,
                target_1_2_b REAL,
                target_1_3_a REAL,
                risk_amount_currency REAL,
                reward_amount_currency REAL,
                risk_reward_ratio REAL,
                break_even REAL,
                trailing_stop REAL,
                is_filled BOOLEAN DEFAULT 0,
                stop_entered BOOLEAN DEFAULT 0,
                target_entered BOOLEAN DEFAULT 0,
                journal_entered BOOLEAN DEFAULT 0,
                comments TEXT,
                status TEXT DEFAULT 'active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
            
            -- Trade Log (detailed journal matching Excel)
            CREATE TABLE IF NOT EXISTS trade_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                entry_date TEXT,
                symbol TEXT NOT NULL,
                strategy TEXT,
                direction TEXT DEFAULT 'Long',
                entry_price REAL,
                shares INTEGER,
                stop_loss REAL,
                take_profit REAL,
                exit_date TEXT,
                exit_price REAL,
                trade_costs REAL DEFAULT 0,
                gross_pnl REAL,
                net_pnl REAL,
                planned_rrr REAL,
                actual_rrr REAL,
                r_value REAL,
                account_change_pct REAL,
                mistake TEXT,
                discipline_rating INTEGER DEFAULT 8,
                notes TEXT,
                status TEXT DEFAULT 'open',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
            
            -- Indicator Filters Config
            CREATE TABLE IF NOT EXISTS indicator_filters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL UNIQUE,
                config JSON NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );

            -- Historical OHLCV data (2-year cache)
            CREATE TABLE IF NOT EXISTS stock_historical_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, date)
            );
            CREATE INDEX IF NOT EXISTS idx_symbol_date ON stock_historical_data(symbol, date);

            -- Track last update for each stock
            CREATE TABLE IF NOT EXISTS stock_data_sync (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL UNIQUE,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                earliest_date TEXT,
                latest_date TEXT,
                record_count INTEGER DEFAULT 0
            );
        ''')
        conn.commit()
        conn.close()

        # Initialize default data
        self._init_defaults()

    def _init_defaults(self):
        """Initialize default user, strategies, and watchlists"""
        conn = self.get_connection()

        # Check if defaults exist
        cursor = conn.execute('SELECT COUNT(*) FROM users')
        if cursor.fetchone()[0] == 0:
            from werkzeug.security import generate_password_hash

            # Create default user
            conn.execute('''
                INSERT INTO users (username, password_hash)
                VALUES (?, ?)
            ''', ('default', generate_password_hash('elder2024')))

            user_id = conn.execute('SELECT last_insert_rowid()').fetchone()[0]

            # Create default strategy
            elder_config = {
                "name": "Elder Triple Screen",
                "timeframes": {"screen1": "weekly", "screen2": "daily"},
                "indicators": {
                    "weekly": ["EMA_22", "MACD_Histogram"],
                    "daily": ["Force_Index_2", "Stochastic_14", "EMA_22", "Impulse"]
                }
            }

            conn.execute('''
                INSERT INTO strategies (user_id, name, description, config)
                VALUES (?, ?, ?, ?)
            ''', (user_id, 'Elder Triple Screen',
                  "Dr. Alexander Elder's Triple Screen Trading System",
                  json.dumps(elder_config)))

            strategy_id = conn.execute(
                'SELECT last_insert_rowid()').fetchone()[0]

            # APGAR parameters
            apgar_params = [
                ('weekly_ema', 'Weekly EMA (22) Slope', [
                    {"score": 2, "label": "Strongly Rising"},
                    {"score": 1, "label": "Rising"},
                    {"score": 0, "label": "Flat/Falling"}
                ]),
                ('weekly_macd', 'Weekly MACD-Histogram', [
                    {"score": 2, "label": "Rising + Divergence"},
                    {"score": 1, "label": "Rising"},
                    {"score": 0, "label": "Falling"}
                ]),
                ('force_index', 'Daily Force Index (2-EMA)', [
                    {"score": 2, "label": "Below Zero + Uptick"},
                    {"score": 1, "label": "Below Zero"},
                    {"score": 0, "label": "Above Zero"}
                ]),
                ('stochastic', 'Daily Stochastic', [
                    {"score": 2, "label": "Below 30 (Oversold)"},
                    {"score": 1, "label": "30-50"},
                    {"score": 0, "label": "Above 50"}
                ]),
                ('price_ema', 'Price vs 22-Day EMA', [
                    {"score": 2, "label": "At or Below EMA"},
                    {"score": 1, "label": "Slightly Above (<2%)"},
                    {"score": 0, "label": "Far Above"}
                ])
            ]

            for i, (name, label, options) in enumerate(apgar_params):
                conn.execute('''
                    INSERT INTO apgar_parameters 
                    (strategy_id, parameter_name, parameter_label, options, display_order)
                    VALUES (?, ?, ?, ?, ?)
                ''', (strategy_id, name, label, json.dumps(options), i))

            # Default watchlists
            nasdaq_100 = [
                # Top 50 by market cap
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AMD', 'AVGO', 'NFLX',
                'COST', 'PEP', 'ADBE', 'CSCO', 'INTC', 'QCOM', 'TXN', 'INTU', 'AMAT', 'MU',
                'LRCX', 'KLAC', 'SNPS', 'CDNS', 'MRVL', 'ON', 'NXPI', 'ADI', 'MCHP', 'FTNT',
                'VRTX', 'CHTR', 'ASML', 'CRWD', 'PANW', 'MNST', 'TEAM', 'PAYX', 'AEP', 'REGN',
                'DXCM', 'CPRT', 'PCAR', 'ALGN', 'AMGN', 'MRNA', 'XEL', 'WDAY', 'ABNB', 'MDLZ',
                # Next 50
                'GILD', 'ISRG', 'BKNG', 'ADP', 'SBUX', 'PYPL', 'CME', 'ORLY', 'IDXX', 'CTAS',
                'MAR', 'CSX', 'ODFL', 'FAST', 'ROST', 'KDP', 'EXC', 'DLTR', 'BIIB', 'EA',
                'VRSK', 'ANSS', 'ILMN', 'SIRI', 'ZS', 'DDOG', 'CTSH', 'WBD', 'EBAY', 'FANG',
                'GFS', 'LCID', 'RIVN', 'CEG', 'TTD', 'GEHC', 'ZM', 'ROKU', 'OKTA', 'SPLK',
                'DOCU', 'BILL', 'ENPH', 'SEDG', 'DASH', 'COIN', 'HOOD', 'SOFI', 'PLTR', 'NET'
            ]

            nifty_50 = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS',
                        'ICICIBANK.NS', 'HINDUNILVR.NS', 'SBIN.NS', 'BHARTIARTL.NS',
                        'ITC.NS', 'KOTAKBANK.NS', 'LT.NS', 'AXISBANK.NS']

            conn.execute('''
                INSERT INTO watchlists (user_id, name, market, symbols, is_default)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, 'NASDAQ 100', 'US', json.dumps(nasdaq_100), 1))

            conn.execute('''
                INSERT INTO watchlists (user_id, name, market, symbols, is_default)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, 'NIFTY 50', 'IN', json.dumps(nifty_50), 1))

            # Default account settings
            conn.execute('''
                INSERT INTO account_settings 
                (user_id, account_name, market, trading_capital, currency, broker)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (user_id, 'ISA Account', 'US', 6000, 'GBP', 'Trading212'))

            conn.execute('''
                INSERT INTO account_settings 
                (user_id, account_name, market, trading_capital, currency, broker)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (user_id, 'Zerodha Account', 'IN', 570749, 'INR', 'Zerodha'))

            conn.commit()

        conn.close()

    # ========== TRADE BILLS METHODS ==========
    def create_trade_bill(self, user_id: int, data: Dict) -> int:
        """Create a new trade bill"""
        conn = self.get_connection()
        cursor = conn.cursor()

        # Auto-calculate derived fields
        if 'entry_price' in data and 'stop_loss' in data and 'quantity' in data:
            risk_per_share = (data['entry_price'] -
                              data['stop_loss']) * data['quantity']
            data['risk_per_share'] = abs(
                data['entry_price'] - data['stop_loss'])
            data['risk_amount_currency'] = risk_per_share

        if 'target_price' in data and 'entry_price' in data and 'quantity' in data:
            potential_gain = (data['target_price'] -
                              data['entry_price']) * data['quantity']
            data['potential_gain'] = potential_gain

        if 'risk_amount_currency' in data and 'reward_amount_currency' in data:
            if data['risk_amount_currency'] > 0:
                data['risk_reward_ratio'] = data['reward_amount_currency'] / \
                    data['risk_amount_currency']

        # Build insert statement dynamically
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['?' for _ in data])
        values = tuple(data.values())

        cursor.execute(f'''
            INSERT INTO trade_bills (user_id, {columns})
            VALUES (?, {placeholders})
        ''', (user_id, *values))

        conn.commit()
        trade_bill_id = cursor.lastrowid
        conn.close()
        return trade_bill_id

    def get_trade_bill(self, trade_bill_id: int) -> Optional[Dict]:
        """Get a specific trade bill"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM trade_bills WHERE id = ?',
                       (trade_bill_id,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

    def get_trade_bills(self, user_id: int, status: str = None) -> List[Dict]:
        """Get all trade bills for a user"""
        conn = self.get_connection()
        cursor = conn.cursor()

        if status:
            cursor.execute('''
                SELECT * FROM trade_bills 
                WHERE user_id = ? AND status = ?
                ORDER BY created_at DESC
            ''', (user_id, status))
        else:
            cursor.execute('''
                SELECT * FROM trade_bills 
                WHERE user_id = ?
                ORDER BY created_at DESC
            ''', (user_id,))

        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def update_trade_bill(self, trade_bill_id: int, data: Dict) -> bool:
        """Update a trade bill"""
        conn = self.get_connection()
        cursor = conn.cursor()

        data['updated_at'] = datetime.now().isoformat()

        set_clause = ', '.join([f'{k} = ?' for k in data.keys()])
        values = tuple(data.values())

        cursor.execute(f'''
            UPDATE trade_bills
            SET {set_clause}
            WHERE id = ?
        ''', (*values, trade_bill_id))

        conn.commit()
        success = cursor.rowcount > 0
        conn.close()
        return success

    def delete_trade_bill(self, trade_bill_id: int) -> bool:
        """Delete a trade bill"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM trade_bills WHERE id = ?',
                       (trade_bill_id,))
        conn.commit()
        success = cursor.rowcount > 0
        conn.close()
        return success

    def calculate_trade_metrics(self, entry_price: float, stop_loss: float,
                                target_price: float, quantity: float,
                                account_capital: float, risk_percent: float) -> Dict:
        """Calculate trade metrics and position sizing"""
        metrics = {}

        # Risk and Stop Loss
        metrics['risk_per_share'] = abs(entry_price - stop_loss)
        metrics['stop_loss_pips'] = metrics['risk_per_share']

        # Target and Reward
        metrics['target_pips'] = abs(target_price - entry_price)
        metrics['potential_gain'] = metrics['target_pips'] * quantity

        # Risk Reward Ratio
        if metrics['risk_per_share'] > 0:
            metrics['risk_reward_ratio'] = metrics['target_pips'] / \
                metrics['risk_per_share']

        # Position Sizing
        max_risk_amount = (account_capital * risk_percent) / 100
        metrics['max_qty_for_risk'] = max_risk_amount / \
            metrics['risk_per_share'] if metrics['risk_per_share'] > 0 else 0
        metrics['position_size'] = quantity
        metrics['risk_amount_currency'] = quantity * metrics['risk_per_share']

        # Break Even
        metrics['break_even'] = entry_price

        return metrics


# Singleton instance
_db_instance = None


def get_database() -> Database:
    """Get database singleton instance"""
    global _db_instance
    if _db_instance is None:
        _db_instance = Database()
    return _db_instance
