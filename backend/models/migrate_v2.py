"""
Elder Trading System - Database Migration v2
Adds new columns for connected workflow: Screener → Trade Bill → IBKR → Trade Log → Positions
"""

import sqlite3
import os


def migrate_database(db_path: str = None):
    """Run database migrations to add new columns and tables"""
    
    if db_path is None:
        db_path = os.environ.get('DATABASE_PATH', '')
        if not db_path:
            if os.path.exists('/home'):
                db_path = '/home/data/elder_trading.db'
            else:
                db_path = 'elder_trading.db'
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    migrations = []
    
    # Migration 1: Add order_id and related columns to trade_bills
    migrations.append("""
        ALTER TABLE trade_bills ADD COLUMN order_id TEXT;
    """)
    
    migrations.append("""
        ALTER TABLE trade_bills ADD COLUMN signal_strength INTEGER;
    """)
    
    migrations.append("""
        ALTER TABLE trade_bills ADD COLUMN grade TEXT;
    """)
    
    migrations.append("""
        ALTER TABLE trade_bills ADD COLUMN symbol TEXT;
    """)
    
    migrations.append("""
        ALTER TABLE trade_bills ADD COLUMN market TEXT DEFAULT 'US';
    """)
    
    migrations.append("""
        ALTER TABLE trade_bills ADD COLUMN direction TEXT DEFAULT 'LONG';
    """)
    
    migrations.append("""
        ALTER TABLE trade_bills ADD COLUMN risk_amount REAL;
    """)
    
    migrations.append("""
        ALTER TABLE trade_bills ADD COLUMN position_value REAL;
    """)
    
    # Migration 2: Add IBKR-related columns to trade_log
    migrations.append("""
        ALTER TABLE trade_log ADD COLUMN ibkr_order_id TEXT;
    """)
    
    migrations.append("""
        ALTER TABLE trade_log ADD COLUMN ibkr_execution_id TEXT;
    """)
    
    migrations.append("""
        ALTER TABLE trade_log ADD COLUMN trade_bill_id INTEGER;
    """)
    
    migrations.append("""
        ALTER TABLE trade_log ADD COLUMN synced_from_ibkr BOOLEAN DEFAULT 0;
    """)
    
    # Migration 3: Create positions tracking table
    migrations.append("""
        CREATE TABLE IF NOT EXISTS positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            conid INTEGER,
            quantity INTEGER NOT NULL,
            avg_price REAL NOT NULL,
            current_price REAL,
            market_value REAL,
            unrealized_pnl REAL,
            realized_pnl REAL,
            pnl_percent REAL,
            stop_loss REAL,
            target REAL,
            trade_bill_id INTEGER,
            entry_date TEXT,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'open',
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (trade_bill_id) REFERENCES trade_bills(id)
        );
    """)
    
    # Migration 4: Create position alerts table
    migrations.append("""
        CREATE TABLE IF NOT EXISTS position_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            position_id INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            alert_type TEXT NOT NULL,
            severity TEXT NOT NULL,
            message TEXT NOT NULL,
            action_suggested TEXT,
            is_read BOOLEAN DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (position_id) REFERENCES positions(id)
        );
    """)
    
    # Migration 5: Create IBKR orders tracking table
    migrations.append("""
        CREATE TABLE IF NOT EXISTS ibkr_orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            order_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            order_type TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            price REAL,
            stop_price REAL,
            status TEXT DEFAULT 'pending',
            trade_bill_id INTEGER,
            filled_quantity INTEGER DEFAULT 0,
            filled_price REAL,
            commission REAL DEFAULT 0,
            submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            filled_at TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (trade_bill_id) REFERENCES trade_bills(id)
        );
    """)
    
    # Migration 6: Add screener version to scans
    migrations.append("""
        ALTER TABLE weekly_scans ADD COLUMN screener_version TEXT DEFAULT '1.0';
    """)
    
    migrations.append("""
        ALTER TABLE daily_scans ADD COLUMN screener_version TEXT DEFAULT '1.0';
    """)
    
    # Run migrations
    success = 0
    skipped = 0
    errors = []
    
    for migration in migrations:
        try:
            cursor.execute(migration)
            success += 1
        except sqlite3.OperationalError as e:
            error_msg = str(e)
            if 'duplicate column name' in error_msg.lower() or 'already exists' in error_msg.lower():
                skipped += 1
            else:
                errors.append(f"{migration[:50]}... -> {error_msg}")
    
    conn.commit()
    conn.close()
    
    print(f"Migration complete: {success} applied, {skipped} skipped, {len(errors)} errors")
    if errors:
        print("Errors:")
        for e in errors:
            print(f"  - {e}")
    
    return {
        'success': success,
        'skipped': skipped,
        'errors': errors
    }


if __name__ == '__main__':
    result = migrate_database()
    print(f"\nMigration result: {result}")
