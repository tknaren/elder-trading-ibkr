"""
Elder Trading System - API Routes
Flask Blueprint with all REST API endpoints
"""

from flask import Blueprint, request, jsonify, g
from datetime import datetime, timedelta
import json

from models.database import get_database
from services.screener import run_weekly_screen, run_daily_screen, scan_stock
from services.indicators import get_grading_criteria
from services.indicator_config import (
    INDICATOR_CATALOG,
    DEFAULT_INDICATOR_CONFIG,
    ALTERNATIVE_CONFIGS,
    get_indicator_info,
    get_config_summary
)
from services.candlestick_patterns import CANDLESTICK_PATTERNS
from services.ibkr_client import check_connection, get_client

api = Blueprint('api', __name__, url_prefix='/api')


def sanitize_for_json(obj):
    """
    Recursively sanitize all string values to be JSON-safe.
    Handles apostrophes and other special characters.
    """
    if isinstance(obj, dict):
        return {key: sanitize_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, str):
        # Ensure the string is properly encoded for JSON
        # json.dumps will handle escaping, but let's ensure no control chars
        return obj.encode('utf-8', errors='replace').decode('utf-8')
    else:
        return obj


def get_db():
    """Get database connection for current request"""
    if 'db' not in g:
        g.db = get_database().get_connection()
    return g.db


def get_user_id():
    """Get current user ID (default to 1 for now)"""
    return getattr(g, 'user_id', 1)


# ============ HEALTH CHECK ============
@api.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '2.0.0'
    })


@api.route('/ibkr/status', methods=['GET'])
def ibkr_status():
    """Check IBKR Gateway connection status"""
    connected, message = check_connection()
    return jsonify({
        'connected': connected,
        'message': message,
        'gateway_url': 'https://localhost:5000',
        'instructions': 'Start Client Portal Gateway and login at https://localhost:5000' if not connected else None
    })


# ============ SCREENER ENDPOINTS ============
@api.route('/screener/weekly', methods=['POST'])
def weekly_screener():
    """
    Run weekly screener (Screen 1)

    Request body:
        market: 'US' or 'IN'
        watchlist_id: (optional) specific watchlist to scan

    Returns:
        Complete scan results with all stocks and indicators
    """
    data = request.get_json() or {}
    market = data.get('market', 'US')
    watchlist_id = data.get('watchlist_id')

    db = get_db()
    user_id = get_user_id()

    # Get symbols from watchlist
    symbols = None
    if watchlist_id:
        watchlist = db.execute(
            'SELECT symbols FROM watchlists WHERE id = ?',
            (watchlist_id,)
        ).fetchone()
        if watchlist:
            symbols = json.loads(watchlist['symbols'])

    # If no specific watchlist requested and no symbols provided, use full market list (not default watchlist)
    # This ensures we always scan the full NASDAQ_100 or NIFTY_100
    # symbols will be None, and run_weekly_screen will use the full list

    # Run the screener
    results = run_weekly_screen(market, symbols)

    # Calculate week boundaries
    today = datetime.now().date()
    week_start = today - timedelta(days=today.weekday())
    week_end = week_start + timedelta(days=6)

    # Save scan to database
    db.execute('''
        INSERT INTO weekly_scans 
        (user_id, market, scan_date, week_start, week_end, results, summary)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        user_id, market, today, week_start, week_end,
        json.dumps(results['all_results']),
        json.dumps(results['summary'])
    ))
    db.commit()

    scan_id = db.execute('SELECT last_insert_rowid()').fetchone()[0]
    results['scan_id'] = scan_id
    results['week_start'] = week_start.isoformat()
    results['week_end'] = week_end.isoformat()

    # Sanitize results for JSON serialization
    results = sanitize_for_json(results)

    return jsonify(results)


@api.route('/screener/daily', methods=['POST'])
def daily_screener():
    """
    Run daily screener (Screen 2) on weekly results

    Request body:
        weekly_scan_id: ID of weekly scan to use

    Returns:
        Daily screen results filtered from weekly
    """
    data = request.get_json() or {}
    weekly_scan_id = data.get('weekly_scan_id')

    if not weekly_scan_id:
        return jsonify({'error': 'weekly_scan_id required'}), 400

    db = get_db()
    user_id = get_user_id()

    # Get weekly scan results
    weekly_scan = db.execute(
        'SELECT * FROM weekly_scans WHERE id = ?',
        (weekly_scan_id,)
    ).fetchone()

    if not weekly_scan:
        return jsonify({'error': 'Weekly scan not found'}), 404

    # Get stocks that passed weekly screen
    weekly_results = json.loads(weekly_scan['results'])
    bullish_stocks = [r for r in weekly_results if r.get('weekly_bullish')]

    # Run daily screen
    results = run_daily_screen(bullish_stocks)

    # Save to database
    today = datetime.now().date()
    db.execute('''
        INSERT INTO daily_scans 
        (user_id, weekly_scan_id, market, scan_date, results)
        VALUES (?, ?, ?, ?, ?)
    ''', (
        user_id, weekly_scan_id, weekly_scan['market'],
        today, json.dumps(results['all_results'])
    ))
    db.commit()

    scan_id = db.execute('SELECT last_insert_rowid()').fetchone()[0]
    results['scan_id'] = scan_id
    results['weekly_scan_id'] = weekly_scan_id

    # Sanitize results for JSON serialization
    results = sanitize_for_json(results)

    return jsonify(results)


@api.route('/screener/criteria', methods=['GET'])
def get_criteria():
    """Get grading criteria explanation"""
    return jsonify({
        'criteria': get_grading_criteria(),
        'scoring': {
            'a_trade': 'Signal Strength ≥ 5 AND Impulse not RED',
            'b_trade': 'Signal Strength 3-4 AND Impulse GREEN/BLUE',
            'watch': 'Signal Strength 1-2',
            'avoid': 'Signal Strength ≤ 0 OR Impulse RED'
        }
    })


# ============ INDICATOR CONFIGURATION ============
@api.route('/indicators/catalog', methods=['GET'])
def get_indicator_catalog():
    """Get full indicator catalog with all options per category"""
    return jsonify(INDICATOR_CATALOG)


@api.route('/indicators/category/<category>', methods=['GET'])
def get_category_indicators(category):
    """Get indicators for a specific category"""
    category = category.upper()
    if category in INDICATOR_CATALOG:
        return jsonify({
            'category': category,
            **INDICATOR_CATALOG[category]
        })
    return jsonify({'error': f'Unknown category: {category}'}), 404


@api.route('/indicators/configs', methods=['GET'])
def get_indicator_configs():
    """Get all available indicator configurations"""
    return jsonify({
        'default': DEFAULT_INDICATOR_CONFIG,
        'alternatives': ALTERNATIVE_CONFIGS
    })


@api.route('/indicators/config/<config_name>', methods=['GET'])
def get_indicator_config(config_name):
    """Get a specific indicator configuration"""
    if config_name == 'default':
        return jsonify(DEFAULT_INDICATOR_CONFIG)
    elif config_name in ALTERNATIVE_CONFIGS:
        return jsonify(ALTERNATIVE_CONFIGS[config_name])
    return jsonify({'error': f'Unknown config: {config_name}'}), 404


@api.route('/indicators/recommended', methods=['GET'])
def get_recommended_indicators():
    """Get Elder's recommended indicators"""
    recommended = {}
    for category, data in INDICATOR_CATALOG.items():
        for ind_id, ind_data in data['indicators'].items():
            if ind_data.get('recommended'):
                recommended[ind_id] = {
                    'category': category,
                    'category_description': data['description'],
                    **ind_data
                }
    return jsonify(recommended)


# ============ CANDLESTICK PATTERNS ============
@api.route('/patterns/catalog', methods=['GET'])
def get_pattern_catalog():
    """Get all candlestick patterns"""
    return jsonify(CANDLESTICK_PATTERNS)


@api.route('/patterns/bullish', methods=['GET'])
def get_bullish_patterns():
    """Get bullish candlestick patterns only"""
    bullish = {k: v for k, v in CANDLESTICK_PATTERNS.items()
               if 'bullish' in v.get('type', '')}
    return jsonify(bullish)


@api.route('/patterns/bearish', methods=['GET'])
def get_bearish_patterns():
    """Get bearish candlestick patterns only"""
    bearish = {k: v for k, v in CANDLESTICK_PATTERNS.items()
               if 'bearish' in v.get('type', '')}
    return jsonify(bearish)


@api.route('/screener/weekly/latest', methods=['GET'])
def get_latest_weekly():
    """Get latest weekly scan for current week"""
    market = request.args.get('market', 'US')

    today = datetime.now().date()
    week_start = today - timedelta(days=today.weekday())

    db = get_db()
    user_id = get_user_id()

    scan = db.execute('''
        SELECT * FROM weekly_scans 
        WHERE user_id = ? AND market = ? AND week_start = ?
        ORDER BY created_at DESC LIMIT 1
    ''', (user_id, market, week_start)).fetchone()

    if scan:
        return jsonify({
            'scan_id': scan['id'],
            'market': scan['market'],
            'scan_date': scan['scan_date'],
            'week_start': scan['week_start'],
            'week_end': scan['week_end'],
            'results': json.loads(scan['results']),
            'summary': json.loads(scan['summary']) if scan['summary'] else None
        })

    return jsonify({'message': 'No weekly scan found for current week'}), 404


# ============ STOCK INFO ============
@api.route('/stock/<symbol>', methods=['GET'])
def get_stock_analysis(symbol):
    """Get complete analysis for a single stock"""
    analysis = scan_stock(symbol)
    if analysis:
        return jsonify(analysis)
    return jsonify({'error': f'Could not analyze {symbol}'}), 404


# ============ SETTINGS ============
@api.route('/settings', methods=['GET'])
def get_settings():
    """Get all account settings"""
    db = get_db()
    user_id = get_user_id()

    settings = db.execute(
        'SELECT * FROM account_settings WHERE user_id = ?',
        (user_id,)
    ).fetchall()

    return jsonify([dict(s) for s in settings])


@api.route('/settings', methods=['POST'])
def create_setting():
    """Create new account setting"""
    data = request.get_json()
    db = get_db()
    user_id = get_user_id()

    db.execute('''
        INSERT INTO account_settings 
        (user_id, account_name, market, trading_capital, risk_per_trade,
         max_monthly_drawdown, target_rr, max_open_positions, currency, broker)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        user_id, data['account_name'], data['market'], data['trading_capital'],
        data.get('risk_per_trade', 2), data.get('max_monthly_drawdown', 6),
        data.get('target_rr', 2), data.get('max_open_positions', 5),
        data['currency'], data.get('broker')
    ))
    db.commit()

    return jsonify({'message': 'Setting created', 'id': db.execute('SELECT last_insert_rowid()').fetchone()[0]})


@api.route('/settings/<int:id>', methods=['PUT'])
def update_setting(id):
    """Update account setting"""
    data = request.get_json()
    db = get_db()
    user_id = get_user_id()

    db.execute('''
        UPDATE account_settings 
        SET account_name = ?, trading_capital = ?, risk_per_trade = ?,
            max_monthly_drawdown = ?, target_rr = ?, max_open_positions = ?,
            currency = ?, broker = ?, updated_at = CURRENT_TIMESTAMP
        WHERE id = ? AND user_id = ?
    ''', (
        data['account_name'], data['trading_capital'], data['risk_per_trade'],
        data['max_monthly_drawdown'], data['target_rr'], data['max_open_positions'],
        data['currency'], data.get('broker'), id, user_id
    ))
    db.commit()

    return jsonify({'message': 'Setting updated'})


# ============ STRATEGIES ============
@api.route('/strategies', methods=['GET'])
def get_strategies():
    """Get all strategies with APGAR parameters"""
    db = get_db()
    user_id = get_user_id()

    strategies = db.execute(
        'SELECT * FROM strategies WHERE user_id = ?',
        (user_id,)
    ).fetchall()

    result = []
    for s in strategies:
        strategy = dict(s)
        strategy['config'] = json.loads(strategy['config'])

        params = db.execute('''
            SELECT * FROM apgar_parameters 
            WHERE strategy_id = ? ORDER BY display_order
        ''', (s['id'],)).fetchall()

        strategy['apgar_parameters'] = [
            {**dict(p), 'options': json.loads(p['options'])} for p in params
        ]
        result.append(strategy)

    return jsonify(result)


# ============ WATCHLISTS ============
@api.route('/watchlists', methods=['GET'])
def get_watchlists():
    """Get all watchlists"""
    db = get_db()
    user_id = get_user_id()

    watchlists = db.execute(
        'SELECT * FROM watchlists WHERE user_id = ? ORDER BY created_at DESC',
        (user_id,)
    ).fetchall()

    result = []
    for w in watchlists:
        wd = dict(w)
        result.append({
            'id': wd['id'],
            'name': wd['name'],
            'market': wd['market'],
            'symbols': json.loads(wd['symbols']),
            'is_default': wd['is_default'],
            'symbol_count': len(json.loads(wd['symbols'])),
            'created_at': wd['created_at']
        })

    return jsonify(result)


@api.route('/watchlists', methods=['POST'])
def create_watchlist():
    """Create or add ticker to watchlist"""
    data = request.get_json()
    db = get_db()
    user_id = get_user_id()

    # If adding a single ticker to default watchlist
    if 'ticker' in data and 'market' in data:
        # Get or create default watchlist for market
        watchlist = db.execute('''
            SELECT * FROM watchlists WHERE user_id = ? AND market = ?
            ORDER BY is_default DESC LIMIT 1
        ''', (user_id, data['market'])).fetchone()

        if watchlist:
            symbols = json.loads(watchlist['symbols'])
            ticker = data['ticker'].upper()
            if ticker not in symbols:
                symbols.append(ticker)
                db.execute('''
                    UPDATE watchlists SET symbols = ? WHERE id = ?
                ''', (json.dumps(symbols), watchlist['id']))
                db.commit()
                return jsonify({'success': True, 'message': f'{ticker} added to watchlist', 'watchlist_id': watchlist['id']})
            else:
                return jsonify({'success': True, 'message': f'{ticker} already in watchlist'})
        else:
            # Create default watchlist
            db.execute('''
                INSERT INTO watchlists (user_id, name, market, symbols, is_default)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, f'{data["market"]} Default', data['market'], json.dumps([data['ticker'].upper()]), 1))
            db.commit()
            return jsonify({'success': True, 'message': f'Watchlist created with {data["ticker"]}'})

    # Create new named watchlist
    if 'name' in data and 'symbols' in data:
        db.execute('''
            INSERT INTO watchlists (user_id, name, market, symbols, is_default)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, data['name'], data.get('market', 'US'), json.dumps(data['symbols']), 0))
        db.commit()
        return jsonify({'success': True, 'message': 'Watchlist created'})

    return jsonify({'error': 'Invalid request'}), 400


# ============ TRADE SETUPS ============
@api.route('/setups', methods=['GET'])
def get_setups():
    """Get trade setups"""
    status = request.args.get('status', 'pending')
    db = get_db()
    user_id = get_user_id()

    setups = db.execute('''
        SELECT ts.*, s.name as strategy_name 
        FROM trade_setups ts
        LEFT JOIN strategies s ON ts.strategy_id = s.id
        WHERE ts.user_id = ? AND ts.status = ?
        ORDER BY ts.created_at DESC
    ''', (user_id, status)).fetchall()

    return jsonify([
        {**dict(s), 'apgar_details': json.loads(s['apgar_details'] or '{}')}
        for s in setups
    ])


@api.route('/setups', methods=['POST'])
def create_setup():
    """Create new trade setup"""
    data = request.get_json()
    db = get_db()
    user_id = get_user_id()

    db.execute('''
        INSERT INTO trade_setups 
        (user_id, daily_scan_id, symbol, market, strategy_id, apgar_score,
         apgar_details, entry_price, stop_loss, target_price, position_size,
         risk_amount, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        user_id, data.get('daily_scan_id'), data['symbol'], data['market'],
        data.get('strategy_id', 1), data.get('apgar_score'),
        json.dumps(data.get('apgar_details', {})),
        data['entry_price'], data['stop_loss'], data['target_price'],
        data.get('position_size'), data.get('risk_amount'), 'pending'
    ))
    db.commit()

    return jsonify({'message': 'Setup created', 'id': db.execute('SELECT last_insert_rowid()').fetchone()[0]})


# ============ TRADE JOURNAL ============
@api.route('/journal', methods=['GET'])
def get_journal():
    """Get trade journal entries"""
    status = request.args.get('status')
    limit = request.args.get('limit', 50, type=int)

    db = get_db()
    user_id = get_user_id()

    if status:
        entries = db.execute('''
            SELECT * FROM trade_journal
            WHERE user_id = ? AND status = ?
            ORDER BY created_at DESC LIMIT ?
        ''', (user_id, status, limit)).fetchall()
    else:
        entries = db.execute('''
            SELECT * FROM trade_journal
            WHERE user_id = ?
            ORDER BY created_at DESC LIMIT ?
        ''', (user_id, limit)).fetchall()

    return jsonify([dict(e) for e in entries])


@api.route('/journal', methods=['POST'])
def create_journal_entry():
    """Create trade journal entry"""
    data = request.get_json()
    db = get_db()
    user_id = get_user_id()

    db.execute('''
        INSERT INTO trade_journal 
        (user_id, symbol, market, direction, entry_date, entry_price,
         position_size, stop_loss, target_price, strategy_id, apgar_score,
         notes, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        user_id, data['symbol'], data['market'], data.get('direction', 'LONG'),
        data.get('entry_date'), data['entry_price'], data['position_size'],
        data['stop_loss'], data['target_price'], data.get('strategy_id'),
        data.get('apgar_score'), data.get('notes'), 'open'
    ))
    db.commit()

    return jsonify({'message': 'Entry created', 'id': db.execute('SELECT last_insert_rowid()').fetchone()[0]})


@api.route('/journal/<int:id>', methods=['PUT'])
def update_journal_entry(id):
    """Update/close trade journal entry"""
    data = request.get_json()
    db = get_db()
    user_id = get_user_id()

    entry = db.execute(
        'SELECT * FROM trade_journal WHERE id = ? AND user_id = ?',
        (id, user_id)
    ).fetchone()

    if not entry:
        return jsonify({'error': 'Entry not found'}), 404

    # Calculate P&L if closing
    pnl = pnl_percent = None
    if data.get('status') == 'closed' and data.get('exit_price'):
        exit_price = data['exit_price']
        entry_price = entry['entry_price']
        position_size = entry['position_size']

        pnl = (exit_price - entry_price) * position_size
        if entry['direction'] == 'SHORT':
            pnl = -pnl
        pnl -= data.get('fees', 0)
        pnl_percent = (pnl / (entry_price * position_size)) * 100

    db.execute('''
        UPDATE trade_journal 
        SET exit_date = ?, exit_price = ?, pnl = ?, pnl_percent = ?,
            fees = ?, notes = ?, lessons_learned = ?, grade = ?,
            status = ?, updated_at = CURRENT_TIMESTAMP
        WHERE id = ? AND user_id = ?
    ''', (
        data.get('exit_date'), data.get('exit_price'), pnl, pnl_percent,
        data.get('fees', 0), data.get('notes'), data.get('lessons_learned'),
        data.get('grade'), data.get('status', entry['status']), id, user_id
    ))
    db.commit()

    return jsonify({'message': 'Entry updated', 'pnl': pnl, 'pnl_percent': pnl_percent})


@api.route('/journal/stats', methods=['GET'])
def get_journal_stats():
    """Get trading statistics"""
    period = request.args.get('period', 'all')

    db = get_db()
    user_id = get_user_id()

    where = 'WHERE user_id = ? AND status = ?'
    params = [user_id, 'closed']

    if period == 'month':
        where += ' AND exit_date >= date("now", "-30 days")'
    elif period == 'year':
        where += ' AND exit_date >= date("now", "-365 days")'

    stats = db.execute(f'''
        SELECT 
            COUNT(*) as total_trades,
            SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
            SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
            SUM(pnl) as total_pnl,
            AVG(pnl) as avg_pnl,
            MAX(pnl) as best_trade,
            MIN(pnl) as worst_trade
        FROM trade_journal {where}
    ''', params).fetchone()

    result = dict(stats)
    result['win_rate'] = (
        (result['winning_trades'] / result['total_trades'] * 100)
        if result['total_trades'] > 0 else 0
    )

    return jsonify(result)


# ============ CHECKLIST ============
@api.route('/checklist', methods=['GET'])
def get_checklist():
    """Get today's checklist"""
    today = datetime.now().date()
    db = get_db()
    user_id = get_user_id()

    checklist = db.execute('''
        SELECT * FROM daily_checklist 
        WHERE user_id = ? AND checklist_date = ?
    ''', (user_id, today)).fetchone()

    if checklist:
        return jsonify({
            'date': checklist['checklist_date'],
            'items': json.loads(checklist['items']),
            'completed': checklist['completed_at'] is not None
        })

    default_items = {f'step{i}': False for i in range(1, 8)}
    return jsonify({
        'date': today.isoformat(),
        'items': default_items,
        'completed': False
    })


@api.route('/checklist', methods=['POST'])
def update_checklist():
    """Update checklist"""
    data = request.get_json()
    today = datetime.now().date()
    db = get_db()
    user_id = get_user_id()

    all_done = all(data['items'].values())
    completed_at = datetime.now() if all_done else None

    db.execute('''
        INSERT OR REPLACE INTO daily_checklist 
        (user_id, checklist_date, items, completed_at)
        VALUES (?, ?, ?, ?)
    ''', (user_id, today, json.dumps(data['items']), completed_at))
    db.commit()

    return jsonify({'message': 'Checklist updated', 'completed': all_done})


# ============ TRADE BILLS ============
@api.route('/trade-bills', methods=['POST'])
def create_trade_bill():
    """Create a new trade bill"""
    data = request.get_json()
    user_id = get_user_id()
    db = get_database()

    try:
        conn = db.get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO trade_bills (
                user_id, ticker, current_market_price, entry_price, stop_loss, target_price,
                quantity, upper_channel, lower_channel, target_pips, stop_loss_pips,
                max_qty_for_risk, overnight_charges, risk_per_share, position_size,
                risk_percent, channel_height, potential_gain, target_1_1_c, target_1_2_b,
                target_1_3_a, risk_amount_currency, reward_amount_currency, risk_reward_ratio,
                break_even, trailing_stop, is_filled, stop_entered, target_entered,
                journal_entered, comments, status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_id, data.get('ticker'), data.get('current_market_price'),
            data.get('entry_price'), data.get(
                'stop_loss'), data.get('target_price'),
            data.get('quantity'), data.get(
                'upper_channel'), data.get('lower_channel'),
            data.get('target_pips'), data.get(
                'stop_loss_pips'), data.get('max_qty_for_risk'),
            data.get('overnight_charges'), data.get(
                'risk_per_share'), data.get('position_size'),
            data.get('risk_percent'), data.get(
                'channel_height'), data.get('potential_gain'),
            data.get('target_1_1_c'), data.get(
                'target_1_2_b'), data.get('target_1_3_a'),
            data.get('risk_amount_currency'), data.get(
                'reward_amount_currency'),
            data.get('risk_reward_ratio'), data.get(
                'break_even'), data.get('trailing_stop'),
            1 if data.get('is_filled') else 0, 1 if data.get(
                'stop_entered') else 0,
            1 if data.get('target_entered') else 0, 1 if data.get(
                'journal_entered') else 0,
            data.get('comments', ''), 'active', datetime.now().isoformat()
        ))
        conn.commit()
        trade_bill_id = cursor.lastrowid
        conn.close()

        return jsonify({
            'success': True,
            'id': trade_bill_id,
            'message': f'Trade Bill for {data.get("ticker")} created successfully'
        }), 201
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@api.route('/trade-bills', methods=['GET'])
def get_trade_bills():
    """Get all trade bills for current user"""
    user_id = get_user_id()
    status = request.args.get('status')
    db = get_database()

    try:
        conn = db.get_connection()
        cursor = conn.cursor()

        if status:
            cursor.execute(
                'SELECT * FROM trade_bills WHERE user_id = ? AND status = ? ORDER BY created_at DESC', (user_id, status))
        else:
            cursor.execute(
                'SELECT * FROM trade_bills WHERE user_id = ? ORDER BY created_at DESC', (user_id,))

        rows = cursor.fetchall()
        conn.close()

        trade_bills = [dict(row) for row in rows]
        return jsonify(trade_bills)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@api.route('/trade-bills/<int:trade_bill_id>', methods=['GET'])
def get_trade_bill_detail(trade_bill_id):
    """Get a specific trade bill"""
    db = get_database()
    trade_bill = db.get_trade_bill(trade_bill_id)

    if not trade_bill:
        return jsonify({'error': 'Trade bill not found'}), 404

    return jsonify(trade_bill)


@api.route('/trade-bills/<int:trade_bill_id>', methods=['PUT'])
def update_trade_bill(trade_bill_id):
    """Update a trade bill"""
    data = request.get_json()
    db = get_database()

    try:
        success = db.update_trade_bill(trade_bill_id, data)
        if success:
            return jsonify({'success': True, 'message': 'Trade Bill updated'})
        else:
            return jsonify({'error': 'Trade Bill not found'}), 404
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@api.route('/trade-bills/<int:trade_bill_id>', methods=['DELETE'])
def delete_trade_bill(trade_bill_id):
    """Delete a trade bill"""
    db = get_database()
    success = db.delete_trade_bill(trade_bill_id)

    if success:
        return jsonify({'success': True, 'message': 'Trade Bill deleted'})
    else:
        return jsonify({'error': 'Trade Bill not found'}), 404


@api.route('/trade-bills/calculate', methods=['POST'])
def calculate_trade_metrics():
    """Calculate trade metrics and position sizing"""
    data = request.get_json()
    db = get_database()

    try:
        metrics = db.calculate_trade_metrics(
            entry_price=data['entry_price'],
            stop_loss=data['stop_loss'],
            target_price=data['target_price'],
            quantity=data['quantity'],
            account_capital=data.get('account_capital', 10000),
            risk_percent=data.get('risk_percent', 2)
        )
        return jsonify(metrics)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


# ============ ACCOUNT INFORMATION ============
@api.route('/account/info', methods=['GET'])
def get_account_info():
    """Get account information and metrics"""
    user_id = get_user_id()
    db = get_db()

    # Get account settings
    account = db.execute('''
        SELECT * FROM account_settings 
        WHERE user_id = ?
    ''', (user_id,)).fetchone()

    if not account:
        return jsonify({'error': 'Account not found'}), 404

    account = dict(account)

    # Get open positions count
    open_positions = db.execute('''
        SELECT COUNT(*) as count FROM trade_journal
        WHERE user_id = ? AND status = 'open'
    ''', (user_id,)).fetchone()

    account['no_of_open_positions'] = open_positions['count']

    # Get money locked in positions
    money_locked = db.execute('''
        SELECT COALESCE(SUM(position_size * entry_price), 0) as locked
        FROM trade_journal
        WHERE user_id = ? AND status = 'open'
    ''', (user_id,)).fetchone()

    account['money_locked_in_positions'] = money_locked['locked']

    # Calculate money remaining to risk
    max_monthly_risk = (account['trading_capital']
                        * account['max_monthly_drawdown']) / 100
    risk_per_trade = (account['trading_capital'] *
                      account['risk_per_trade']) / 100

    account['money_remaining_to_risk'] = max_monthly_risk - \
        money_locked['locked']
    account['risk_percent_remaining'] = (
        account['money_remaining_to_risk'] / account['trading_capital']) * 100

    return jsonify(account)


@api.route('/account/info', methods=['PUT'])
def update_account_info():
    """Update account information"""
    user_id = get_user_id()
    data = request.get_json()
    db = get_db()

    set_clause = ', '.join([f'{k} = ?' for k in data.keys()])
    values = tuple(data.values())

    db.execute(f'''
        UPDATE account_settings
        SET {set_clause}, updated_at = CURRENT_TIMESTAMP
        WHERE user_id = ?
    ''', (*values, user_id))
    db.commit()

    return jsonify({'success': True, 'message': 'Account information updated'})


# ============ TRADE LOG (JOURNAL) ============
@api.route('/trade-log', methods=['GET'])
def get_trade_log():
    """Get all trades from trade log"""
    user_id = get_user_id()
    db = get_db()

    trades = db.execute('''
        SELECT * FROM trade_log 
        WHERE user_id = ? 
        ORDER BY entry_date DESC
    ''', (user_id,)).fetchall()

    return jsonify({'trades': [dict(t) for t in trades]})


@api.route('/trade-log', methods=['POST'])
def create_trade_log_entry():
    """Create a new trade log entry"""
    data = request.get_json()
    user_id = get_user_id()
    db = get_db()

    try:
        db.execute('''
            INSERT INTO trade_log (
                user_id, entry_date, symbol, strategy, direction,
                entry_price, shares, stop_loss, take_profit,
                exit_date, exit_price, trade_costs, gross_pnl, net_pnl,
                mistake, discipline_rating, notes, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_id, data.get('entry_date'), data.get('symbol'),
            data.get('strategy'), data.get('direction'),
            data.get('entry_price'), data.get('shares'),
            data.get('stop_loss'), data.get('take_profit'),
            data.get('exit_date'), data.get('exit_price'),
            data.get('trade_costs', 0), data.get('gross_pnl'),
            data.get('net_pnl'), data.get('mistake'),
            data.get('discipline_rating', 8), data.get('notes'),
            'closed' if data.get('exit_date') else 'open'
        ))
        db.commit()

        trade_id = db.execute('SELECT last_insert_rowid()').fetchone()[0]
        return jsonify({'success': True, 'id': trade_id, 'message': 'Trade saved'}), 201
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@api.route('/trade-log/<int:trade_id>', methods=['PUT'])
def update_trade_log_entry(trade_id):
    """Update a trade log entry"""
    data = request.get_json()
    user_id = get_user_id()
    db = get_db()

    try:
        # Build dynamic update query
        fields = []
        values = []
        for key, value in data.items():
            if key not in ['id', 'user_id']:
                fields.append(f'{key} = ?')
                values.append(value)

        if fields:
            values.append(trade_id)
            values.append(user_id)
            db.execute(f'''
                UPDATE trade_log 
                SET {', '.join(fields)}
                WHERE id = ? AND user_id = ?
            ''', tuple(values))
            db.commit()

        return jsonify({'success': True, 'message': 'Trade updated'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@api.route('/trade-log/<int:trade_id>', methods=['DELETE'])
def delete_trade_log_entry(trade_id):
    """Delete a trade log entry"""
    user_id = get_user_id()
    db = get_db()

    db.execute('DELETE FROM trade_log WHERE id = ? AND user_id = ?',
               (trade_id, user_id))
    db.commit()

    return jsonify({'success': True, 'message': 'Trade deleted'})


@api.route('/trade-log/summary', methods=['GET'])
def get_trade_summary():
    """Get trade summary statistics"""
    user_id = get_user_id()
    db = get_db()

    closed = db.execute('''
        SELECT * FROM trade_log 
        WHERE user_id = ? AND status = 'closed'
    ''', (user_id,)).fetchall()

    trades = [dict(t) for t in closed]

    winners = [t for t in trades if (t.get('net_pnl') or 0) > 0]
    losers = [t for t in trades if (t.get('net_pnl') or 0) < 0]

    total_pnl = sum(t.get('net_pnl', 0) or 0 for t in trades)
    avg_win = sum(t['net_pnl'] for t in winners) / \
        len(winners) if winners else 0
    avg_loss = sum(abs(t['net_pnl'])
                   for t in losers) / len(losers) if losers else 0

    return jsonify({
        'total_trades': len(trades),
        'winners': len(winners),
        'losers': len(losers),
        'breakeven': len(trades) - len(winners) - len(losers),
        'win_rate': len(winners) / len(trades) * 100 if trades else 0,
        'total_pnl': total_pnl,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': avg_win / avg_loss if avg_loss > 0 else 0,
        'expectancy': total_pnl / len(trades) if trades else 0
    })


# ============ INDICATOR FILTER CONFIG ============
@api.route('/indicator-filters', methods=['GET'])
def get_indicator_filters():
    """Get saved indicator filter configuration"""
    user_id = get_user_id()
    db = get_db()

    config = db.execute('''
        SELECT config FROM indicator_filters WHERE user_id = ?
    ''', (user_id,)).fetchone()

    if config:
        return jsonify(json.loads(config['config']))

    # Return default config
    return jsonify(DEFAULT_INDICATOR_CONFIG)


@api.route('/indicator-filters', methods=['POST'])
def save_indicator_filters():
    """Save indicator filter configuration"""
    user_id = get_user_id()
    data = request.get_json()
    db = get_db()

    db.execute('''
        INSERT OR REPLACE INTO indicator_filters (user_id, config, updated_at)
        VALUES (?, ?, CURRENT_TIMESTAMP)
    ''', (user_id, json.dumps(data)))
    db.commit()

    return jsonify({'success': True, 'message': 'Indicator filters saved'})


# ========== FAVORITES ==========
@api.route('/favorites', methods=['GET'])
def get_favorites():
    """Get all favorite stocks for current market with price data"""
    user_id = get_user_id()
    market = request.args.get('market', 'US')
    db = get_db()

    favorites = db.execute('''
        SELECT id, symbol, market, notes, created_at 
        FROM favorite_stocks 
        WHERE user_id = ? AND market = ?
        ORDER BY created_at DESC
    ''', (user_id, market)).fetchall()

    # Get stock data for each favorite
    from services.screener import scan_stock
    results = []
    for fav in favorites:
        try:
            stock_data = scan_stock(fav['symbol'], market)
            stock_data['notes'] = fav['notes']
            stock_data['fav_id'] = fav['id']
            stock_data['created_at'] = fav['created_at']
            results.append(stock_data)
        except Exception as e:
            print(f"Error scanning {fav['symbol']}: {e}")
            # Still add favorite even if scan failed
            results.append({
                'symbol': fav['symbol'],
                'name': fav['symbol'],
                'price': 0,
                'notes': fav['notes'],
                'fav_id': fav['id'],
                'created_at': fav['created_at'],
                'error': str(e)
            })

    return jsonify({
        'success': True,
        'favorites': results,
        'count': len(results)
    })


@api.route('/favorites/<symbol>', methods=['POST'])
def toggle_favorite(symbol):
    """Toggle favorite status for a stock"""
    user_id = get_user_id()
    data = request.get_json() or {}
    market = data.get('market', 'US')
    notes = data.get('notes', '')
    db = get_db()

    # Check if already favorited
    existing = db.execute('''
        SELECT id FROM favorite_stocks 
        WHERE user_id = ? AND symbol = ? AND market = ?
    ''', (user_id, symbol, market)).fetchone()

    if existing:
        # Remove favorite
        db.execute('''
            DELETE FROM favorite_stocks 
            WHERE user_id = ? AND symbol = ? AND market = ?
        ''', (user_id, symbol, market))
        db.commit()
        return jsonify({'success': True, 'favorited': False})
    else:
        # Add favorite
        db.execute('''
            INSERT INTO favorite_stocks (user_id, symbol, market, notes)
            VALUES (?, ?, ?, ?)
        ''', (user_id, symbol, market, notes))
        db.commit()
        return jsonify({'success': True, 'favorited': True})


@api.route('/favorites/<symbol>', methods=['DELETE'])
def remove_favorite(symbol):
    """Remove a stock from favorites"""
    user_id = get_user_id()
    market = request.args.get('market', 'US')
    db = get_db()

    db.execute('''
        DELETE FROM favorite_stocks 
        WHERE user_id = ? AND symbol = ? AND market = ?
    ''', (user_id, symbol, market))
    db.commit()

    return jsonify({'success': True, 'message': f'{symbol} removed from favorites'})


@api.route('/favorites/<symbol>/notes', methods=['PUT'])
def update_favorite_notes(symbol):
    """Update notes for a favorite stock"""
    user_id = get_user_id()
    data = request.get_json() or {}
    market = data.get('market', 'US')
    notes = data.get('notes', '')
    db = get_db()

    db.execute('''
        UPDATE favorite_stocks 
        SET notes = ?, updated_at = CURRENT_TIMESTAMP
        WHERE user_id = ? AND symbol = ? AND market = ?
    ''', (notes, user_id, symbol, market))
    db.commit()

    return jsonify({'success': True, 'message': 'Notes updated'})


@api.route('/favorites/check/<symbol>', methods=['GET'])
def check_favorite(symbol):
    """Check if stock is favorited"""
    user_id = get_user_id()
    market = request.args.get('market', 'US')
    db = get_db()

    favorite = db.execute('''
        SELECT id FROM favorite_stocks 
        WHERE user_id = ? AND symbol = ? AND market = ?
    ''', (user_id, symbol, market)).fetchone()

    return jsonify({'favorited': favorite is not None})


# ========== BACKTESTING ==========
@api.route('/backtest/run', methods=['POST'])
def run_backtest():
    """Run backtest for a symbol"""
    from services.backtesting import run_backtest_for_symbol

    data = request.get_json() or {}
    symbol = data.get('symbol', '').upper()
    market = data.get('market', 'US')
    lookback_days = data.get('lookback_days', 90)
    config = data.get('config', {})

    if not symbol:
        return jsonify({'error': 'Symbol required'}), 400

    try:
        results = run_backtest_for_symbol(
            symbol, market, lookback_days, config)
        results = sanitize_for_json(results)
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api.route('/backtest/available-stocks', methods=['GET'])
def get_backtest_stocks():
    """Get list of available stocks for backtesting"""
    market = request.args.get('market', 'US')

    # Get stocks that have at least 90 days of historical data
    db = get_db()
    stocks = db.execute('''
        SELECT DISTINCT symbol
        FROM stock_historical_data
        GROUP BY symbol
        HAVING COUNT(*) >= 90
        ORDER BY symbol
        LIMIT 100
    ''').fetchall()

    return jsonify({
        'success': True,
        'stocks': [s['symbol'] for s in stocks],
        'count': len(stocks)
    })
