"""
Elder Trading System - Enhanced API Routes v2
New endpoints for connected workflow: Screener → Trade Bill → IBKR → Trade Log → Position Management
"""

from flask import Blueprint, request, jsonify, g
from datetime import datetime, timedelta
import json

from models.database import get_database
from services.screener_v2 import (
    run_weekly_screen_v2, 
    run_daily_screen_v2, 
    scan_stock_v2,
    calculate_elder_trade_levels
)
from services.indicators import get_grading_criteria
from services.ibkr_orders import (
    check_ibkr_connection,
    get_account_id,
    place_bracket_order,
    place_single_order,
    get_open_orders,
    cancel_order,
    get_positions,
    get_position_alerts,
    get_filled_trades,
    create_trade_from_bill,
    get_account_summary,
    modify_order,
    get_market_data
)

api_v2 = Blueprint('api_v2', __name__, url_prefix='/api/v2')


def get_db():
    if 'db' not in g:
        g.db = get_database().get_connection()
    return g.db


def get_user_id():
    return getattr(g, 'user_id', 1)


# ══════════════════════════════════════════════════════════════════════════════
# ENHANCED SCREENER ENDPOINTS (v2 with validation fixes)
# ══════════════════════════════════════════════════════════════════════════════

@api_v2.route('/screener/run', methods=['POST'])
def run_screener_v2():
    """
    Run enhanced screener v2 with all validation fixes
    
    Features:
    - Screen 1 as mandatory gate
    - Impulse RED blocks trades
    - Correct daily_ready logic
    - New high-scoring rules
    - Elder Entry/Stop/Target calculations
    """
    data = request.get_json() or {}
    market = data.get('market', 'US')
    symbols = data.get('symbols')
    
    results = run_weekly_screen_v2(market, symbols)
    
    # Save to database
    db = get_db()
    user_id = get_user_id()
    today = datetime.now().date()
    
    db.execute('''
        INSERT INTO weekly_scans 
        (user_id, market, scan_date, week_start, week_end, results, summary)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        user_id, market, today, 
        today - timedelta(days=today.weekday()),
        today + timedelta(days=6-today.weekday()),
        json.dumps(results['all_results']),
        json.dumps(results['summary'])
    ))
    db.commit()
    
    scan_id = db.execute('SELECT last_insert_rowid()').fetchone()[0]
    results['scan_id'] = scan_id
    
    return jsonify(results)


@api_v2.route('/screener/stock/<symbol>', methods=['GET'])
def analyze_stock_v2(symbol):
    """Analyze a single stock with v2 logic"""
    result = scan_stock_v2(symbol)
    if result:
        return jsonify(result)
    return jsonify({'error': f'Could not analyze {symbol}'}), 404


# ══════════════════════════════════════════════════════════════════════════════
# IBKR ORDER PLACEMENT
# ══════════════════════════════════════════════════════════════════════════════

@api_v2.route('/ibkr/status', methods=['GET'])
def ibkr_status_v2():
    """Check IBKR Gateway connection status"""
    connected, message = check_ibkr_connection()
    account_id = get_account_id() if connected else None
    
    return jsonify({
        'connected': connected,
        'message': message,
        'account_id': account_id,
        'gateway_url': 'https://localhost:5000'
    })


@api_v2.route('/ibkr/account', methods=['GET'])
def get_account():
    """Get IBKR account summary"""
    result = get_account_summary()
    return jsonify(result)


@api_v2.route('/ibkr/orders', methods=['GET'])
def get_orders():
    """Get all open/pending orders"""
    result = get_open_orders()
    return jsonify(result)


@api_v2.route('/ibkr/orders', methods=['POST'])
def create_order():
    """
    Place a new order
    
    Body:
    {
        "symbol": "AAPL",
        "side": "BUY",
        "quantity": 10,
        "price": 150.00,
        "order_type": "LMT",  // LMT, MKT, STP
        "tif": "GTC"  // GTC, DAY
    }
    """
    data = request.get_json()
    
    result = place_single_order(
        symbol=data['symbol'],
        side=data['side'],
        quantity=data['quantity'],
        price=data['price'],
        order_type=data.get('order_type', 'LMT'),
        tif=data.get('tif', 'GTC')
    )
    
    if result['success']:
        return jsonify(result), 201
    return jsonify(result), 400


@api_v2.route('/ibkr/orders/bracket', methods=['POST'])
def create_bracket_order():
    """
    Place bracket order (Entry + Stop + Target)
    
    Body:
    {
        "symbol": "AAPL",
        "quantity": 10,
        "entry_price": 150.00,
        "stop_loss": 145.00,
        "take_profit": 160.00
    }
    """
    data = request.get_json()
    
    result = place_bracket_order(
        symbol=data['symbol'],
        quantity=data['quantity'],
        entry_price=data['entry_price'],
        stop_loss=data['stop_loss'],
        take_profit=data['take_profit']
    )
    
    if result['success']:
        return jsonify(result), 201
    return jsonify(result), 400


@api_v2.route('/ibkr/orders/<order_id>', methods=['DELETE'])
def cancel_order_endpoint(order_id):
    """Cancel an order"""
    result = cancel_order(order_id)
    if result['success']:
        return jsonify(result)
    return jsonify(result), 400


@api_v2.route('/ibkr/orders/<order_id>', methods=['PUT'])
def modify_order_endpoint(order_id):
    """Modify an existing order"""
    data = request.get_json()
    result = modify_order(
        order_id,
        new_price=data.get('price'),
        new_quantity=data.get('quantity')
    )
    if result['success']:
        return jsonify(result)
    return jsonify(result), 400


# ══════════════════════════════════════════════════════════════════════════════
# POSITION MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

@api_v2.route('/positions', methods=['GET'])
def get_all_positions():
    """
    Get all open positions with current P/L
    Returns positions from IBKR with real-time market prices
    """
    result = get_positions()
    
    if result['success']:
        # Add alerts
        db = get_db()
        user_id = get_user_id()
        
        # Get trade bills for matching
        bills = db.execute('''
            SELECT * FROM trade_bills WHERE user_id = ? AND status = 'PENDING'
        ''', (user_id,)).fetchall()
        trade_bills = [dict(b) for b in bills]
        
        alerts = get_position_alerts(result['positions'], trade_bills)
        result['alerts'] = alerts
        result['alert_count'] = len(alerts)
        result['high_priority_alerts'] = len([a for a in alerts if a.get('severity') == 'HIGH'])
    
    return jsonify(result)


@api_v2.route('/positions/summary', methods=['GET'])
def get_position_summary():
    """Get position summary with totals"""
    result = get_positions()
    
    if result['success']:
        positions = result['positions']
        
        return jsonify({
            'success': True,
            'total_positions': len(positions),
            'total_market_value': sum(p['market_value'] for p in positions),
            'total_unrealized_pnl': sum(p['unrealized_pnl'] for p in positions),
            'total_realized_pnl': sum(p['realized_pnl'] for p in positions),
            'winning_positions': len([p for p in positions if p['unrealized_pnl'] > 0]),
            'losing_positions': len([p for p in positions if p['unrealized_pnl'] < 0]),
            'positions': positions
        })
    
    return jsonify(result)


@api_v2.route('/positions/<symbol>/close', methods=['POST'])
def close_position(symbol):
    """
    Close a position (market sell)
    
    Body (optional):
    {
        "quantity": 50  // Partial close
    }
    """
    data = request.get_json() or {}
    
    # Get current position
    positions = get_positions()
    if not positions['success']:
        return jsonify(positions), 400
    
    position = None
    for p in positions['positions']:
        if p['symbol'] == symbol:
            position = p
            break
    
    if not position:
        return jsonify({'success': False, 'error': f'No position found for {symbol}'}), 404
    
    quantity = data.get('quantity', position['quantity'])
    
    # Place market sell order
    result = place_single_order(
        symbol=symbol,
        side='SELL',
        quantity=quantity,
        price=0,  # Market order
        order_type='MKT',
        tif='DAY'
    )
    
    if result['success']:
        result['message'] = f'Closing {quantity} shares of {symbol}'
    
    return jsonify(result)


# ══════════════════════════════════════════════════════════════════════════════
# CONNECTED WORKFLOW: Trade Bill → IBKR Order
# ══════════════════════════════════════════════════════════════════════════════

@api_v2.route('/trade-bills/<int:bill_id>/place-order', methods=['POST'])
def place_order_from_bill(bill_id):
    """
    Place IBKR order directly from Trade Bill
    
    This is the key connection in the workflow:
    Screener → Trade Bill → IBKR Order
    
    The Trade Bill contains:
    - Entry price (EMA-22)
    - Stop loss (deepest penetration)
    - Target (KC upper)
    - Quantity (calculated from risk)
    """
    db = get_db()
    user_id = get_user_id()
    
    # Get trade bill
    bill = db.execute('''
        SELECT * FROM trade_bills WHERE id = ? AND user_id = ?
    ''', (bill_id, user_id)).fetchone()
    
    if not bill:
        return jsonify({'success': False, 'error': 'Trade Bill not found'}), 404
    
    bill_data = dict(bill)
    
    # Place the order
    result = create_trade_from_bill({
        'id': bill_id,
        'symbol': bill_data['symbol'],
        'entry': bill_data['entry_price'],
        'stop_loss': bill_data['stop_loss'],
        'target': bill_data['target_price'],
        'quantity': bill_data['quantity']
    })
    
    if result['success']:
        # Update trade bill status
        db.execute('''
            UPDATE trade_bills 
            SET status = 'ORDERED', order_id = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (result.get('order_id'), bill_id))
        db.commit()
        
        result['trade_bill_updated'] = True
    
    return jsonify(result)


@api_v2.route('/trade-bills/from-screener', methods=['POST'])
def create_bill_from_screener():
    """
    Create Trade Bill directly from screener result
    
    Body:
    {
        "symbol": "AAPL",
        "screener_data": { ... }  // Optional, will fetch if not provided
    }
    """
    data = request.get_json()
    symbol = data.get('symbol')
    
    if not symbol:
        return jsonify({'success': False, 'error': 'Symbol required'}), 400
    
    # Get fresh analysis if not provided
    screener_data = data.get('screener_data')
    if not screener_data:
        screener_data = scan_stock_v2(symbol)
        if not screener_data:
            return jsonify({'success': False, 'error': f'Could not analyze {symbol}'}), 400
    
    # Get account settings for position sizing
    db = get_db()
    user_id = get_user_id()
    
    account = db.execute('''
        SELECT * FROM account_settings WHERE user_id = ?
    ''', (user_id,)).fetchone()
    
    if not account:
        return jsonify({'success': False, 'error': 'Account settings not found'}), 400
    
    account = dict(account)
    risk_per_trade = account['trading_capital'] * (account['risk_per_trade'] / 100)
    
    # Calculate position size
    entry = screener_data['entry']
    stop = screener_data['stop_loss']
    risk_per_share = entry - stop
    
    if risk_per_share <= 0:
        return jsonify({'success': False, 'error': 'Invalid stop loss (above entry)'}), 400
    
    quantity = int(risk_per_trade / risk_per_share)
    position_value = quantity * entry
    
    # Create trade bill
    db.execute('''
        INSERT INTO trade_bills (
            user_id, symbol, market, direction, entry_price, stop_loss,
            target_price, quantity, position_value, risk_amount,
            risk_reward_ratio, signal_strength, grade, notes, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        user_id, symbol, 'US', 'LONG',
        entry, stop, screener_data['target'],
        quantity, position_value, risk_per_trade,
        screener_data['risk_reward_ratio'],
        screener_data['signal_strength'],
        screener_data['grade'],
        f"Created from screener. High-value signals: {', '.join(screener_data.get('high_value_signals', []))}",
        'PENDING'
    ))
    db.commit()
    
    bill_id = db.execute('SELECT last_insert_rowid()').fetchone()[0]
    
    return jsonify({
        'success': True,
        'trade_bill_id': bill_id,
        'symbol': symbol,
        'entry': entry,
        'stop_loss': stop,
        'target': screener_data['target'],
        'quantity': quantity,
        'position_value': position_value,
        'risk_amount': risk_per_trade,
        'risk_reward': screener_data['rr_display'],
        'grade': screener_data['grade'],
        'signal_strength': screener_data['signal_strength']
    }), 201


# ══════════════════════════════════════════════════════════════════════════════
# AUTO-SYNC TRADE LOG FROM IBKR
# ══════════════════════════════════════════════════════════════════════════════

@api_v2.route('/trade-log/sync-ibkr', methods=['POST'])
def sync_trade_log_from_ibkr():
    """
    Sync trade log with filled orders from IBKR
    
    This pulls executed trades from IBKR and creates/updates trade log entries
    """
    result = get_filled_trades(days_back=7)
    
    if not result['success']:
        return jsonify(result), 400
    
    db = get_db()
    user_id = get_user_id()
    
    synced = 0
    skipped = 0
    
    for trade in result['trades']:
        # Check if already exists
        existing = db.execute('''
            SELECT id FROM trade_log 
            WHERE user_id = ? AND symbol = ? AND entry_date = ?
        ''', (user_id, trade['symbol'], trade['execution_time'])).fetchone()
        
        if existing:
            skipped += 1
            continue
        
        # Create trade log entry
        side = 'Long' if trade['side'] == 'BOT' else 'Short'
        status = 'open' if trade['side'] == 'BOT' else 'closed'
        
        db.execute('''
            INSERT INTO trade_log (
                user_id, entry_date, symbol, strategy, direction,
                entry_price, shares, trade_costs, status, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_id, trade['execution_time'], trade['symbol'],
            'EL - Elder System', side, trade['price'],
            trade['quantity'], trade['commission'], status,
            f"Auto-synced from IBKR. Order ref: {trade['order_ref']}"
        ))
        synced += 1
    
    db.commit()
    
    return jsonify({
        'success': True,
        'synced': synced,
        'skipped': skipped,
        'total_ibkr_trades': len(result['trades']),
        'message': f'Synced {synced} new trades from IBKR ({skipped} already existed)'
    })


@api_v2.route('/trade-log/update-from-positions', methods=['POST'])
def update_trade_log_from_positions():
    """
    Update open trade log entries with current position P/L
    """
    positions = get_positions()
    
    if not positions['success']:
        return jsonify(positions), 400
    
    db = get_db()
    user_id = get_user_id()
    
    updated = 0
    
    for pos in positions['positions']:
        # Find matching open trade
        trade = db.execute('''
            SELECT id FROM trade_log 
            WHERE user_id = ? AND symbol = ? AND status = 'open'
        ''', (user_id, pos['symbol'])).fetchone()
        
        if trade:
            # Update with current P/L
            db.execute('''
                UPDATE trade_log
                SET gross_pnl = ?, notes = ?
                WHERE id = ?
            ''', (
                pos['unrealized_pnl'],
                f"Live P/L: ${pos['unrealized_pnl']:.2f} ({pos['pnl_percent']:.1f}%)",
                trade['id']
            ))
            updated += 1
    
    db.commit()
    
    return jsonify({
        'success': True,
        'updated': updated,
        'total_positions': len(positions['positions'])
    })


# ══════════════════════════════════════════════════════════════════════════════
# MARKET DATA
# ══════════════════════════════════════════════════════════════════════════════

@api_v2.route('/market-data/<symbol>', methods=['GET'])
def get_symbol_market_data(symbol):
    """Get current market data for a symbol"""
    result = get_market_data(symbol)
    return jsonify(result)


@api_v2.route('/market-data/batch', methods=['POST'])
def get_batch_market_data():
    """Get market data for multiple symbols"""
    data = request.get_json()
    symbols = data.get('symbols', [])
    
    results = {}
    for symbol in symbols[:20]:  # Limit to 20
        results[symbol] = get_market_data(symbol)
    
    return jsonify({'results': results})


# ══════════════════════════════════════════════════════════════════════════════
# WORKFLOW STATUS
# ══════════════════════════════════════════════════════════════════════════════

@api_v2.route('/workflow/status', methods=['GET'])
def get_workflow_status():
    """
    Get overall workflow status for dashboard
    
    Shows:
    - Latest screener results
    - Pending trade bills
    - Open orders
    - Open positions with P/L
    - Alerts
    """
    db = get_db()
    user_id = get_user_id()
    
    # Get latest scan
    latest_scan = db.execute('''
        SELECT * FROM weekly_scans 
        WHERE user_id = ? 
        ORDER BY scan_date DESC LIMIT 1
    ''', (user_id,)).fetchone()
    
    scan_summary = None
    if latest_scan:
        scan_summary = json.loads(latest_scan['summary'])
        scan_summary['scan_date'] = latest_scan['scan_date']
    
    # Get pending trade bills
    pending_bills = db.execute('''
        SELECT COUNT(*) as count FROM trade_bills 
        WHERE user_id = ? AND status = 'PENDING'
    ''', (user_id,)).fetchone()
    
    # Get open orders from IBKR
    orders = get_open_orders()
    
    # Get positions
    positions = get_positions()
    
    # Get trade bills for alerts
    bills = db.execute('''
        SELECT * FROM trade_bills WHERE user_id = ?
    ''', (user_id,)).fetchall()
    trade_bills = [dict(b) for b in bills]
    
    alerts = []
    if positions['success']:
        alerts = get_position_alerts(positions['positions'], trade_bills)
    
    return jsonify({
        'ibkr_connected': positions['success'],
        'latest_scan': scan_summary,
        'pending_trade_bills': pending_bills['count'] if pending_bills else 0,
        'open_orders': orders.get('count', 0),
        'open_positions': positions.get('count', 0),
        'total_unrealized_pnl': positions.get('total_unrealized_pnl', 0),
        'total_market_value': positions.get('total_market_value', 0),
        'alerts': alerts,
        'high_priority_alerts': len([a for a in alerts if a.get('severity') == 'HIGH'])
    })
