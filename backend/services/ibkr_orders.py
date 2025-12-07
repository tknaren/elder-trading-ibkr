"""
Elder Trading System - IBKR Order & Position Management
Handles order placement, position tracking, and P/L management

WORKFLOW:
1. Screener identifies A-grade setups
2. Trade Bill creates order parameters
3. This module places orders to IBKR
4. Trade Log automatically pulls filled orders from IBKR
5. Position Management tracks all open positions with live P/L

Order Types Supported:
- Limit Orders (entry at EMA-22)
- Stop Loss Orders (bracket orders)
- Take Profit Orders (at KC upper)
"""

import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import urllib3

# Disable SSL warnings for IBKR Gateway
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# IBKR Client Portal Gateway URL
IBKR_GATEWAY_URL = "https://localhost:5000/v1/api"


def check_ibkr_connection() -> tuple:
    """Check if IBKR Gateway is connected and authenticated"""
    try:
        response = requests.get(
            f"{IBKR_GATEWAY_URL}/iserver/auth/status",
            verify=False,
            timeout=5
        )
        if response.status_code == 200:
            data = response.json()
            if data.get('authenticated'):
                return True, "Connected to IBKR Gateway"
            return False, "Gateway running but not authenticated. Login required."
        return False, f"Gateway error: {response.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "Cannot connect to IBKR Gateway. Start the gateway first."
    except Exception as e:
        return False, f"Connection error: {str(e)}"


def get_account_id() -> Optional[str]:
    """Get the IBKR account ID"""
    try:
        response = requests.get(
            f"{IBKR_GATEWAY_URL}/portfolio/accounts",
            verify=False,
            timeout=10
        )
        if response.status_code == 200:
            accounts = response.json()
            if accounts:
                return accounts[0].get('accountId')
    except Exception as e:
        print(f"Error getting account ID: {e}")
    return None


def get_contract_id(symbol: str) -> Optional[int]:
    """Get IBKR contract ID for a symbol"""
    try:
        # Search for the stock
        response = requests.get(
            f"{IBKR_GATEWAY_URL}/iserver/secdef/search",
            params={'symbol': symbol, 'secType': 'STK'},
            verify=False,
            timeout=10
        )
        if response.status_code == 200:
            results = response.json()
            if results:
                # Return first US stock match
                for r in results:
                    if r.get('sections'):
                        for section in r['sections']:
                            if 'NASDAQ' in section.get('exchange', '') or 'NYSE' in section.get('exchange', ''):
                                return r.get('conid')
                # Fallback to first result
                return results[0].get('conid')
    except Exception as e:
        print(f"Error getting contract ID for {symbol}: {e}")
    return None


def place_bracket_order(
    symbol: str,
    quantity: int,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    order_type: str = 'LMT',
    tif: str = 'GTC'
) -> Dict:
    """
    Place a bracket order (Entry + Stop Loss + Take Profit)
    
    Args:
        symbol: Stock symbol
        quantity: Number of shares
        entry_price: Limit price for entry
        stop_loss: Stop loss price
        take_profit: Take profit price
        order_type: 'LMT' or 'MKT'
        tif: Time in force ('GTC', 'DAY', etc.)
    
    Returns:
        Order result with order IDs
    """
    connected, message = check_ibkr_connection()
    if not connected:
        return {'success': False, 'error': message}
    
    account_id = get_account_id()
    if not account_id:
        return {'success': False, 'error': 'Could not get account ID'}
    
    conid = get_contract_id(symbol)
    if not conid:
        return {'success': False, 'error': f'Could not find contract for {symbol}'}
    
    try:
        # Create bracket order
        order_data = {
            'orders': [{
                'acctId': account_id,
                'conid': conid,
                'orderType': order_type,
                'side': 'BUY',
                'quantity': quantity,
                'price': entry_price,
                'tif': tif,
                'outsideRTH': False,  # Only during market hours
                'cOID': f'ELDER_{symbol}_{datetime.now().strftime("%Y%m%d%H%M%S")}',
                # Attached orders (bracket)
                'attachedOrders': [
                    {
                        # Stop Loss
                        'orderType': 'STP',
                        'side': 'SELL',
                        'quantity': quantity,
                        'price': stop_loss,
                        'tif': 'GTC'
                    },
                    {
                        # Take Profit
                        'orderType': 'LMT',
                        'side': 'SELL',
                        'quantity': quantity,
                        'price': take_profit,
                        'tif': 'GTC'
                    }
                ]
            }]
        }
        
        response = requests.post(
            f"{IBKR_GATEWAY_URL}/iserver/account/{account_id}/orders",
            json=order_data,
            verify=False,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Handle confirmation messages
            if isinstance(result, list) and result:
                first = result[0]
                if first.get('message'):
                    # Need to confirm
                    confirm_response = requests.post(
                        f"{IBKR_GATEWAY_URL}/iserver/reply/{first.get('id')}",
                        json={'confirmed': True},
                        verify=False,
                        timeout=10
                    )
                    if confirm_response.status_code == 200:
                        result = confirm_response.json()
            
            return {
                'success': True,
                'order_id': result[0].get('order_id') if isinstance(result, list) else result.get('order_id'),
                'message': 'Bracket order placed successfully',
                'details': result
            }
        else:
            return {
                'success': False,
                'error': f'Order failed: {response.text}'
            }
            
    except Exception as e:
        return {'success': False, 'error': str(e)}


def place_single_order(
    symbol: str,
    side: str,
    quantity: int,
    price: float,
    order_type: str = 'LMT',
    tif: str = 'GTC'
) -> Dict:
    """
    Place a single order (not bracket)
    
    Args:
        symbol: Stock symbol
        side: 'BUY' or 'SELL'
        quantity: Number of shares
        price: Limit price
        order_type: 'LMT', 'MKT', or 'STP'
        tif: Time in force
    """
    connected, message = check_ibkr_connection()
    if not connected:
        return {'success': False, 'error': message}
    
    account_id = get_account_id()
    if not account_id:
        return {'success': False, 'error': 'Could not get account ID'}
    
    conid = get_contract_id(symbol)
    if not conid:
        return {'success': False, 'error': f'Could not find contract for {symbol}'}
    
    try:
        order_data = {
            'orders': [{
                'acctId': account_id,
                'conid': conid,
                'orderType': order_type,
                'side': side,
                'quantity': quantity,
                'price': price,
                'tif': tif,
                'outsideRTH': False,
                'cOID': f'ELDER_{symbol}_{side}_{datetime.now().strftime("%Y%m%d%H%M%S")}'
            }]
        }
        
        response = requests.post(
            f"{IBKR_GATEWAY_URL}/iserver/account/{account_id}/orders",
            json=order_data,
            verify=False,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Handle confirmation
            if isinstance(result, list) and result and result[0].get('message'):
                confirm_response = requests.post(
                    f"{IBKR_GATEWAY_URL}/iserver/reply/{result[0].get('id')}",
                    json={'confirmed': True},
                    verify=False,
                    timeout=10
                )
                if confirm_response.status_code == 200:
                    result = confirm_response.json()
            
            return {
                'success': True,
                'order_id': result[0].get('order_id') if isinstance(result, list) else result.get('order_id'),
                'message': f'{side} order placed successfully',
                'details': result
            }
        else:
            return {'success': False, 'error': f'Order failed: {response.text}'}
            
    except Exception as e:
        return {'success': False, 'error': str(e)}


def get_open_orders() -> Dict:
    """Get all open/pending orders"""
    connected, message = check_ibkr_connection()
    if not connected:
        return {'success': False, 'error': message, 'orders': []}
    
    try:
        response = requests.get(
            f"{IBKR_GATEWAY_URL}/iserver/account/orders",
            verify=False,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            orders = data.get('orders', [])
            return {
                'success': True,
                'orders': orders,
                'count': len(orders)
            }
        return {'success': False, 'error': response.text, 'orders': []}
        
    except Exception as e:
        return {'success': False, 'error': str(e), 'orders': []}


def cancel_order(order_id: str) -> Dict:
    """Cancel an open order"""
    connected, message = check_ibkr_connection()
    if not connected:
        return {'success': False, 'error': message}
    
    account_id = get_account_id()
    if not account_id:
        return {'success': False, 'error': 'Could not get account ID'}
    
    try:
        response = requests.delete(
            f"{IBKR_GATEWAY_URL}/iserver/account/{account_id}/order/{order_id}",
            verify=False,
            timeout=10
        )
        
        if response.status_code == 200:
            return {'success': True, 'message': f'Order {order_id} cancelled'}
        return {'success': False, 'error': response.text}
        
    except Exception as e:
        return {'success': False, 'error': str(e)}


def get_positions() -> Dict:
    """
    Get all open positions from IBKR
    Returns positions with current market value and P/L
    """
    connected, message = check_ibkr_connection()
    if not connected:
        return {'success': False, 'error': message, 'positions': []}
    
    account_id = get_account_id()
    if not account_id:
        return {'success': False, 'error': 'Could not get account ID', 'positions': []}
    
    try:
        response = requests.get(
            f"{IBKR_GATEWAY_URL}/portfolio/{account_id}/positions/0",
            verify=False,
            timeout=10
        )
        
        if response.status_code == 200:
            positions = response.json()
            
            formatted = []
            for pos in positions:
                formatted.append({
                    'symbol': pos.get('contractDesc', '').split()[0],
                    'contract_desc': pos.get('contractDesc', ''),
                    'conid': pos.get('conid'),
                    'quantity': pos.get('position', 0),
                    'avg_cost': pos.get('avgCost', 0),
                    'avg_price': pos.get('avgPrice', 0),
                    'market_price': pos.get('mktPrice', 0),
                    'market_value': pos.get('mktValue', 0),
                    'unrealized_pnl': pos.get('unrealizedPnl', 0),
                    'realized_pnl': pos.get('realizedPnl', 0),
                    'pnl_percent': round(
                        (pos.get('unrealizedPnl', 0) / (pos.get('avgCost', 1) * pos.get('position', 1))) * 100, 2
                    ) if pos.get('avgCost', 0) > 0 and pos.get('position', 0) != 0 else 0,
                    'currency': pos.get('currency', 'USD')
                })
            
            return {
                'success': True,
                'positions': formatted,
                'count': len(formatted),
                'total_unrealized_pnl': sum(p['unrealized_pnl'] for p in formatted),
                'total_market_value': sum(p['market_value'] for p in formatted)
            }
        return {'success': False, 'error': response.text, 'positions': []}
        
    except Exception as e:
        return {'success': False, 'error': str(e), 'positions': []}


def get_position_alerts(positions: List[Dict], trade_bills: List[Dict]) -> List[Dict]:
    """
    Generate alerts for positions that need attention
    
    Alerts:
    - Price approaching stop loss
    - Price above target (consider taking profit)
    - Position going against (losing more than expected)
    - Time-based (holding too long without progress)
    """
    alerts = []
    
    for pos in positions:
        symbol = pos.get('symbol')
        current_price = pos.get('market_price', 0)
        avg_cost = pos.get('avg_price', 0)
        pnl_percent = pos.get('pnl_percent', 0)
        unrealized_pnl = pos.get('unrealized_pnl', 0)
        
        # Find matching trade bill for stop/target
        matching_bill = None
        for bill in trade_bills:
            if bill.get('symbol') == symbol:
                matching_bill = bill
                break
        
        if matching_bill:
            stop_loss = matching_bill.get('stop_loss', 0)
            target = matching_bill.get('target', 0)
            
            # Alert: Near stop loss
            if stop_loss > 0:
                distance_to_stop = ((current_price - stop_loss) / current_price) * 100
                if distance_to_stop < 2:  # Within 2% of stop
                    alerts.append({
                        'symbol': symbol,
                        'type': 'STOP_APPROACHING',
                        'severity': 'HIGH',
                        'message': f'{symbol}: Price ${current_price:.2f} is {distance_to_stop:.1f}% from stop ${stop_loss:.2f}',
                        'action': 'Consider closing position or tightening stop',
                        'current_price': current_price,
                        'stop_loss': stop_loss
                    })
            
            # Alert: Above target
            if target > 0 and current_price >= target:
                alerts.append({
                    'symbol': symbol,
                    'type': 'TARGET_REACHED',
                    'severity': 'MEDIUM',
                    'message': f'{symbol}: Price ${current_price:.2f} reached target ${target:.2f}',
                    'action': 'Consider taking profits',
                    'current_price': current_price,
                    'target': target
                })
        
        # Alert: Significant loss
        if pnl_percent < -5:
            alerts.append({
                'symbol': symbol,
                'type': 'SIGNIFICANT_LOSS',
                'severity': 'HIGH',
                'message': f'{symbol}: Position down {abs(pnl_percent):.1f}% (${unrealized_pnl:.2f})',
                'action': 'Review position - consider cutting losses',
                'pnl_percent': pnl_percent,
                'unrealized_pnl': unrealized_pnl
            })
        
        # Alert: Strong gain (potential to lock in profits)
        if pnl_percent > 10:
            alerts.append({
                'symbol': symbol,
                'type': 'STRONG_GAIN',
                'severity': 'LOW',
                'message': f'{symbol}: Position up {pnl_percent:.1f}% (${unrealized_pnl:.2f})',
                'action': 'Consider trailing stop or partial profit-taking',
                'pnl_percent': pnl_percent,
                'unrealized_pnl': unrealized_pnl
            })
    
    # Sort by severity
    severity_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
    alerts.sort(key=lambda x: severity_order.get(x.get('severity', 'LOW'), 3))
    
    return alerts


def get_filled_trades(days_back: int = 7) -> Dict:
    """
    Get filled trades from IBKR for auto-populating trade log
    """
    connected, message = check_ibkr_connection()
    if not connected:
        return {'success': False, 'error': message, 'trades': []}
    
    account_id = get_account_id()
    if not account_id:
        return {'success': False, 'error': 'Could not get account ID', 'trades': []}
    
    try:
        response = requests.get(
            f"{IBKR_GATEWAY_URL}/iserver/account/trades",
            verify=False,
            timeout=10
        )
        
        if response.status_code == 200:
            trades = response.json()
            
            formatted = []
            for trade in trades:
                formatted.append({
                    'execution_id': trade.get('execution_id'),
                    'symbol': trade.get('symbol'),
                    'side': trade.get('side'),  # BOT or SLD
                    'quantity': trade.get('size'),
                    'price': trade.get('price'),
                    'execution_time': trade.get('trade_time'),
                    'order_ref': trade.get('order_ref'),
                    'commission': trade.get('commission', 0),
                    'realized_pnl': trade.get('realized_pnl', 0),
                    'account': account_id
                })
            
            return {
                'success': True,
                'trades': formatted,
                'count': len(formatted)
            }
        return {'success': False, 'error': response.text, 'trades': []}
        
    except Exception as e:
        return {'success': False, 'error': str(e), 'trades': []}


def create_trade_from_bill(trade_bill: Dict) -> Dict:
    """
    Create and place order from a Trade Bill
    
    This is the connection between Screener → Trade Bill → IBKR Order
    """
    required_fields = ['symbol', 'entry', 'stop_loss', 'target', 'quantity']
    for field in required_fields:
        if field not in trade_bill:
            return {'success': False, 'error': f'Missing required field: {field}'}
    
    result = place_bracket_order(
        symbol=trade_bill['symbol'],
        quantity=trade_bill['quantity'],
        entry_price=trade_bill['entry'],
        stop_loss=trade_bill['stop_loss'],
        take_profit=trade_bill['target'],
        order_type='LMT',
        tif='GTC'
    )
    
    if result['success']:
        result['trade_bill_id'] = trade_bill.get('id')
        result['symbol'] = trade_bill['symbol']
        result['message'] = f"Order placed for {trade_bill['symbol']}: Entry ${trade_bill['entry']}, Stop ${trade_bill['stop_loss']}, Target ${trade_bill['target']}"
    
    return result


def get_account_summary() -> Dict:
    """Get account summary with buying power and current value"""
    connected, message = check_ibkr_connection()
    if not connected:
        return {'success': False, 'error': message}
    
    account_id = get_account_id()
    if not account_id:
        return {'success': False, 'error': 'Could not get account ID'}
    
    try:
        response = requests.get(
            f"{IBKR_GATEWAY_URL}/portfolio/{account_id}/summary",
            verify=False,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            
            return {
                'success': True,
                'account_id': account_id,
                'net_liquidation': data.get('netliquidation', {}).get('amount', 0),
                'total_cash': data.get('totalcashvalue', {}).get('amount', 0),
                'buying_power': data.get('buyingpower', {}).get('amount', 0),
                'gross_position_value': data.get('grosspositionvalue', {}).get('amount', 0),
                'maintenance_margin': data.get('maintenancemarginreq', {}).get('amount', 0),
                'available_funds': data.get('availablefunds', {}).get('amount', 0),
                'unrealized_pnl': data.get('unrealizedpnl', {}).get('amount', 0),
                'realized_pnl': data.get('realizedpnl', {}).get('amount', 0),
                'currency': data.get('netliquidation', {}).get('currency', 'USD')
            }
        return {'success': False, 'error': response.text}
        
    except Exception as e:
        return {'success': False, 'error': str(e)}


def modify_order(order_id: str, new_price: float = None, new_quantity: int = None) -> Dict:
    """Modify an existing order"""
    connected, message = check_ibkr_connection()
    if not connected:
        return {'success': False, 'error': message}
    
    account_id = get_account_id()
    if not account_id:
        return {'success': False, 'error': 'Could not get account ID'}
    
    try:
        modify_data = {}
        if new_price is not None:
            modify_data['price'] = new_price
        if new_quantity is not None:
            modify_data['quantity'] = new_quantity
        
        if not modify_data:
            return {'success': False, 'error': 'No modifications specified'}
        
        response = requests.post(
            f"{IBKR_GATEWAY_URL}/iserver/account/{account_id}/order/{order_id}",
            json=modify_data,
            verify=False,
            timeout=10
        )
        
        if response.status_code == 200:
            return {'success': True, 'message': f'Order {order_id} modified'}
        return {'success': False, 'error': response.text}
        
    except Exception as e:
        return {'success': False, 'error': str(e)}


def get_market_data(symbol: str) -> Dict:
    """Get current market data for a symbol"""
    connected, message = check_ibkr_connection()
    if not connected:
        return {'success': False, 'error': message}
    
    conid = get_contract_id(symbol)
    if not conid:
        return {'success': False, 'error': f'Could not find contract for {symbol}'}
    
    try:
        # Request snapshot
        response = requests.get(
            f"{IBKR_GATEWAY_URL}/iserver/marketdata/snapshot",
            params={'conids': conid, 'fields': '31,84,85,86,87,88'},
            verify=False,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            if data:
                item = data[0] if isinstance(data, list) else data
                return {
                    'success': True,
                    'symbol': symbol,
                    'conid': conid,
                    'last_price': item.get('31'),
                    'bid': item.get('84'),
                    'ask': item.get('85'),
                    'bid_size': item.get('88'),
                    'ask_size': item.get('87'),
                    'volume': item.get('87')
                }
        return {'success': False, 'error': response.text}
        
    except Exception as e:
        return {'success': False, 'error': str(e)}
