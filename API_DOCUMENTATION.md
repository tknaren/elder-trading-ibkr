# API Documentation - Trade Bill & Account Endpoints

## Overview

All endpoints are prefixed with `/api` and require proper JSON formatting.

---

## Trade Bills Endpoints

### Create Trade Bill

```
POST /api/trade-bills
Content-Type: application/json

Request Body:
{
  "ticker": "AAPL",
  "current_market_price": 150.25,
  "entry_price": 149.50,
  "stop_loss": 145.00,
  "target_price": 160.00,
  "quantity": 10.5,
  "upper_channel": 157.50,
  "lower_channel": 142.50,
  "overnight_charges": 0,
  "risk_per_share": 4.50,
  "position_size": 1569.75,
  "risk_percent": 1.5,
  "channel_height": 15.00,
  "potential_gain": 110.25,
  "target_1_1_c": 160.00,
  "target_1_2_b": 170.00,
  "target_1_3_a": 180.00,
  "risk_amount_currency": 47.25,
  "reward_amount_currency": 110.25,
  "risk_reward_ratio": 2.33,
  "break_even": 149.50,
  "trailing_stop": 151.00,
  "is_filled": false,
  "stop_entered": false,
  "target_entered": false,
  "journal_entered": false,
  "comments": "Strong breakout setup on daily chart"
}

Response:
{
  "success": true,
  "id": 1,
  "message": "Trade Bill for AAPL created successfully"
}
```

### List Trade Bills

```
GET /api/trade-bills?status=active

Query Parameters:
- status (optional): "active", "filled", "closed"

Response:
[
  {
    "id": 1,
    "user_id": 1,
    "ticker": "AAPL",
    "entry_price": 149.50,
    "stop_loss": 145.00,
    "target_price": 160.00,
    "quantity": 10.5,
    "risk_amount_currency": 47.25,
    "reward_amount_currency": 110.25,
    "risk_reward_ratio": 2.33,
    "is_filled": false,
    "status": "active",
    "created_at": "2024-12-06T10:30:00",
    "updated_at": "2024-12-06T10:30:00"
  }
]
```

### Get Specific Trade Bill

```
GET /api/trade-bills/{id}

Response:
{
  "id": 1,
  "user_id": 1,
  "ticker": "AAPL",
  "current_market_price": 150.25,
  "entry_price": 149.50,
  "stop_loss": 145.00,
  "target_price": 160.00,
  "quantity": 10.5,
  "upper_channel": 157.50,
  "lower_channel": 142.50,
  "target_pips": 10.50,
  "stop_loss_pips": 4.50,
  "max_qty_for_risk": 42.35,
  "overnight_charges": 0,
  "risk_per_share": 4.50,
  "position_size": 1569.75,
  "risk_percent": 1.5,
  "channel_height": 15.00,
  "potential_gain": 110.25,
  "target_1_1_c": 160.00,
  "target_1_2_b": 170.00,
  "target_1_3_a": 180.00,
  "risk_amount_currency": 47.25,
  "reward_amount_currency": 110.25,
  "risk_reward_ratio": 2.33,
  "break_even": 149.50,
  "trailing_stop": 151.00,
  "is_filled": false,
  "stop_entered": false,
  "target_entered": false,
  "journal_entered": false,
  "comments": "Strong breakout setup on daily chart",
  "status": "active",
  "created_at": "2024-12-06T10:30:00",
  "updated_at": "2024-12-06T10:30:00"
}
```

### Update Trade Bill

```
PUT /api/trade-bills/{id}
Content-Type: application/json

Request Body (send only fields to update):
{
  "entry_price": 149.00,
  "is_filled": true,
  "stop_entered": true,
  "comments": "Updated entry price"
}

Response:
{
  "success": true,
  "message": "Trade Bill updated"
}
```

### Delete Trade Bill

```
DELETE /api/trade-bills/{id}

Response:
{
  "success": true,
  "message": "Trade Bill deleted"
}
```

### Calculate Trade Metrics

```
POST /api/trade-bills/calculate
Content-Type: application/json

Request Body:
{
  "entry_price": 149.50,
  "stop_loss": 145.00,
  "target_price": 160.00,
  "quantity": 10.5,
  "account_capital": 10000,
  "risk_percent": 2
}

Response:
{
  "risk_per_share": 4.50,
  "stop_loss_pips": 4.50,
  "target_pips": 10.50,
  "potential_gain": 110.25,
  "risk_reward_ratio": 2.33,
  "max_qty_for_risk": 42.35,
  "position_size": 1569.75,
  "risk_amount_currency": 47.25,
  "break_even": 149.50
}
```

---

## Account Information Endpoints

### Get Account Info

```
GET /api/account/info

Response:
{
  "id": 1,
  "user_id": 1,
  "account_name": "ISA Account",
  "market": "US",
  "trading_capital": 10000,
  "risk_per_trade": 2.0,
  "max_monthly_drawdown": 6.0,
  "target_rr": 2.0,
  "max_open_positions": 5,
  "currency": "GBP",
  "broker": "Trading212",
  "no_of_open_positions": 2,
  "money_locked_in_positions": 3000,
  "money_remaining_to_risk": 300,
  "risk_percent_remaining": 3.0,
  "created_at": "2024-12-01T10:00:00",
  "updated_at": "2024-12-06T15:30:00"
}
```

### Update Account Info

```
PUT /api/account/info
Content-Type: application/json

Request Body (send only fields to update):
{
  "account_name": "ISA Account",
  "trading_capital": 12000,
  "risk_per_trade": 2.5,
  "max_monthly_drawdown": 6.0,
  "target_rr": 2.0,
  "currency": "GBP"
}

Response:
{
  "success": true,
  "message": "Account information updated"
}
```

---

## Data Types

### Fields

- **id**: Integer (auto-generated)
- **user_id**: Integer (defaults to 1)
- **ticker**: String (e.g., "AAPL")
- **prices**: Float (e.g., 149.50)
- **quantity**: Float (supports fractional, e.g., 10.5)
- **booleans**: true/false (e.g., is_filled: true)
- **status**: String ("active", "filled", "closed")
- **timestamps**: ISO 8601 format (e.g., "2024-12-06T10:30:00")

### Calculated Fields (Read-Only)

- `risk_per_share`: Calculated as |entry_price - stop_loss|
- `target_pips`: Calculated as |target_price - entry_price|
- `position_size`: Calculated as quantity × entry_price
- `risk_amount_currency`: Calculated as quantity × risk_per_share
- `potential_gain`: Calculated as quantity × target_pips
- `risk_reward_ratio`: Calculated as potential_gain / risk_amount
- `break_even`: Same as entry_price
- `max_qty_for_risk`: Calculated based on account risk allocation

---

## Error Responses

### 400 Bad Request

```json
{
  "success": false,
  "error": "Missing required fields: entry_price, stop_loss"
}
```

### 404 Not Found

```json
{
  "error": "Trade Bill not found"
}
```

### 500 Internal Server Error

```json
{
  "success": false,
  "error": "Database error: [error details]"
}
```

---

## Usage Examples

### JavaScript/Frontend

```javascript
// Create Trade Bill
const response = await fetch('/api/trade-bills', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    ticker: 'AAPL',
    entry_price: 149.50,
    stop_loss: 145.00,
    target_price: 160.00,
    quantity: 10.5,
    risk_per_share: 4.50,
    risk_amount_currency: 47.25,
    reward_amount_currency: 110.25,
    risk_reward_ratio: 2.33
  })
});
const result = await response.json();
console.log('Created Trade Bill:', result.id);

// Get All Trade Bills
const bills = await fetch('/api/trade-bills').then(r => r.json());
console.log('Trade Bills:', bills);

// Calculate Metrics
const metrics = await fetch('/api/trade-bills/calculate', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    entry_price: 149.50,
    stop_loss: 145.00,
    target_price: 160.00,
    quantity: 10.5,
    account_capital: 10000,
    risk_percent: 2
  })
}).then(r => r.json());
console.log('Metrics:', metrics);

// Update Account
await fetch('/api/account/info', {
  method: 'PUT',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    trading_capital: 12000,
    risk_per_trade: 2.5
  })
});

// Get Account Info
const account = await fetch('/api/account/info').then(r => r.json());
console.log('Account:', account);
```

### cURL Examples

```bash
# Create Trade Bill
curl -X POST http://localhost:5000/api/trade-bills \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "AAPL",
    "entry_price": 149.50,
    "stop_loss": 145.00,
    "target_price": 160.00,
    "quantity": 10.5,
    "risk_per_share": 4.50,
    "risk_amount_currency": 47.25,
    "reward_amount_currency": 110.25,
    "risk_reward_ratio": 2.33
  }'

# List Trade Bills
curl -X GET http://localhost:5000/api/trade-bills

# Get Account Info
curl -X GET http://localhost:5000/api/account/info

# Update Account
curl -X PUT http://localhost:5000/api/account/info \
  -H "Content-Type: application/json" \
  -d '{
    "trading_capital": 12000,
    "risk_per_trade": 2.5
  }'
```

---

## Notes

- All timestamps are in UTC/ISO 8601 format
- All monetary values are in the account's specified currency
- Fractional quantities are supported for all markets
- Quantity can be decimal (e.g., 10.5 shares)
- All calculations use floating-point arithmetic
- Trade bills are user-specific and isolated by user_id

---

## Rate Limiting

Currently no rate limiting implemented. For production use, recommend:

- Max 10 requests per second per IP
- Max 1000 trade bills per user
- Cache account info for 1 second

---

## Authentication

Currently uses implicit user_id = 1. For production, implement:

- JWT token validation
- Per-user data isolation
- Account ownership verification
- Permission-based access control
