# Quick Start Guide - Interactive Features

## New Tabs

### 💼 Trade Bill (Create/Edit)

**Purpose**: Create detailed trade setups with risk calculations

**Steps**:

1. Fill in ticker, entry price, stop loss, target
2. Enter quantity (supports fractional)
3. Click "Calculate Metrics" button
4. Review auto-calculated fields
5. Add comments if needed
6. Click "Save Trade Bill"

**What gets calculated**:

- Risk per share
- Position size
- Risk amount & percentage
- Potential gain
- Risk:Reward ratio
- Max quantity for your risk tolerance

---

### 📋 Trade Bills (View/Manage)

**Purpose**: List and manage all your saved trade setups

**Features**:

- View all trade bills in a table
- See entry, stop loss, target at a glance
- View risk amount and R:R ratio
- Edit trade bill (click Edit button)
- Delete trade bill (click Delete button)
- Create new trade bill (click Create button)

**Color codes**:

- 🟢 Green = Good R:R ratio (≥2:1)
- 🟡 Yellow = Acceptable R:R ratio
- ✓ Filled = Trade has been filled

---

### 💰 Account (Dashboard)

**Purpose**: Monitor your trading account and risk allocation

**Key Metrics**:

- **Account Size**: Your total trading capital
- **Risk per Trade**: Max you risk on each trade
- **Open Positions**: How many active trades
- **Money Locked**: Capital tied up in positions
- **Money Remaining**: How much more you can risk
- **Position Slots**: How many more trades you can open

**Risk Bar**: Shows how much of your monthly drawdown limit you've used

---

### ⚙️ Settings (Configuration)

**Purpose**: Configure your account settings

**What to set up**:

1. **Account Name**: Your account nickname
2. **Trading Capital**: Your total account size
3. **Currency**: GBP, USD, or INR
4. **Risk % per Trade**: Typically 1-3%
5. **Max Monthly Drawdown**: Typically 5-6%
6. **Target R:R**: Your minimum reward-to-risk ratio

*All changes save to database immediately*

---

## Workflow: From Screener to Trade

### Method 1: Create from Scanner

1. Go to **Screener** tab
2. Run scan or find stock you like
3. Click "Details" button
4. Click **"Create Trade Bill"** button
5. Form opens with ticker and price pre-filled
6. Continue with normal Trade Bill creation

### Method 2: Manual Entry

1. Go to **Trade Bill** tab
2. Enter ticker manually
3. Fill in prices (entry, stop loss, target)
4. Enter quantity
5. Click **"Calculate Metrics"**
6. Review calculations
7. Click **"Save Trade Bill"**

### Method 3: Edit Existing

1. Go to **Trade Bills** tab
2. Find the trade you want to edit
3. Click **"Edit"** button
4. Make your changes
5. Click **"Save Trade Bill"**
6. Changes are saved

---

## Auto-Calculated Fields (Read-Only)

These fields automatically calculate when you fill in the required fields:

### Required for Calculations

- Entry Price
- Stop Loss Price
- Target Price
- Quantity

### Auto-Calculated Fields

- **Risk per Share** = |Entry - Stop Loss|
- **Position Size** = Quantity × Entry Price
- **Risk Amount** = Quantity × Risk per Share
- **Target Pips** = |Target - Entry|
- **Potential Gain** = Quantity × Target Pips
- **R:R Ratio** = Potential Gain ÷ Risk Amount
- **Break Even** = Entry Price

---

## Understanding Your Numbers

### Risk per Share (RPS)

How much you lose per share if stop loss is hit.

- Entry: $150, Stop Loss: $145 = RPS $5

### Position Size

Total money put into the trade.

- 10 shares @ $150 = $1,500 position

### Risk Amount

Total money you'll lose if stopped out.

- 10 shares × $5 RPS = $50 risk

### Risk %

Percentage of your account you're risking.

- $50 risk ÷ $10,000 capital = 0.5% risk

### R:R Ratio

How much you make vs. how much you risk.

- Make $150, Risk $50 = 3:1 ratio
- Good target: ≥ 2:1

---

## Tips for Success

### Position Sizing

1. Decide your risk % (usually 1-3%)
2. Calculate: Risk % × Capital = Max Risk Amount
3. Example: 2% × $10,000 = $200 max risk
4. Use "Max Qty for Risk" field to see how many shares you can buy

### Risk Management

- Never risk more than 2% per trade
- Sum of all open positions shouldn't exceed 6% monthly loss limit
- Monitor "Money Remaining to Risk" in Account tab
- Know when to stop trading if you hit your limit

### Trade Setup

- Always enter your entry, stop loss, and target BEFORE buying
- Calculate metrics to verify your R:R is at least 2:1
- Save the trade bill BEFORE placing your order
- Update checkboxes as you progress:
  - ✓ "Filled" when you buy
  - ✓ "Stop Entered" when stop order is in
  - ✓ "Target Entered" when target order is in
  - ✓ "Journal Entered" when you log the trade

---

## Currency Support

The app supports GBP (£), USD ($), and INR (₹).

To change currency:

1. Go to **Settings**
2. Select your currency
3. Click Save
4. All displays update immediately

---

## Account Metrics Explained

### Money Locked in Positions

How much of your capital is currently in open trades.

### Money Remaining to Risk

How much more capital you can risk before hitting your monthly drawdown limit.

- Formula: (Max Drawdown % × Capital) - Money Locked

### Risk % Remaining

The percentage of your capital still available for trading.

### Position Slots

Number of additional positions you can open (based on max_open_positions setting, typically 5).

---

## Troubleshooting

### Fields not calculating?

- Make sure entry, stop loss, target, and quantity are all filled in
- Click "Calculate Metrics" button
- All read-only fields should now show values

### Currency symbols wrong?

- Check Settings tab
- Verify correct currency is selected
- Click Save
- Refresh page if needed

### Trade Bill not saving?

- Check that all required fields are filled
- Look for error message in red toast notification
- Try again, or contact support if issue persists

### Account info not showing?

- Make sure settings are configured first
- Go to Settings and save your details
- Then go to Account tab

---

## Keyboard Shortcuts

*Coming soon - Tab between fields with Tab key, Enter to save*

---

## Data Storage

All your data is stored locally in: `./data/elder_trading.db`

This SQLite database contains:

- Account settings
- All trade bills
- Trade journal entries
- Watchlists
- Checklist items

**Backup your data regularly!**

---

Need help? All calculations are shown in the form - review them to understand your position better.
