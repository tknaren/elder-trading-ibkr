"""
Elder Trading System - Local Application v2
Enhanced with connected workflow: Screener → Trade Bill → IBKR → Trade Log → Positions

Run with: python app.py
Access at: http://localhost:5001
IBKR Gateway: https://localhost:5000

Features v2:
- Screen 1 as mandatory gate
- New high-scoring rules (divergence, false breakout, kangaroo tail, etc.)
- Elder Entry/Stop/Target calculations
- IBKR order placement from Trade Bills
- Auto-sync Trade Log from IBKR
- Live position management with P/L tracking
"""

from routes.api_v2 import api_v2
from routes.api import api
from models.database import Database, get_database
from flask import Flask, render_template
import os
import sys

# Add backend directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def create_app():
    """Create and configure the Flask application"""
    app = Flask(__name__)

    # Configuration for local development
    app.config['SECRET_KEY'] = 'elder-trading-local-dev-key'

    # Database path - local data folder
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)
    db_path = os.path.join(data_dir, 'elder_trading.db')
    app.config['DATABASE_PATH'] = db_path

    # Set environment variable for database module
    os.environ['DATABASE_PATH'] = db_path

    # Register API blueprints
    app.register_blueprint(api, url_prefix='/api')      # Original API
    app.register_blueprint(api_v2, url_prefix='/api/v2')  # Enhanced v2 API

    # Initialize database (creates tables and default data)
    with app.app_context():
        db = Database(db_path)
        app.config['DB'] = db

        # Run migrations for v2 features
        try:
            from models.migrate_v2 import migrate_database
            migrate_database(db_path)
        except Exception as e:
            print(f"Migration note: {e}")

    # Serve frontend
    @app.route('/')
    def index():
        return render_template('index.html')

    return app


# Create application instance
app = create_app()

if __name__ == '__main__':
    print("\n" + "="*50)
    print("  Elder Trading System - Local Server")
    print("="*50)
    print(f"\n  Open in browser: http://localhost:5001")
    print(f"  Data stored in: ./data/elder_trading.db")
    print(f"  IBKR Gateway: https://localhost:5000")
    print(f"\n  Press Ctrl+C to stop the server")
    print("="*50 + "\n")

    app.run(
        host='0.0.0.0',
        port=5001,        # Changed from 5000 to avoid conflict with IBKR Gateway
        debug=True
    )
