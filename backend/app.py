"""
Elder Trading System - Local Application
Run with: python app.py
Access at: http://localhost:5000
"""

import os
import sys

# Add backend directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template
from models.database import Database, get_database
from routes.api import api

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
    
    # Register API blueprint
    app.register_blueprint(api, url_prefix='/api')
    
    # Initialize database (creates tables and default data)
    with app.app_context():
        db = Database(db_path)
        app.config['DB'] = db
    
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
    print(f"\n  📊 Open in browser: http://localhost:5001")
    print(f"  📁 Data stored in: ./data/elder_trading.db")
    print(f"  🔗 IBKR Gateway: https://localhost:5000")
    print(f"\n  Press Ctrl+C to stop the server")
    print("="*50 + "\n")
    
    app.run(
        host='0.0.0.0',
        port=5001,        # Changed from 5000 to avoid conflict with IBKR Gateway
        debug=True
    )
