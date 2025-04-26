import os
import sys
import json
from flask import Flask, request, jsonify, session
from flask_session import Session
from flask_cors import CORS 

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import src.config.settings
from src.routes import bp
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.before_request
def log_request_info():
    if request.method == 'POST':
        try:
            logger.debug(f"Content-Type: {request.content_type}")
            
            raw_data = request.get_data().decode('utf-8')
            logger.debug(f"Raw data length: {len(raw_data)}")
            
            # Validate JSON before Flask processes it
            json.loads(raw_data)
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON Parse Error: {str(e)}")
            logger.debug(f"Raw data: {raw_data}")
            return jsonify(error=f"Invalid JSON: {str(e)}"), 400
        except UnicodeDecodeError as e:
            logger.error(f"Encoding Error: {str(e)}")
            return jsonify(error="Invalid character encoding"), 400
# CORS(app, supports_credentials=True)
CORS(app, 
     resources={
         r"/*": {
             "origins": "*",  # Allow specific origin
             "methods": ["GET", "POST", "OPTIONS"],  # Allowed methods
             "allow_headers": ["Content-Type", "Authorization"],      # Allowed headers
             "supports_credentials": True,           # Allow credentials
              "max_age": 3600,                       # Cache preflight response for 1 hour
         }
     })
app.secret_key = os.environ.get("FLASK_SECRET_KEY", os.urandom(24))
app.config['SESSION_TYPE'] = 'filesystem'

Session(app)

app.register_blueprint(bp)

if __name__ == '__main__':
    app.run(debug=True, port=5001)