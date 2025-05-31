import os
import sys
from flask import Flask

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.routes import blueprints_for_routes

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", os.urandom(24))

app.register_blueprint(blueprints_for_routes)

if __name__ == '__main__':
    app.run(debug=True, port=5001)