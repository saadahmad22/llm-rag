from flask import Blueprint

from .queries import handle_query, handle_generate, handle_embed

bp = Blueprint('main', __name__)

# Register the route with the blueprint
bp.add_url_rule('/query', view_func=handle_query, methods=['POST'])

bp.add_url_rule('/generate', view_func=handle_generate, methods=['POST'])

bp.add_url_rule('/embed', view_func=handle_embed, methods=['POST'])