from flask import Blueprint

from .queries import handle_query, handle_query_llm, handle_embed

blueprints_for_routes = Blueprint('main', __name__)

# Register the route with the blueprint
blueprints_for_routes.add_url_rule('/query', view_func=handle_query, methods=['POST'])
'''Get call for `/query`'''
#blueprints_for_routes.add_url_rule('/query', view_func=handle_query)

blueprints_for_routes.add_url_rule('/query_llm', view_func=handle_query_llm, methods=['POST'])
'''Get call for `/query_llm`'''
# blueprints_for_routes.add_url_rule('/query_llm', view_func=handle_query_llm)

blueprints_for_routes.add_url_rule('/embed', view_func=handle_embed, methods=['POST'])
'''Get call for `/embed`'''
# blueprints_for_routes.add_url_rule('/embed', view_func=handle_embed)





# IMPORTS HERE from `app.py`