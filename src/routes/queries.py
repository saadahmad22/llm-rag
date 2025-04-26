import uuid
from flask import jsonify, session, request

from .init_system import rag_chain

def handle_query():
    if not session.get("user_id"):
        session["user_id"] = str(uuid.uuid4())
    try:
        response = request.get_json()
        query = response["query"]
        user_id = session.get("user_id")

        history = rag_chain.get_user_history(user_id)
        response = rag_chain.run(query, history)

        return jsonify({"response": response})

    except Exception as e:
        return jsonify({"error": f"Could not get request as JSON: {e}"}), 400
    
def handle_generate():
    print("here")
    if not session.get("user_id"):
        session["user_id"] = str(uuid.uuid4())
    try:
        response = request.get_json()
        query = response["query"]
        user_id = session.get("user_id")

        history = rag_chain.get_user_history(user_id)
        response = rag_chain.generate(query, history)

        return jsonify({"response": response})

    except Exception as e:
        return jsonify({"error": f"Could not get request as JSON: {e}"}), 400
    
def handle_embed():
    try:
        response = request.get_json()
        query = response["query"]
        response = rag_chain.embed(query)

        return jsonify({"response": response})

    except Exception as e:
        return jsonify({"error": f"Could not get request as JSON: {e}"}), 400