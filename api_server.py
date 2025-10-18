# -*- coding: utf-8 -*-
"""
BlockWordAI API Server
Provides a web API for the Japanese toxicity detection model.
"""

from flask import Flask, request, jsonify
from flask_babel import Babel, gettext
import os
import sys

# Add the app module to the path so we can import the functions
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import predict_toxicity, batch_predict_toxicity, load_model

app = Flask(__name__)

# Configuration for Babel
app.config['BABEL_DEFAULT_LOCALE'] = 'ja_JP'
app.config['BABEL_TRANSLATION_DIRECTORIES'] = 'translations'

# Define locale selection function
def get_locale():
    lang = request.headers.get('Accept-Language')
    if not lang:
        lang = request.args.get('lang', 'ja')
    return 'ja_JP' if lang.startswith('ja') else 'en_US'

# ✅ Correct initialization for Flask-Babel v3+
babel = Babel(app, locale_selector=get_locale)

# Load the model once when the server starts
tokenizer, session = None, None

def initialize_model():
    global tokenizer, session
    tokenizer, session = load_model()
    if tokenizer is None or session is None:
        print("❌ Error: Failed to load model at startup")
        return False
    return True

# Initialize model on startup
with app.app_context():
    success = initialize_model()
    if not success:
        print("⚠️ Warning: Model failed to load at startup. API endpoints may not work until model is available.")

@app.route('/')
def home():
    return jsonify({
        "message": gettext("BlockWordAI - Japanese Toxicity Detection API"),
        "version": "1.0.0",
        "endpoints": {
            "/api/toxicity": gettext("Single text toxicity detection"),
            "/api/batch-toxicity": gettext("Batch toxicity detection"),
            "/api/health": gettext("Health check")
        }
    })

@app.route('/api/toxicity', methods=['POST'])
def toxicity_detection():
    global tokenizer, session
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": gettext("Missing 'text' field in request body")}), 400
        text = data['text']
        if not isinstance(text, str):
            return jsonify({"error": gettext("'text' field must be a string")}), 400
        if len(text) == 0:
            return jsonify({"error": gettext("'text' field cannot be empty")}), 400
        result = predict_toxicity(text, tokenizer, session)
        if "error" in result:
            return jsonify(result), 500
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": gettext("Internal server error: %(error)s", error=str(e))}), 500

@app.route('/api/batch-toxicity', methods=['POST'])
def batch_toxicity_detection():
    global tokenizer, session
    try:
        data = request.get_json()
        if not data or 'texts' not in data:
            return jsonify({"error": gettext("Missing 'texts' field in request body")}), 400
        texts = data['texts']
        if not isinstance(texts, list):
            return jsonify({"error": gettext("'texts' field must be a list")}), 400
        if len(texts) == 0:
            return jsonify({"error": gettext("'texts' list cannot be empty")}), 400
        results = batch_predict_toxicity(texts, tokenizer, session)
        if all("error" in result for result in results):
            error_msgs = [r['error'] for r in results if 'error' in r][:1]
            return jsonify({"error": gettext("All predictions failed: %(error_list)s", error_list=str(error_msgs))}), 500
        return jsonify({"results": results}), 200
    except Exception as e:
        return jsonify({"error": gettext("Internal server error: %(error)s", error=str(e))}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    global tokenizer, session
    status = {
        "status": gettext("healthy"),
        "model_loaded": tokenizer is not None and session is not None
    }
    if not status["model_loaded"]:
        initialize_model()
        status["model_loaded"] = tokenizer is not None and session is not None
        if not status["model_loaded"]:
            status["status"] = gettext("unhealthy")
    return jsonify(status), 200

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8000))
    host = os.getenv('HOST', '0.0.0.0')
    app.run(host=host, port=port, debug=False)
