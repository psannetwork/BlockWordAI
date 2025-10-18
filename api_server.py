# -*- coding: utf-8 -*-
"""
BlockWordAI API Server
日本語トキシック判定モデルのAPIサーバー
"""

from flask import Flask, request, jsonify
from flask_babel import Babel, gettext
import os
import sys

# app.py をインポートできるようにパスを追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import predict_toxicity, batch_predict_toxicity, load_model

# Flask アプリケーションの作成
app = Flask(__name__)

# JSON出力時に日本語をUnicodeエスケープしない設定
app.config['JSON_AS_ASCII'] = False  

# Babel（多言語対応）の設定
app.config['BABEL_DEFAULT_LOCALE'] = 'ja_JP'
app.config['BABEL_TRANSLATION_DIRECTORIES'] = 'translations'

# 使用言語を決定する関数（ヘッダーやクエリから判定）
def get_locale():
    lang = request.headers.get('Accept-Language')
    if not lang:
        lang = request.args.get('lang', 'ja')
    return 'ja_JP' if lang.startswith('ja') else 'en_US'

# Flask-Babel v3 以降の初期化方法
babel = Babel(app, locale_selector=get_locale)

# モデルをグローバルに保持する変数
tokenizer, session = None, None

# モデルを初期化（ロード）する関数
def initialize_model():
    global tokenizer, session
    tokenizer, session = load_model()
    if tokenizer is None or session is None:
        print("❌ モデルの読み込みに失敗しました")
        return False
    print("✅ モデルの読み込みに成功しました")
    return True

# アプリ起動時にモデルをロード
with app.app_context():
    success = initialize_model()
    if not success:
        print("⚠️ モデルの読み込みに失敗しました。APIの一部機能が動作しない可能性があります。")

# --- ルートエンドポイント（API概要） ---
@app.route('/')
def home():
    return jsonify({
        "message": str(gettext("BlockWordAI - 日本語トキシック検出API")),
        "version": "1.0.0",
        "endpoints": {
            "/api/toxicity": str(gettext("単一テキストのトキシック検出")),
            "/api/batch-toxicity": str(gettext("複数テキストのトキシック検出")),
            "/api/health": str(gettext("APIの動作確認"))
        }
    })

# --- 単一テキストのトキシック検出 ---
@app.route('/api/toxicity', methods=['POST'])
def toxicity_detection():
    global tokenizer, session
    try:
        data = request.get_json()

        # 入力チェック
        if not data or 'text' not in data:
            return jsonify({"error": str(gettext("リクエストに 'text' フィールドがありません"))}), 400
        
        text = data['text']

        if not isinstance(text, str):
            return jsonify({"error": str(gettext("'text' フィールドは文字列でなければなりません"))}), 400
        
        if len(text.strip()) == 0:
            return jsonify({"error": str(gettext("'text' フィールドが空です"))}), 400

        # トキシック判定を実行
        result = predict_toxicity(text, tokenizer, session)

        if "error" in result:
            return jsonify(result), 500
        
        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": f"サーバー内部エラー: {str(e)}"}), 500

# --- 複数テキストのトキシック検出 ---
@app.route('/api/batch-toxicity', methods=['POST'])
def batch_toxicity_detection():
    global tokenizer, session
    try:
        data = request.get_json()

        # 入力チェック
        if not data or 'texts' not in data:
            return jsonify({"error": str(gettext("リクエストに 'texts' フィールドがありません"))}), 400
        
        texts = data['texts']

        if not isinstance(texts, list):
            return jsonify({"error": str(gettext("'texts' フィールドはリストでなければなりません"))}), 400
        
        if len(texts) == 0:
            return jsonify({"error": str(gettext("'texts' リストが空です"))}), 400

        # バッチでトキシック判定を実行
        results = batch_predict_toxicity(texts, tokenizer, session)

        # 全てエラーだった場合
        if all("error" in result for result in results):
            error_msgs = [r['error'] for r in results if 'error' in r]
            return jsonify({"error": f"すべての予測に失敗しました: {error_msgs}"}), 500
        
        return jsonify({"results": results}), 200

    except Exception as e:
        return jsonify({"error": f"サーバー内部エラー: {str(e)}"}), 500

# --- APIの動作確認（ヘルスチェック） ---
@app.route('/api/health', methods=['GET'])
def health_check():
    global tokenizer, session

    # モデルがロード済みか確認
    model_loaded = tokenizer is not None and session is not None

    status = {
        "status": "正常" if model_loaded else "異常",
        "model_loaded": model_loaded
    }

    # モデルが未ロードの場合、再読み込みを試行
    if not model_loaded:
        initialize_model()
        status["model_loaded"] = tokenizer is not None and session is not None
        if not status["model_loaded"]:
            status["status"] = "異常"

    return jsonify(status), 200

# --- メイン ---
if __name__ == '__main__':
    port = int(os.getenv('PORT', 8000))
    host = os.getenv('HOST', '0.0.0.0')
    app.run(host=host, port=port, debug=False)
