# -*- coding: utf-8 -*-
"""
BlockWordAI API Server
Provides a web API for the Japanese toxicity detection model.
"""

from flask import Flask, request, Response
from flask_babel import Babel, gettext
import json
import os
import sys

# パス設定（app.py をインポートするため）
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import predict_toxicity, batch_predict_toxicity, load_model

app = Flask(__name__)

# Flask設定
app.config['JSON_AS_ASCII'] = False 
app.config['BABEL_DEFAULT_LOCALE'] = 'ja_JP'
app.config['BABEL_TRANSLATION_DIRECTORIES'] = 'translations'

# 言語選択関数
def get_locale():
    lang = request.headers.get('Accept-Language')
    if not lang:
        lang = request.args.get('lang', 'ja')
    return 'ja_JP' if lang.startswith('ja') else 'en_US'

# Flask-Babel v3 以降の初期化
babel = Babel(app, locale_selector=get_locale)

# モデルのロード
tokenizer, session = None, None

def initialize_model():
    """モデルを初期化（ロード）"""
    global tokenizer, session
    tokenizer, session = load_model()
    if tokenizer is None or session is None:
        print("❌ モデルのロードに失敗しました")
        return False
    print("✅ モデルのロードに成功しました")
    return True

# サーバ起動時にモデルを読み込み
with app.app_context():
    initialize_model()

def json_response(data, status=200):
    """UTF-8でJSONレスポンスを返す関数"""
    return Response(
        json.dumps(data, ensure_ascii=False),
        status=status,
        mimetype='application/json; charset=utf-8'
    )

@app.route('/')
def home():
    """ホーム（API情報）"""
    return json_response({
        "message": gettext("BlockWordAI - 日本語有害表現検出API"),
        "version": "1.0.0",
        "endpoints": {
            "/api/toxicity": gettext("単一テキストの有害性検出"),
            "/api/batch-toxicity": gettext("複数テキストの有害性検出"),
            "/api/health": gettext("ヘルスチェック")
        }
    })

@app.route('/api/toxicity', methods=['POST'])
def toxicity_detection():
    """単一テキストの有害性検出API"""
    global tokenizer, session
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return json_response({"error": gettext("'text' フィールドが不足しています")}, 400)

        text = data['text']
        if not isinstance(text, str):
            return json_response({"error": gettext("'text' フィールドは文字列である必要があります")}, 400)
        if len(text) == 0:
            return json_response({"error": gettext("'text' フィールドは空ではいけません")}, 400)

        result = predict_toxicity(text, tokenizer, session)
        if "error" in result:
            return json_response(result, 500)

        return json_response(result, 200)

    except Exception as e:
        return json_response({"error": gettext("内部サーバーエラー: %(error)s", error=str(e))}, 500)

@app.route('/api/batch-toxicity', methods=['POST'])
def batch_toxicity_detection():
    """複数テキストの有害性検出API"""
    global tokenizer, session
    try:
        data = request.get_json()
        if not data or 'texts' not in data:
            return json_response({"error": gettext("'texts' フィールドが不足しています")}, 400)

        texts = data['texts']
        if not isinstance(texts, list):
            return json_response({"error": gettext("'texts' フィールドはリストである必要があります")}, 400)
        if len(texts) == 0:
            return json_response({"error": gettext("'texts' リストは空ではいけません")}, 400)

        results = batch_predict_toxicity(texts, tokenizer, session)
        if all("error" in result for result in results):
            error_msgs = [r['error'] for r in results if 'error' in r][:1]
            return json_response({"error": gettext("すべての予測が失敗しました: %(error_list)s", error_list=str(error_msgs))}, 500)

        return json_response({"results": results}, 200)

    except Exception as e:
        return json_response({"error": gettext("内部サーバーエラー: %(error)s", error=str(e))}, 500)

@app.route('/api/health', methods=['GET'])
def health_check():
    """モデルの状態を確認するヘルスチェック"""
    global tokenizer, session
    status = {
        "status": gettext("正常" if tokenizer and session else "異常"),
        "model_loaded": tokenizer is not None and session is not None
    }

    if not status["model_loaded"]:
        initialize_model()
        status["model_loaded"] = tokenizer is not None and session is not None
        if not status["model_loaded"]:
            status["status"] = gettext("異常")

    return json_response(status, 200)

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8000))
    host = os.getenv('HOST', '0.0.0.0')
    app.run(host=host, port=port, debug=False)
