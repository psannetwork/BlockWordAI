# -*- coding: utf-8 -*-
"""
Localization utilities for BlockWordAI
Handles internationalization of console messages and other UI elements
"""

import os
from typing import Dict

class SimpleI18n:
    """
    Simple localization class for console messages
    In a production environment, this would be replaced with proper i18n tools
    """
    
    def __init__(self, default_language='ja_JP'):
        self.default_language = default_language
        self.current_language = default_language
        self.translations = {
            'ja_JP': {
                # Console messages
                "🔄 Loading model from: {model_path}": "🔄 モデルをロード中: {model_path}",
                "✅ Model loaded successfully": "✅ モデルのロードに成功しました",
                "❌ Missing required packages: {e}": "❌ 必要なパッケージがありません: {e}",
                "💡 Install required packages with: pip install -r requirements.txt": "💡 必要なパッケージをインストールしてください: pip install -r requirements.txt",
                "❌ Error loading model: {e}": "❌ モデルのロード中にエラーが発生しました: {e}",
                "❌ ONNX model file not found: {model_file}\nPlease run training first with: python train/train.py": 
                    "❌ ONNXモデルファイルが見つかりません: {model_file}\n最初にトレーニングを実行してください: python train/train.py",
                "❌ Tokenizer configuration not found: {tokenizer_config_file}\nPlease run training first with: python train/train.py": 
                    "❌ トークナイザー設定が見つかりません: {tokenizer_config_file}\n最初にトレーニングを実行してください: python train/train.py",
                "🔍 BlockWordAI - Japanese Toxicity Detection Demo": "🔍 BlockWordAI - 日本語毒性検出デモ",
                "❌ Cannot run demo without a valid model": "❌ 有効なモデルがないためデモを実行できません",
                "🔍 Japanese toxicity detection demo:": "🔍 日本語毒性検出デモ:",
                "\n🎉 Demo completed": "\n🎉 デモ完了",
                "BlockWordAI - Japanese Toxicity Detection": "BlockWordAI - 日本語毒性検出",
                "Usage:": "使用法:",
                "  python app.py --demo                    # Run demonstration": "  python app.py --demo                    # デモを実行",
                "  python app.py --predict \"TEXT\"         # Predict toxicity for text": "  python app.py --predict \"TEXT\"         # テキストの毒性を予測",
                "Environment variables:": "環境変数:",
                "  BLOCKWORDAI_MODEL_PATH    Path to trained model (default: ./models/latest)": "  BLOCKWORDAI_MODEL_PATH    トレーニング済みモデルへのパス (デフォルト: ./models/latest)",
                "To train a new model: python train/train.py": "新しいモデルをトレーニングするには: python train/train.py",
                "To test the model: python train/test.py": "モデルをテストするには: python train/test.py",
                # API messages
                "BlockWordAI - Japanese Toxicity Detection API": "BlockWordAI - 日本語毒性検出API",
                "Single text toxicity detection": "単一テキストの毒性検出",
                "Batch toxicity detection": "一括毒性検出",
                "Health check": "ヘルスチェック",
                "Missing 'text' field in request body": "'text' フィールドがリクエストボディにありません",
                "'text' field must be a string": "'text' フィールドは文字列である必要があります",
                "'text' field cannot be empty": "'text' フィールドは空にできません",
                "Internal server error: %(error)s": "内部サーバーエラー: %(error)s",
                "Missing 'texts' field in request body": "'texts' フィールドがリクエストボディにありません",
                "'texts' field must be a list": "'texts' フィールドはリストである必要があります",
                "'texts' list cannot be empty": "'texts' リストは空にできません",
                "All predictions failed: %(error_list)s": "すべての予測に失敗しました: %(error_list)s",
                "healthy": "正常",
                "unhealthy": "異常"
            }
        }
    
    def set_language(self, lang_code: str):
        """Set the current language for translations"""
        self.current_language = lang_code
    
    def gettext(self, message: str, **kwargs) -> str:
        """Translate a message to the current language"""
        # Get translations for current language, fallback to default
        lang_translations = self.translations.get(self.current_language, {})
        if not lang_translations:
            lang_translations = self.translations.get(self.default_language, {})
        
        # Get the translated message, fallback to original
        translated = lang_translations.get(message, message)
        
        # Apply any formatting with the kwargs
        try:
            return translated % kwargs
        except (TypeError, ValueError):
            # If formatting fails, return the translated string as is
            return translated

# Create a global instance
i18n = SimpleI18n()

# Convenience function to match Flask-Babel's gettext
def gettext(message: str, **kwargs) -> str:
    """Convenience function to translate messages"""
    return i18n.gettext(message, **kwargs)

# Set language based on environment variable or default to Japanese
def setup_language():
    """Initialize language settings"""
    lang = os.getenv('BLOCKWORDAI_LANGUAGE', 'ja_JP')
    i18n.set_language(lang)