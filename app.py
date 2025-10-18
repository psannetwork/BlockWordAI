# -*- coding: utf-8 -*-
"""
BlockWordAI - Japanese Toxicity Detection API
Main application file for the toxicity detection service.

This file provides a user-friendly interface to the trained model
and can be used either directly in Python or as a web service.
"""

import os
import sys
import json
from datetime import datetime

from localization import gettext


def load_model(model_path=None):
    """
    Load the trained toxicity detection model
    """
    if model_path is None:
        model_path = os.getenv('BLOCKWORDAI_MODEL_PATH', './models/latest')
    
    print(f"🔄 Loading model from: {model_path}")
    
    # Import required libraries inside the function to avoid import errors if not needed
    try:
        from transformers import AutoTokenizer
        import onnxruntime as ort
        import numpy as np
        
        # Check if ONNX model exists
        model_file = os.path.join(model_path, "toxic-bert-jp.onnx")
        tokenizer_config_file = os.path.join(model_path, "tokenizer_config.json")
        
        if not os.path.exists(model_file):
            raise FileNotFoundError(gettext("❌ ONNX model file not found: {model_file}\nPlease run training first with: python train/train.py", model_file=model_file))
        
        if not os.path.exists(tokenizer_config_file):
            raise FileNotFoundError(gettext("❌ Tokenizer configuration not found: {tokenizer_config_file}\nPlease run training first with: python train/train.py", tokenizer_config_file=tokenizer_config_file))
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        session = ort.InferenceSession(model_file)
        
        print(gettext("✅ Model loaded successfully"))
        return tokenizer, session
    
    except ImportError as e:
        print(gettext("❌ Missing required packages: {e}", e=e))
        print(gettext("💡 Install required packages with: pip install -r requirements.txt"))
        return None, None
    except Exception as e:
        print(gettext("❌ Error loading model: {e}", e=e))
        return None, None


def predict_toxicity(text: str, tokenizer=None, session=None) -> dict:
    """
    Predict toxicity probability for a given text
    
    Args:
        text (str): Input text to analyze
        tokenizer: Loaded tokenizer instance
        session: ONNX session instance
    
    Returns:
        dict: Contains probability and classification result
    """
    if tokenizer is None or session is None:
        tokenizer, session = load_model()
        if tokenizer is None or session is None:
            return {"error": "Failed to load model"}
    
    try:
        # Limit text length to 200 characters and tokenize
        text = text[:200]
        enc = tokenizer(text, truncation=True, padding=True, max_length=128)
        input_ids = [enc["input_ids"]]
        attention_mask = [enc["attention_mask"]]
        
        # Run inference
        outputs = session.run(None, {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        })
        
        logits = outputs[0][0]
        # Apply softmax to get probabilities
        import numpy as np
        exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
        probs = exp_logits / np.sum(exp_logits)
        
        # Assuming class 1 is the toxic class
        toxic_prob = float(probs[1])
        
        # Classification based on threshold
        is_toxic = toxic_prob > 0.5
        
        return {
            "text": text,
            "toxic_probability": round(toxic_prob, 4),
            "is_toxic": is_toxic,
            "classification": "toxic" if is_toxic else "non-toxic"
        }
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}


def batch_predict_toxicity(texts: list, tokenizer=None, session=None) -> list:
    """
    Predict toxicity for a list of texts
    
    Args:
        texts (list): List of texts to analyze
        tokenizer: Loaded tokenizer instance
        session: ONNX session instance
    
    Returns:
        list: List of prediction results
    """
    if tokenizer is None or session is None:
        tokenizer, session = load_model()
        if tokenizer is None or session is None:
            return [{"error": "Failed to load model"} for _ in texts]
    
    results = []
    for text in texts:
        result = predict_toxicity(text, tokenizer, session)
        results.append(result)
    
    return results


def run_demo():
    """
    Run a demonstration of the toxicity detection
    """
    print(gettext("🔍 BlockWordAI - Japanese Toxicity Detection Demo"))
    print("="*50)
    
    # Load model
    tokenizer, session = load_model()
    if tokenizer is None or session is None:
        print(gettext("❌ Cannot run demo without a valid model"))
        return
    
    # Test examples
    test_comments = [
        "えらいね〜、また失敗かよ",
        "素晴らしいですね！",
        "クソゲーすぎるわ",
        "改善点を教えてください",
        "無理にしろよ、どうせできないくせに",
        "ありがとうございます",
        "マジ無理すぎる",
        "がんばってください",
        "死ねばいいのに",
        "ちょっと考えすぎじゃない？",
        "パチンカスが",
        "ひよってるね",
        "ミスった",
        "ざぁこ",
        "よわww",
        "お腹いっぱい",
        "社畜",
        "予想が付きませんね",
        "じじい",
        "ガキ",
    ]
    
    print(gettext("🔍 Japanese toxicity detection demo:"))
    for comment in test_comments:
        result = predict_toxicity(comment, tokenizer, session)
        if "error" not in result:
            print(f"Comment: {comment} → {result['classification']} ({result['toxic_probability']})")
        else:
            print(f"Comment: {comment} → Error: {result['error']}")
    
    print(gettext("\n🎉 Demo completed"))


def main():
    """
    Main function to handle command line arguments
    """
    import argparse
    
    parser = argparse.ArgumentParser(description=gettext('BlockWordAI - Japanese Toxicity Detection'))
    parser.add_argument('--demo', action='store_true', help=gettext('Run a demonstration'))
    parser.add_argument('--predict', type=str, help=gettext('Predict toxicity for a single text'))
    parser.add_argument('--model-path', type=str, help=gettext('Path to the trained model'))
    
    args = parser.parse_args()
    
    if args.demo:
        run_demo()
    elif args.predict:
        result = predict_toxicity(args.predict)
        if "error" not in result:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(f"Error: {result['error']}")
    else:
        print(gettext("BlockWordAI - Japanese Toxicity Detection"))
        print(gettext("Usage:"))
        print(gettext("  python app.py --demo                    # Run demonstration"))
        print(gettext("  python app.py --predict \"TEXT\"         # Predict toxicity for text"))
        print("")
        print(gettext("Environment variables:"))
        print(gettext("  BLOCKWORDAI_MODEL_PATH    Path to trained model (default: ./models/latest)"))
        print("")
        print(gettext("To train a new model: python train/train.py"))
        print(gettext("To test the model: python train/test.py"))


if __name__ == "__main__":
    main()