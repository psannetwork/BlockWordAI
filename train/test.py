# -*- coding: utf-8 -*-
# Google Colab で実行

# 1. Google Drive 接続
from google.colab import drive
drive.mount('/content/drive')

# 2. 必要ライブラリのインストール
# fugashi と、fugashiが動作するための辞書 unidic-lite を追加
#!pip install onnxruntime transformers fugashi unidic-lite torch onnx

# 3. 最新の学習済みモデルディレクトリを自動検出
import os
import glob

# 最も新しい日付付きディレクトリを取得
model_dirs = glob.glob('/content/drive/MyDrive/toxicity_model_*')
if not model_dirs:
    raise Exception("❌ 学習済みモデルディレクトリが見つかりません")

latest_dir = sorted(model_dirs, key=lambda x: os.path.getmtime(x))[-1]

print(f"✅ 最新モデルディレクトリ: {latest_dir}")

# 4. ファイルが存在するか確認
model_file = f"{latest_dir}/toxic-bert-jp.onnx"
tokenizer_config_file = f"{latest_dir}/tokenizer_config.json"

# ONNXファイルが存在しない場合は変換を実行
if not os.path.exists(model_file):
    print("🔄 ONNXファイルが見つかりませんでした。変換を実行します...")
    import torch
    from transformers import AutoModelForSequenceClassification

    # モデルの読み込み
    model = AutoModelForSequenceClassification.from_pretrained(latest_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    # ダミー入力の準備
    dummy_input_ids = torch.randint(1, 1000, (1, 128)).to(device)
    dummy_attention_mask = torch.ones((1, 128), dtype=torch.long).to(device)

    # ONNX変換
    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask),
        model_file,
        export_params=True,
        opset_version=14,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch"},
        }
    )
    print(f"✅ ONNX変換完了: {model_file}")

if not os.path.exists(tokenizer_config_file):
    raise FileNotFoundError(f"❌ トークナイザー設定がありません: {tokenizer_config_file}")

# 5. トークナイザーの読み込み（transformers版）
from transformers import AutoTokenizer
import onnxruntime as ort
import numpy as np

# AutoTokenizer が適切なトークナイザークラス（この場合は MecabTokenizer）を自動で読み込みます
# 辞書 (unidic-lite) がインストールされている必要があります
tokenizer = AutoTokenizer.from_pretrained(latest_dir)

session = ort.InferenceSession(model_file)

def predict_toxicity(text: str) -> float:
    # トークナイズ処理が fugashi/unidic を使用して行われます
    enc = tokenizer(text[:200], truncation=True, padding=True, max_length=128)
    input_ids = np.array([enc["input_ids"]], dtype=np.int64)
    attention_mask = np.array([enc["attention_mask"]], dtype=np.int64)
    logits = session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})[0][0]
    probs = np.exp(logits) / np.sum(np.exp(logits))
    # クラス 1 が有害である確率と仮定
    return float(probs[1])  # 有害確率（0.0～1.0）

# 6. 動作テスト
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

print("🔍 風紀乱れ検出テスト:")
for comment in test_comments:
    score = predict_toxicity(comment)
    print(f"コメント: {comment} → 確率: {score:.3f}")
