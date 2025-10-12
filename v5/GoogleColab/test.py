print("📦 [1] パッケージをインストール中...")
!pip install -q transformers datasets sentencepiece scikit-learn torch onnx onnxruntime optimum fugashi[unidic-lite] emoji pandas
!pip install -q --upgrade optimum

# 🔗 Google Drive マウント
print("🔗 Google Drive をマウント中...")
from google.colab import drive
drive.mount('/content/drive')

# テストセットで簡単な評価
from sklearn.metrics import classification_report
import numpy as np
import os
from onnxruntime import InferenceSession
import torch # torchをインポートしてsoftmaxをテンソルで計算できるようにする
from transformers import AutoTokenizer # ここでトークナイザをインポート

# LOCAL_MODEL_DIR と DRIVE_MODEL_DIR が定義されていないエラーを修正
LOCAL_MODEL_DIR = "./psan_comment_ai"
DRIVE_MODEL_DIR = "/content/drive/MyDrive/psan_comment_ai_drive" # こちらも定義が必要

LOCAL_ONNX_MODEL_PATH = os.path.join(LOCAL_MODEL_DIR, "model.onnx")
DRIVE_ONNX_MODEL_PATH = os.path.join(DRIVE_MODEL_DIR, "model.onnx")

# ローカルにONNXモデルがない場合、Google Driveを確認してディレクトリごとコピー
if not os.path.exists(LOCAL_ONNX_MODEL_PATH) and os.path.exists(DRIVE_ONNX_MODEL_PATH):
    print(f"🔍 Google Drive にONNXモデルがあります。関連ファイルを含めローカルにコピーします: {DRIVE_MODEL_DIR}/* -> {LOCAL_MODEL_DIR}/")
    # コピー先のディレクトリが存在することを確認
    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
    # ディレクトリの内容を全てコピー
    !cp -r {DRIVE_MODEL_DIR}/* {LOCAL_MODEL_DIR}/
    print("✅ コピー完了。")


if os.path.exists(LOCAL_ONNX_MODEL_PATH):
    print("🧪 ONNX 推論テストを実行中...")

    # softmax を正しく定義 (numpy ではなく torch を使う)
    def softmax(x, axis=-1):
        # torchテンソルに変換してから計算
        x_tensor = torch.tensor(x)
        # torch.max returns a tuple (values, indices)
        max_values, _ = torch.max(x_tensor, dim=axis, keepdim=True)
        exp_x = torch.exp(x_tensor - max_values)
        return (exp_x / torch.sum(exp_x, dim=axis, keepdims=True)).numpy() # numpyに戻す

    # トークナイザのロード (もし前回の実行で消えていたら)
    try:
        # トークナイザはモデルと同じディレクトリに保存されているはず
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR)
        print(f"✅ トークナイザを {LOCAL_MODEL_DIR} からロードしました。")
    except Exception as e:
        print(f"⚠️ トークナイザのロードに失敗しました: {e}")
        print(f"代わりに元のモデル名からトークナイザを初期化します: {MODEL_NAME}") # MODEL_NAME は前のセルで定義
        try:
            MODEL_NAME = "cl-tohoku/bert-base-japanese" # もし前のセルが実行されてなければここで定義
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            print(f"✅ トークナイザを {MODEL_NAME} から初期化しました。")
        except Exception as e_fallback:
             print(f"❌ トークナイザの初期化に失敗しました: {e_fallback}")
             print("トークナイザが利用できません。推論テストをスキップします。")
             tokenizer = None # トークナイザがNoneの場合は推論をスキップ

    # トークナイザが正常にロードまたは初期化された場合のみ推論を実行
    if tokenizer is not None:
        # 複数のテキストをリストとして定義
        test_texts = [
            "マジで存在価値ないよねw",
            "死ねばいいのに",
            "お前ほんと使えないよな（笑）",
            "こんにちは！",
            "Have a nice day!",
            "君の存在、ない方がスッキリするかも",
            "お前は本当に無能だな",
            "死ねや",
            "笑顔が邪悪だ",
            "あなたはただのクズだ"
        ]

        # 複数のテキストをバッチでトークナイズ
        inputs = tokenizer(
            test_texts,
            return_tensors="np",
            padding="max_length",
            truncation=True, # truncationを追加
            max_length=128,
        )

        ort_session = InferenceSession(LOCAL_ONNX_MODEL_PATH)
        ort_out = ort_session.run(
            None,
            {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
            }
        )
        logits = ort_out[0]  # shape: (batch_size, 2)
        probs = softmax(logits, axis=-1) # shape: (batch_size, 2)

        # 各テキストの結果を表示
        for i, text in enumerate(test_texts):
            harmful_prob = probs[i][1]
            # 有害確率の閾値を0.9から0.95に変更
            sentiment = "有害" if harmful_prob > 0.5 else "非有害"

            print(f"テキスト: 「{text}」")
            print(f"予測: {sentiment} (有害確率: {harmful_prob:.3f})")
        print("-" * 20)


    else:
        print("⚠️ トークナイザが利用できないため、推論テストをスキップしました。")

else:
    print("⚠️ ローカルにもGoogle DriveにもONNXモデルが存在しないため、推論テストをスキップします。")

print("\n✨ 全工程完了！")
print(f"モデル保存先（ローカル）: {LOCAL_MODEL_DIR}")
print(f"ONNXファイル: {LOCAL_ONNX_MODEL_PATH if os.path.exists(LOCAL_ONNX_MODEL_PATH) else 'なし'}")
