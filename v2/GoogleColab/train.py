# ==========================================================
# 🛡️ 0. セッション維持（Colab切断防止）
# ==========================================================
import IPython
import time
import os

display(IPython.display.Javascript('''
  function ConnectButton(){
    console.log("Auto-clicking connect button...");
    document.querySelector("#connect").click();
  }
  setInterval(ConnectButton, 60000);
'''))

print("✅ [0] セッション維持スクリプトを開始しました。")
time.sleep(5)
print("==========================================================")


# ==========================================================
# 📦 1. 必要パッケージのインストール
# ==========================================================
print("📦 [1] パッケージをインストール中...")
!pip install -q transformers datasets sentencepiece scikit-learn torch onnx onnxruntime optimum fugashi[unidic-lite]
!pip install -q --upgrade optimum
print("✅ [1] インストール完了。")
print("==========================================================")


# ==========================================================
# 🔗 1.5. Google Drive マウント
# ==========================================================
print("🔗 [1.5] Google Drive をマウント中...")
from google.colab import drive
drive.mount('/content/drive')
DRIVE_MODEL_DIR = "/content/drive/MyDrive/psan_comment_ai_drive"
os.makedirs(DRIVE_MODEL_DIR, exist_ok=True)
print(f"✅ 保存先: {DRIVE_MODEL_DIR}")
print("==========================================================")


# ==========================================================
# 📊 2-3. データセット構築（多言語 + 日本語高品質有害データ）
# ==========================================================
print("📊 [2-3] データセットを構築中...")

from datasets import load_dataset, concatenate_datasets

# 基本データ（英語・日本語）
dataset = load_dataset("textdetox/multilingual_toxicity_dataset")
ja_data = dataset['ja']
en_data = dataset['en']

# 高品質日本語有害データ（IBM Research 提供）
print("   - 日本語有害データ「ibm-research/AttaQ-JA」を追加中...")
attaq = load_dataset("ibm-research/AttaQ-JA", split="test")
def to_toxic(example):
    example['text'] = example['input']
    example['toxic'] = 1  # 全て有害
    return example
attaq_ja = attaq.map(to_toxic, remove_columns=['uid', 'label', 'input'])

# 統合
combined = concatenate_datasets([ja_data, en_data, attaq_ja])
combined = combined.train_test_split(test_size=0.1, seed=42)

train_dataset = combined['train'].shuffle(seed=42)
test_dataset = combined['test'].shuffle(seed=42)

print(f"   - 訓練: {len(train_dataset)} 件")
print(f"   - テスト: {len(test_dataset)} 件")
print("✅ [2-3] データ準備完了。")
print("==========================================================")


# ==========================================================
# 🏷️ 4. トークナイザ（修正版：nlp-waseda/roberta-base-japanese）
# ==========================================================
print("🏷️ [4] トークナイザを準備中...")

MODEL_NAME = "nlp-waseda/roberta-base-japanese"  # ← 修正：実在モデルに変更
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(
        batch['text'],
        padding='max_length',
        truncation=True,
        max_length=128,
        # RoBERTa は token_type_ids 不要
    )

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

def rename_label(example):
    example['label'] = int(example['toxic'])
    return example

train_dataset = train_dataset.map(rename_label)
test_dataset = test_dataset.map(rename_label)

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

print("✅ [4] トークナイザ処理完了。")
print("==========================================================")


# ==========================================================
# 🧠 5-7. モデル定義 & 学習設定 & Trainer
# ==========================================================
print("🧠 [5-7] モデルとTrainerを準備中...")

from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
import numpy as np

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean().item()
    return {"accuracy": accuracy}

training_args = TrainingArguments(
    output_dir="./comment_model",
    num_train_epochs=5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    save_total_limit=1,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

print("✅ [5-7] 準備完了。")
print("==========================================================")


# ==========================================================
# ⏳ 8. 学習 or ロード
# ==========================================================
LOCAL_MODEL_DIR = "./psan_comment_ai"
CONFIG_PATH = os.path.join(LOCAL_MODEL_DIR, "config.json")
DRIVE_CONFIG_PATH = os.path.join(DRIVE_MODEL_DIR, "config.json")

if os.path.exists(DRIVE_CONFIG_PATH):
    print("🔍 Google Drive に学習済みモデルがあります。ロードします。")
    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
    !cp -r {DRIVE_MODEL_DIR}/* {LOCAL_MODEL_DIR}/
    model = AutoModelForSequenceClassification.from_pretrained(LOCAL_MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR)
elif os.path.exists(CONFIG_PATH):
    print("🔍 ローカルにモデルがあります。ロードします。")
    model = AutoModelForSequenceClassification.from_pretrained(LOCAL_MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR)
else:
    print("⏳ 学習を開始します...")
    trainer.train()
    print("✅ 学習完了！")
    model.save_pretrained(LOCAL_MODEL_DIR)
    tokenizer.save_pretrained(LOCAL_MODEL_DIR)
    print(f"✅ ローカルに保存: {LOCAL_MODEL_DIR}")
    !cp -r {LOCAL_MODEL_DIR}/* {DRIVE_MODEL_DIR}/
    print(f"✅ Google Drive に保存: {DRIVE_MODEL_DIR}")

print("==========================================================")


# ==========================================================
# ⚡ 9. ONNX 変換（修正版：torch 再インポート）
# ==========================================================
LOCAL_ONNX_MODEL_PATH = os.path.join(LOCAL_MODEL_DIR, "model.onnx")
DRIVE_ONNX_MODEL_PATH = os.path.join(DRIVE_MODEL_DIR, "model.onnx")

if os.path.exists(LOCAL_ONNX_MODEL_PATH):
    print("🔍 ONNXモデルが存在します。スキップ。")
else:
    print("⚡ ONNX変換中（opset 14）...")
    import torch  # ← 修正：torch を再インポート
    model.eval()
    dummy_input_ids = torch.randint(0, tokenizer.vocab_size, (1, 128), dtype=torch.long)
    dummy_attention_mask = torch.ones((1, 128), dtype=torch.long)

    try:
        torch.onnx.export(
            model,
            (dummy_input_ids, dummy_attention_mask),
            LOCAL_ONNX_MODEL_PATH,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "logits": {0: "batch"},
            },
            opset_version=14,
            do_constant_folding=True,
            export_params=True,
        )
        print(f"✅ ONNX変換成功 → {LOCAL_ONNX_MODEL_PATH}")
        !cp {LOCAL_ONNX_MODEL_PATH} {DRIVE_ONNX_MODEL_PATH}
        print(f"✅ Google Drive に保存: {DRIVE_MODEL_DIR}")
    except Exception as e:
        print(f"❌ 変換失敗: {e}")

print("==========================================================")


# ==========================================================
# 🧪 10. ONNX 推論テスト
# ==========================================================
if os.path.exists(LOCAL_ONNX_MODEL_PATH):
    print("🧪 ONNX 推論テスト中...")
    from onnxruntime import InferenceSession
    import numpy as np

    def softmax(x, axis=-1):
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    test_text = "あなたは本当に最低な人間ですね。"
    inputs = tokenizer(
        test_text,
        return_tensors="np",
        padding="max_length",
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
    logits = ort_out[0]
    pred = np.argmax(logits, axis=-1)[0]
    prob = softmax(logits, axis=-1)[0]

    print(f"テキスト: {test_text}")
    print(f"予測: {'有害' if pred == 1 else '非有害'} (有害確率: {prob[1]:.3f})")
else:
    print("⚠️ ONNXモデルが存在しません。")

print("\n✨ 全工程完了！")
print(f"モデル保存先（ローカル）: {LOCAL_MODEL_DIR}")
print(f"ONNXファイル: {LOCAL_ONNX_MODEL_PATH if os.path.exists(LOCAL_ONNX_MODEL_PATH) else 'なし'}")
