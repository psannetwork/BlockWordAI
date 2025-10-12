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
!pip install -q transformers datasets sentencepiece scikit-learn torch onnx onnxruntime optimum fugashi[unidic-lite] emoji
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
# 📊 2-3. データセット構築（多言語強化 + 日本語特化）
# ==========================================================
print("📊 [2-3] データセットを構築中...")

from datasets import load_dataset, concatenate_datasets
import re
import emoji
import unicodedata

# 日本語前処理（日本語テキストにのみ適用）
def preprocess_ja(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r'[wｗ]{3,}', 'ww', text)
    text = re.sub(r'[草]{3,}', '草草', text)
    text = re.sub(r'[！!]{2,}', '！', text)
    text = re.sub(r'[？?]{2,}', '？', text)
    text = unicodedata.normalize('NFKC', text)
    return text.strip()

# === 英語データ（Jigsaw）===
print("   - 英語データ「Jigsaw Toxic Comments（Kaggle版）」を追加中...")
try:
    jigsaw = load_dataset("jigsaw-toxic-comment-classification-challenge", split="train")
    jigsaw = jigsaw.map(lambda x: {
        "text": x["comment_text"],
        "toxic": int(any([
            x["toxic"],
            x["severe_toxic"],
            x["obscene"],
            x["threat"],
            x["insult"],
            x["identity_hate"]
        ]))
    })
    jigsaw = jigsaw.remove_columns([col for col in jigsaw.column_names if col not in ["text", "toxic"]])
    print("✅ 英語データ読み込み成功")
except Exception as e:
    print(f"❌ 英語データ読み込み失敗: {e}")
    jigsaw = None

# === 多言語データ（Jigsaw Multilingual）===
print("   - 多言語データ「Jigsaw Multilingual」を追加中...")
try:
    jigsaw_multi = load_dataset("jigsaw_multilingual_toxic_comment_classification", "all_languages", split="train")
    jigsaw_multi = jigsaw_multi.map(lambda x: {"text": x["comment_text"], "toxic": int(x["toxic"])})
    jigsaw_multi = jigsaw_multi.remove_columns([col for col in jigsaw_multi.column_names if col not in ["text", "toxic"]])
    print("✅ 多言語データ読み込み成功")
except Exception as e:
    print(f"❌ 多言語データ読み込み失敗: {e}")
    jigsaw_multi = None

# === 日本語データ ===
print("   - 基本日本語データを追加中...")
try:
    dataset = load_dataset("textdetox/multilingual_toxicity_dataset")
    ja_data = dataset['ja']
    print("✅ 日本語データ読み込み成功")
except Exception as e:
    print(f"❌ 日本語データ読み込み失敗: {e}")
    ja_data = None

print("   - Attqa-Q JA を追加中...")
try:
    attaq = load_dataset("ibm-research/AttaQ-JA", split="test")
    attaq_ja = attaq.map(lambda x: {"text": x["input"], "toxic": 1}, remove_columns=['uid', 'label', 'input'])
    print("✅ Attqa-Q JA 読み込み成功")
except Exception as e:
    print(f"❌ Attqa-Q JA 読み込み失敗: {e}")
    attaq_ja = None

print("   - LLM-jp Toxicity を追加中...")
try:
    toxic_llmjp = load_dataset("p1atdev/LLM-jp-Toxicity-Dataset", split="train")
    toxic_llmjp = toxic_llmjp.map(
        lambda x: {"text": x["text"], "toxic": 1 if x["label"] == "toxic" else 0},
        remove_columns=["label"]
    )
    print("✅ LLM-jp Toxicity 読み込み成功")
except Exception as e:
    print(f"❌ LLM-jp Toxicity 読み込み失敗: {e}")
    toxic_llmjp = None

print("   - Japanese-Hate-Speech を追加中...")
try:
    ja_hate = load_dataset("shunk031/Japanese-Hate-Speech", split="train")
    ja_hate = ja_hate.map(lambda x: {"text": x["text"], "toxic": 1})
    print("✅ Japanese-Hate-Speech 読み込み成功")
except Exception as e:
    print(f"❌ Japanese-Hate-Speech 読み込み失敗: {e}")
    ja_hate = None

print("   - Japanese-Toxic-Comments を追加中...")
try:
    ja_toxic = load_dataset("yamaguchi1214/Japanese-Toxic-Comments", split="train")
    ja_toxic = ja_toxic.rename_column("comment", "text")
    ja_toxic = ja_toxic.map(lambda x: {"toxic": 1})
    print("✅ Japanese-Toxic-Comments 読み込み成功")
except Exception as e:
    print(f"❌ Japanese-Toxic-Comments 読み込み失敗: {e}")
    ja_toxic = None

# データセットの組み立て（存在するものだけ）
datasets_to_use = []
if jigsaw is not None:
    datasets_to_use.append(jigsaw)
if jigsaw_multi is not None:
    datasets_to_use.append(jigsaw_multi)
if ja_data is not None:
    datasets_to_use.append(ja_data)
if attaq_ja is not None:
    datasets_to_use.append(attaq_ja)
if toxic_llmjp is not None:
    datasets_to_use.append(toxic_llmjp)
if ja_hate is not None:
    datasets_to_use.append(ja_hate)
if ja_toxic is not None:
    datasets_to_use.append(ja_toxic)

if not datasets_to_use:
    raise ValueError("すべてのデータセットの読み込みに失敗しました。")

# 日本語データに前処理を適用
ja_datasets = [ja_data, attaq_ja, toxic_llmjp, ja_hate, ja_toxic]
ja_datasets_processed = []
for ds in ja_datasets:
    if ds is not None:
        ds = ds.map(lambda x: {"text": preprocess_ja(x["text"]), "toxic": x["toxic"]})
        ja_datasets_processed.append(ds)
    else:
        ja_datasets_processed.append(None)

# サブラベル付きサンプルをオーバーサンプリング（LLM-jp のみ）
def has_sublabel(example):
    if example['toxic'] != 1:
        return False
    return any(example.get(k, 0) == 1 for k in ['obscene', 'discriminatory', 'violent', 'illegal', 'sexual'])

if toxic_llmjp is not None:
    toxic_llmjp_proc = toxic_llmjp
    toxic_with_sub = toxic_llmjp_proc.filter(has_sublabel)
    toxic_llmjp_balanced = concatenate_datasets([toxic_llmjp_proc, toxic_with_sub])
else:
    toxic_llmjp_balanced = None

# 日本語最終データ
ja_final_list = []
for ds in ja_datasets_processed:
    if ds is not None:
        ja_final_list.append(ds)

ja_final = None
if ja_final_list:
    ja_final = concatenate_datasets(ja_final_list)

# 全データ統合（存在するものだけ）
combined = concatenate_datasets(datasets_to_use)
if ja_final is not None:
    combined = concatenate_datasets([combined, ja_final])

combined = combined.filter(lambda x: len(x["text"].strip()) > 0)
combined = combined.train_test_split(test_size=0.1, seed=42)

train_dataset = combined['train'].shuffle(seed=42)
test_dataset = combined['test'].shuffle(seed=42)

print(f"   - 訓練: {len(train_dataset)} 件")
print(f"   - テスト: {len(test_dataset)} 件")
print("✅ [2-3] データ準備完了。")
print("==========================================================")


# ==========================================================
# 🏷️ 4. トークナイザ（xlm-roberta-base：100言語対応）
# ==========================================================
print("🏷️ [4] トークナイザを準備中...")

MODEL_NAME = "xlm-roberta-base"  # ← 多言語対応 + 日本語もOK
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True)

def tokenize(batch):
    return tokenizer(
        batch['text'],
        padding='max_length',
        truncation=True,
        max_length=128,
    )

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

# ラベル統一
def rename_label(example):
    example['label'] = int(example['toxic'])
    return example

train_dataset = train_dataset.map(rename_label)
test_dataset = test_dataset.map(rename_label)

# 不要カラム削除
keep_cols = ['input_ids', 'attention_mask', 'label']
train_dataset = train_dataset.remove_columns(
    [col for col in train_dataset.column_names if col not in keep_cols]
)
test_dataset = test_dataset.remove_columns(
    [col for col in test_dataset.column_names if col not in keep_cols]
)

train_dataset.set_format(type='torch', columns=keep_cols)
test_dataset.set_format(type='torch', columns=keep_cols)

print("✅ [4] トークナイザ処理完了。")
print("==========================================================")


# ==========================================================
# 🧠 5-7. モデル定義 & 学習設定（Recall重視）
# ==========================================================
print("🧠 [5-7] モデルとTrainerを準備中...")

from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
    }

training_args = TrainingArguments(
    output_dir="./comment_model",
    num_train_epochs=3,
    per_device_train_batch_size=16,  # XLM-RoBERTa は重め
    per_device_eval_batch_size=32,
    save_total_limit=1,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="recall",
    greater_is_better=True,
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
    !cp -r {os.path.dirname(DRIVE_CONFIG_PATH)}/* {LOCAL_MODEL_DIR}/
    model = AutoModelForSequenceClassification.from_pretrained(LOCAL_MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR)
elif os.path.exists(CONFIG_PATH):
    print("🔍 ローカルにモデルがあります。ロードします。")
    model = AutoModelForSequenceClassification.from_pretrained(LOCAL_MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR)
else:
    print("⏳ 学習を開始します（多言語 + 日本語強化）...")
    trainer.train()
    print("✅ 学習完了！")
    model.save_pretrained(LOCAL_MODEL_DIR)
    tokenizer.save_pretrained(LOCAL_MODEL_DIR)
    print(f"✅ ローカルに保存: {LOCAL_MODEL_DIR}")
    !cp -r {LOCAL_MODEL_DIR}/* {DRIVE_MODEL_DIR}/
    print(f"✅ Google Drive に保存: {DRIVE_MODEL_DIR}")

print("==========================================================")


# ==========================================================
# ⚡ 9. ONNX 変換
# ==========================================================
LOCAL_ONNX_MODEL_PATH = os.path.join(LOCAL_MODEL_DIR, "model.onnx")
DRIVE_ONNX_MODEL_PATH = os.path.join(DRIVE_MODEL_DIR, "model.onnx")

if not os.path.exists(LOCAL_ONNX_MODEL_PATH):
    print("⚡ ONNX変換中（opset 14）...")
    import torch
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dummy_input_ids = torch.randint(0, tokenizer.vocab_size, (1, 128), dtype=torch.long).to(device)
    dummy_attention_mask = torch.ones((1, 128), dtype=torch.long).to(device)

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
else:
    print("🔍 ONNXモデルが存在します。スキップ。")

print("==========================================================")


# ==========================================================
# ⚡ 9.5. INT8量子化（CPU高速化）
# ==========================================================
QUANTIZED_DIR = os.path.join(LOCAL_MODEL_DIR, "quantized")
QUANTIZED_ONNX = os.path.join(QUANTIZED_DIR, "model_quantized.onnx")

if not os.path.exists(QUANTIZED_ONNX):
    print("🔍 INT8量子化中（CPU高速化）...")
    from optimum.onnxruntime import ORTQuantizer
    quantizer = ORTQuantizer.from_pretrained(LOCAL_MODEL_DIR, file_name="model.onnx")
    quantizer.quantize(save_dir=QUANTIZED_DIR)
    print(f"✅ 量子化完了 → {QUANTIZED_DIR}")
else:
    print("✅ 量子化済みモデルが存在します。")

print("==========================================================")


# ==========================================================
# 🧪 10. 多言語推論テスト
# ==========================================================
if os.path.exists(QUANTIZED_ONNX):
    print("🧪 量子化ONNXで多言語推論テスト中...")
    from onnxruntime import InferenceSession
    import numpy as np

    def softmax(x, axis=-1):
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    test_texts = [
        # 日本語（陰湿）
        "マジで存在価値ないよねw",
        "死ねばいいのに",
        # 英語
        "You are a worthless piece of shit.",
        "Go kill yourself!",
        # 無害
        "こんにちは！",
        "Have a nice day!",
    ]

    ort_session = InferenceSession(QUANTIZED_ONNX)
    for text in test_texts:
        inputs = tokenizer(
            text,
            return_tensors="np",
            padding="max_length",
            max_length=128,
        )
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
        lang = "🇯🇵" if any(c in text for c in "あいうえおかきくけこ") else "🌐"
        print(f"{lang} テキスト: {text}")
        print(f"  → 予測: {'有害' if pred == 1 else '非有害'} (有害確率: {prob[1]:.3f})")
else:
    print("⚠️ 量子化ONNXモデルが存在しません。")

print("\n✨ 全工程完了！")
print(f"モデル保存先（ローカル）: {LOCAL_MODEL_DIR}")
print(f"量子化ONNX: {QUANTIZED_ONNX if os.path.exists(QUANTIZED_ONNX) else 'なし'}")
