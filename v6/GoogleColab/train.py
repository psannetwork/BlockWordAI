# -*- coding: utf-8 -*-
# Google Colab で実行

# 0. fugashi と辞書のインストール
!pip install fugashi[unidic-lite]

# 1. Google Drive 接続
from google.colab import drive
drive.mount('/content/drive')

# 2. 必要ライブラリのインストール
!pip install transformers datasets tokenizers torch onnx

# 3. データ読み込み
import pandas as pd

df = pd.read_json('/content/drive/MyDrive/Dataset/combined_toxicity_dataset.json')
print("✅ データ読み込み完了")
print("データ数:", len(df))
print("ラベル分布:")
print(df['labels'].value_counts())

# 4. データ前処理（必要に応じて）
df = df.dropna(subset=['text'])
df = df[df['text'].str.len() > 0]

# 5. Dataset に変換
from datasets import Dataset
dataset = Dataset.from_pandas(df)

# 6. トークナイザーとモデルの準備
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

model_name = "cl-tohoku/bert-base-japanese-whole-word-masking"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True, max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 7. 日付付きディレクトリ名
from datetime import datetime
import os

date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"/content/drive/MyDrive/toxicity_model_{date_str}"
os.makedirs(output_dir, exist_ok=True)

# 学習済みモデルが既に存在するか確認
model_save_path = f"{output_dir}/pytorch_model.bin"

if os.path.exists(model_save_path):
    print("✅ 学習済みモデルが見つかりました。学習をスキップします。")
    model = AutoModelForSequenceClassification.from_pretrained(output_dir, num_labels=2)
else:
    print("🔄 学習を開始します...")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,
        eval_strategy="no",
        save_strategy="no",
        load_best_model_at_end=False,
        save_total_limit=1,
        gradient_checkpointing=False,
        report_to="none",  # wandbを無効化
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=None,
        tokenizer=tokenizer,
    )

    trainer.train()

    # 学習済みモデルを保存
    model.save_pretrained(output_dir)
    print(f"✅ 学習完了 & モデルを {output_dir} に保存しました")

# 8. ONNX変換
print("🔄 ONNX変換を開始します...")

# GPUが使えるか確認
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# 入力テンソルもGPUに送る
dummy_input_ids = torch.randint(1, 1000, (1, 128)).to(device)
dummy_attention_mask = torch.ones((1, 128), dtype=torch.long).to(device)

torch.onnx.export(
    model,
    (dummy_input_ids, dummy_attention_mask),
    f"{output_dir}/toxic-bert-jp.onnx",
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

# トークナイザー保存
tokenizer.save_pretrained(output_dir)

print(f"✅ ONNX変換完了")
print(f"✅ モデルは以下に保存されました: {output_dir}")
print(f"✅ ファイル: {output_dir}/toxic-bert-jp.onnx, {output_dir}/tokenizer.json")
