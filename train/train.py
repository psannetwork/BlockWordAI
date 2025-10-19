# -*- coding: utf-8 -*-
# Google Colab で実行

!pip install fugashi[unidic-lite] transformers datasets tokenizers torch scikit-learn

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd, random, numpy as np, torch, os, json
from datetime import datetime
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score
import re, unicodedata

# ----------------------------
# 乱数固定（再現性向上）
# ----------------------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# ----------------------------
# 1. データ読み込み
# ----------------------------
file_path = '/content/drive/MyDrive/Dataset/combined_toxicity_dataset.json'

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    print("✅ データ読み込み完了")
    print("データ数:", len(df))
    print("ラベル分布:\n", df['labels'].value_counts())
except Exception as e:
    print(f"❌ データ読み込みエラー: {e}")
    exit()

# ----------------------------
# 2. データ前処理
# ----------------------------
df = df.dropna(subset=['text'])
df = df[df['text'].str.len() > 0].reset_index(drop=True)

def is_toxic_label(v):
    try:
        return int(v) == 1
    except:
        if isinstance(v, str) and v.lower() in {'toxic','true','1'}:
            return True
    return False

df['is_toxic'] = df['labels'].apply(is_toxic_label)

# ----------------------------
# 3. 文字バリエーション拡張
# ----------------------------
toxic_df = df[df['is_toxic']].copy()
non_toxic_df = df[~df['is_toxic']].copy()
print(f"抽出: 有害={len(toxic_df)} / 非有害={len(non_toxic_df)}")

def normalize_text(text):
    text = unicodedata.normalize("NFKC", str(text))
    text = re.sub(r'[○●〇◯*✱★☆⚪️・\u25CB\u25CF\u25A1\u25A0]', '', text)
    return text

# 文字崩しバリエーションを強化
char_variants = {
    'あ': ['ｱ','ぁ','@','a'],
    'い': ['ｲ','1','l','I','!','i'],
    'う': ['ｳ','u'],
    'え': ['ｴ','e'],
    'お': ['0','〇','o','O'],
    'か': ['ｶ','k'], 'き': ['ｷ','k','ki'], 'く': ['ｸ','ku'], 'け': ['ｹ','ke'], 'こ': ['ｺ','0','ko'],
    'さ': ['ｻ','s'], 'し': ['ｼ','5','shi'], 'す': ['ｽ','su'], 'せ': ['ｾ','se'], 'そ': ['ｿ','so','5'],
    'た': ['ﾀ','t'], 'ち': ['ﾁ','chi','c'], 'つ': ['ﾂ','っ','tsu'], 'て': ['ﾃ','te'], 'と': ['ﾄ','to','10','t'],
    'な': ['ﾅ','n'], 'に': ['ﾆ','ni'], 'ぬ': ['ﾇ','nu'], 'ね': ['ﾈ','ne'], 'の': ['ﾉ','no'],
    'は': ['ﾊ','h'], 'ひ': ['ﾋ','hi'], 'ふ': ['ﾌ','fu'], 'へ': ['ﾍ','he'], 'ほ': ['ﾎ','ho'],
    'ま': ['ﾏ','m'], 'み': ['ﾐ','mi'], 'む': ['ﾑ','mu'], 'め': ['ﾒ','me'], 'も': ['ﾓ','mo'],
    'や': ['ﾔ','ゃ','ya'], 'ゆ': ['ﾕ','ゅ','yu'], 'よ': ['ﾖ','ょ','yo'],
    'ら': ['ﾗ','ra','l','r'], 'り': ['ﾘ','l','1','r'], 'る': ['ﾙ','ru'], 'れ': ['ﾚ','re'], 'ろ': ['ﾛ','ro'],
    'わ': ['ﾜ','w','wa'], 'を': ['ｦ','wo'], 'ん': ['ﾝ','n','nn'],
    ' ': ['　','_','-'],  # 空白も置換
    'ー': ['ｰ','―','-'], # 長音も置換
}

# 顔文字・シンプル絵文字の追加
face_variants = ['☆','★','♪','💢','⚡','🔥','💀','🤬']

def augment_text(text, prob=0.35, face_prob=0.05):
    text = normalize_text(text)
    out = []
    for ch in text:
        if random.random() < prob:
            variants = char_variants.get(ch, [])
            if variants:
                out.append(random.choice(variants))
            else:
                out.append(ch)
        else:
            out.append(ch)
    # ランダムで顔文字を挿入
    if random.random() < face_prob:
        pos = random.randint(0, len(out))
        out.insert(pos, random.choice(face_variants))
    return ''.join(out)

def generate_augmented_toxics(toxic_df, multiplier=3, prob=0.35):
    aug_texts = []
    for _, row in toxic_df.iterrows():
        orig = row['text']
        for _ in range(multiplier):
            aug_texts.append({'text': augment_text(orig, prob), 'labels': row['labels']})
    return pd.DataFrame(aug_texts)

aug_df = generate_augmented_toxics(toxic_df, multiplier=3, prob=0.35)
print(f"生成: 拡張有害データ = {len(aug_df)} 件")

final_df = pd.concat([df[['text','labels']], aug_df], ignore_index=True)
final_df = final_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
print(f"最終データ数: {len(final_df)}")

# ----------------------------
# 4. Dataset変換＆分割
# ----------------------------
dataset = Dataset.from_pandas(final_df).train_test_split(test_size=0.1, seed=seed)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# ----------------------------
# 5. トークナイザー＆モデル準備
# ----------------------------
model_name = "cl-tohoku/bert-base-japanese-whole-word-masking"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True, max_length=128)

train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.map(tokenize_function, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"accuracy": accuracy_score(labels, preds), "f1": f1_score(labels, preds)}

# ----------------------------
# 6. 学習設定
# ----------------------------
date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"/content/drive/MyDrive/toxicity_model_{date_str}"
os.makedirs(output_dir, exist_ok=True)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    save_total_limit=2,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# ----------------------------
# 7. 学習実行
# ----------------------------
print("🔄 学習を開始します...")
trainer.train()
print("✅ 学習完了")

# ----------------------------
# 8. モデル保存
# ----------------------------
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"✅ モデル保存完了: {output_dir}")

# ----------------------------
# 9. ONNX変換
# ----------------------------
print("🔄 ONNX変換を開始します...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()
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
print(f"✅ ONNX変換完了: {output_dir}/toxic-bert-jp.onnx")
