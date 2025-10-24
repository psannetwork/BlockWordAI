# -*- coding: utf-8 -*-
# Google Colab 実行用
# 日本語有害コメント分類モデル（拡張強化・過学習抑制版）

!pip install -q fugashi[unidic-lite] transformers datasets tokenizers torch scikit-learn nlpaug onnx

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd, random, numpy as np, torch, os, json, re, unicodedata
from datetime import datetime
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from sklearn.metrics import accuracy_score, f1_score
import nlpaug.augmenter.word as naw
from tqdm import tqdm

# =========================================================
# 乱数固定
# =========================================================
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# =========================================================
# 1. データ読み込み
# =========================================================
file_path = '/content/drive/MyDrive/Dataset/combined_toxicity_dataset.json'
with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
df = pd.DataFrame(data)
print("✅ データ読み込み完了", len(df))

# =========================================================
# 2. データ前処理
# =========================================================
df = df.dropna(subset=['text'])
df = df[df['text'].str.len() > 0].reset_index(drop=True)

def is_toxic_label(v):
    try: return int(v)==1
    except:
        if isinstance(v, str) and v.lower() in {'toxic','true','1'}: return True
    return False

df['is_toxic'] = df['labels'].apply(is_toxic_label)
toxic_df = df[df['is_toxic']].copy()
non_toxic_df = df[~df['is_toxic']].copy()
print(f"抽出: 有害={len(toxic_df)} / 非有害={len(non_toxic_df)}")

# =========================================================
# 3. ノイズ・記号拡張
# =========================================================
def normalize_text(text):
    text = unicodedata.normalize("NFKC", str(text))
    text = re.sub(r'[○●〇◯*✱★☆⚪️・\u25CB\u25CF\u25A1\u25A0]', '', text)
    return text

char_variants = {
    'あ': ['ｱ','ぁ','@','a'], 'い': ['ｲ','1','l','I','!','i'], 'う': ['ｳ','u'],
    'え': ['ｴ','e'], 'お': ['0','〇','o','O'],
    'か': ['ｶ','k'], 'き': ['ｷ','ki'], 'く': ['ｸ','ku'], 'け': ['ｹ','ke'], 'こ': ['ｺ','ko'],
    'さ': ['ｻ','s'], 'し': ['ｼ','shi'], 'す': ['ｽ','su'], 'せ': ['ｾ','se'], 'そ': ['ｿ','so'],
    'た': ['ﾀ','t'], 'ち': ['ﾁ','chi'], 'つ': ['ﾂ','っ','tsu'], 'て': ['ﾃ','te'], 'と': ['ﾄ','to'],
    'な': ['ﾅ','n'], 'に': ['ﾆ','ni'], 'ぬ': ['ﾇ','nu'], 'ね': ['ﾈ','ne'], 'の': ['ﾉ','no'],
    'は': ['ﾊ','h'], 'ひ': ['ﾋ','hi'], 'ふ': ['ﾌ','fu'], 'へ': ['ﾍ','he'], 'ほ': ['ﾎ','ho'],
    'ま': ['ﾏ','m'], 'み': ['ﾐ','mi'], 'む': ['ﾑ','mu'], 'め': ['ﾒ','me'], 'も': ['ﾓ','mo'],
    'や': ['ﾔ','ya'], 'ゆ': ['ﾕ','yu'], 'よ': ['ﾖ','yo'],
    'ら': ['ﾗ','ra'], 'り': ['ﾘ','ri'], 'る': ['ﾙ','ru'], 'れ': ['ﾚ','re'], 'ろ': ['ﾛ','ro'],
    'わ': ['ﾜ','wa'], 'を': ['ｦ','wo'], 'ん': ['ﾝ','n','nn'],
    ' ': ['　','_','-'], 'ー': ['ｰ','―','-'],
}

face_variants = ['☆','★','♪','💢','⚡','🔥','💀','🤬']

def augment_text(text, char_prob=0.35, face_prob=0.05):
    text = normalize_text(text)
    out = []
    for ch in text:
        if random.random() < char_prob:
            variants = char_variants.get(ch, [])
            out.append(random.choice(variants) if variants else ch)
        else:
            out.append(ch)
    if random.random() < face_prob:
        pos = random.randint(0, len(out))
        out.insert(pos, random.choice(face_variants))
    return ''.join(out)

# =========================================================
# 4. フレーズ拡張
# =========================================================
auto_suffixes = [
    "ですね","だね","じゃないの","{}だよ","あんた{}","ほんと{}","もう{}","さすが{}","ねえ{}",
    "{}なのかな","{}でしょ","{}だよね","{}だってば","{}だわ","{}かもね","{}なんだね","{}なのだ","{}だな"
]

def expand_auto_phrases(text, multiplier=8):
    return [(s.format(text) if "{}" in s else text + s) for s in random.choices(auto_suffixes, k=multiplier)]

# =========================================================
# 5. 同義語・多様化拡張（nlpaug使用）
# =========================================================
synonym_aug = naw.SynonymAug(lang='jpn')

def generate_diverse_aug(df_subset, base_multiplier=3):
    aug_texts = []
    for _, row in tqdm(df_subset.iterrows(), total=len(df_subset)):
        orig = row['text']
        label = row['labels']
        length = len(orig)

        # 文長・有害度に応じて倍率調整
        if row['is_toxic']:
            multiplier = base_multiplier + 2
        elif length <= 10:
            multiplier = base_multiplier + 1
        else:
            multiplier = base_multiplier

        for _ in range(multiplier):
            aug_choice = random.choice(['char', 'synonym'])
            if aug_choice == 'char':
                aug_text = augment_text(orig)
            else:
                try:
                    aug_text = synonym_aug.augment(orig)
                except:
                    aug_text = augment_text(orig)
            aug_texts.append({'text': aug_text, 'labels': label})

        # 短文のみ語尾フレーズ追加
        if len(orig) <= 10:
            for phrase in expand_auto_phrases(orig, multiplier=4):
                aug_texts.append({'text': phrase, 'labels': label})

    return pd.DataFrame(aug_texts)

# =========================================================
# 6. 拡張データ生成
# =========================================================
aug_toxic_df = generate_diverse_aug(toxic_df, base_multiplier=4)
aug_non_toxic_df = generate_diverse_aug(non_toxic_df, base_multiplier=3)

# 結合 + 重複削除
final_df = pd.concat([df[['text', 'labels']], aug_toxic_df, aug_non_toxic_df], ignore_index=True)
final_df = final_df.drop_duplicates(subset=['text']).sample(frac=1.0, random_state=seed).reset_index(drop=True)

print(f"最終データ数: {len(final_df)}")  # 目標: 約10万件（データによる）

# =========================================================
# 7. Dataset変換＆分割
# =========================================================
dataset = Dataset.from_pandas(final_df).train_test_split(test_size=0.1, seed=seed)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# =========================================================
# 8. トークナイザー＆モデル
# =========================================================
model_name = "cl-tohoku/bert-base-japanese-whole-word-masking"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True, max_length=128)

train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.map(tokenize_function, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.config.hidden_dropout_prob = 0.3  # dropout増加で過学習緩和

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"accuracy": accuracy_score(labels, preds), "f1": f1_score(labels, preds)}

# =========================================================
# 9. 学習設定
# =========================================================
date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"/content/drive/MyDrive/toxicity_model_{date_str}"
os.makedirs(output_dir, exist_ok=True)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.02,
    learning_rate=2e-5,
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
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# =========================================================
# 10. 学習実行
# =========================================================
print("🔄 学習を開始します...")
trainer.train()
print("✅ 学習完了")

# =========================================================
# 11. モデル保存
# =========================================================
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"✅ モデル保存完了: {output_dir}")

# =========================================================
# 12. ONNX変換
# =========================================================
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
        "logits": {0: "batch"}
    }
)
print(f"✅ ONNX変換完了: {output_dir}/toxic-bert-jp.onnx")
