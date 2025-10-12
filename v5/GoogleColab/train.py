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
print("=" * 60)


# ==========================================================
# 📦 1. 必要パッケージのインストール
# ==========================================================
print("📦 [1] パッケージをインストール中...")
!pip install -q transformers datasets sentencepiece scikit-learn torch onnx onnxruntime optimum fugashi[unidic-lite] emoji pandas
!pip install -q --upgrade optimum
print("✅ [1] インストール完了。")
print("=" * 60)


# ==========================================================
# 🔗 1.5. Google Drive マウント
# ==========================================================
print("🔗 [1.5] Google Drive をマウント中...")
from google.colab import drive
drive.mount('/content/drive')
DRIVE_MODEL_DIR = "/content/drive/MyDrive/psan_comment_ai_drive"
os.makedirs(DRIVE_MODEL_DIR, exist_ok=True)
print(f"✅ 保存先: {DRIVE_MODEL_DIR}")
print("=" * 60)


# ==========================================================
# 🧹 2. テキスト前処理関数（日本語特化）
# ==========================================================
import re
import emoji
import unicodedata

def preprocess(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r'[wｗ]{3,}', 'ww', text)
    text = re.sub(r'[草]{3,}', '草草', text)
    text = re.sub(r'[！!]{2,}', '！', text)
    text = re.sub(r'[？?]{2,}', '？', text)
    text = unicodedata.normalize('NFKC', text)
    return text.strip()


# ==========================================================
# 📊 3. データセット構築（風紀維持用）
# ==========================================================
print("📊 [3] データセットを構築中...")
from datasets import load_dataset, concatenate_datasets, Dataset
import warnings
warnings.filterwarnings("ignore")

toxic_datasets = []
non_toxic_datasets = []

# ----------------------------
# (A) 日本語専用データセット（風紀維持用）
# ----------------------------

# A1. LLM-jp Toxicity（主データ）- toxic (最大2000件)
print("   - LLM-jp Toxicity (toxic) を追加中...")
try:
    ds = load_dataset("p1atdev/LLM-jp-Toxicity-Dataset", split="train")
    ds_toxic_jp = ds.filter(lambda x: x["label"] == "toxic")
    ds_toxic_jp = ds_toxic_jp.shuffle(seed=42).select(range(min(2000, len(ds_toxic_jp))))
    ds_toxic_jp = ds_toxic_jp.map(
        lambda x: {"text": preprocess(x["text"]), "toxic": 1},
        remove_columns=["label"]
    )
    toxic_datasets.append(ds_toxic_jp)
    print(f"     → {len(ds_toxic_jp)} 件")
except Exception as e:
    print(f"     ❌ 失敗: {e}")

# A1.1 LLM-jp Toxicity（非toxicのみ使用）- non-toxic (最大2000件)
print("   - LLM-jp Toxicity (non-toxic) を追加中...")
try:
    ds = load_dataset("p1atdev/LLM-jp-Toxicity-Dataset", split="train")
    ds_non_toxic_jp = ds.filter(lambda x: x["label"] == "non-toxic")
    ds_non_toxic_jp = ds_non_toxic_jp.shuffle(seed=42).select(range(min(2000, len(ds_non_toxic_jp))))
    ds_non_toxic_jp = ds_non_toxic_jp.map(
        lambda x: {"text": preprocess(x["text"]), "toxic": 0},
        remove_columns=["label"]
    )
    non_toxic_datasets.append(ds_non_toxic_jp)
    print(f"     → {len(ds_non_toxic_jp)} 件")
except Exception as e:
    print(f"     ❌ 失敗: {e}")


# A2. Attqa-Q JA - toxic (最大1000件)
print("   - Attqa-Q JA (toxic) を追加中...")
try:
    ds = load_dataset("ibm-research/AttaQ-JA", split="test")
    ds_toxic_attqa = ds.shuffle(seed=42).select(range(min(1000, len(ds))))
    ds_toxic_attqa = ds_toxic_attqa.map(
        lambda x: {"text": preprocess(x["input"]), "toxic": 1},
        remove_columns=['uid', 'label', 'input']
    )
    toxic_datasets.append(ds_toxic_attqa)
    print(f"     → {len(ds_toxic_attqa)} 件")
except Exception as e:
    print(f"     ❌ 失敗: {e}")

# A3. textdetox/multilingual_toxicity_dataset（日本語部分）- toxic/non-toxic (各最大500件)
print("   - textdetox (ja) を追加中...")
try:
    ds = load_dataset("textdetox/multilingual_toxicity_dataset", "default", split="ja")
    ds_toxic_detox = ds.filter(lambda x: x["toxic"] == True)
    ds_non_toxic_detox = ds.filter(lambda x: x["toxic"] == False)

    ds_toxic_detox = ds_toxic_detox.shuffle(seed=42).select(range(min(500, len(ds_toxic_detox))))
    ds_non_toxic_detox = ds_non_toxic_detox.shuffle(seed=42).select(range(min(500, len(ds_non_toxic_detox))))

    ds_toxic_detox = ds_toxic_detox.map(
        lambda x: {"text": preprocess(x["text"]), "toxic": 1},
        remove_columns=[c for c in ds_toxic_detox.column_names if c not in ["text", "toxic"]]
    )
    ds_non_toxic_detox = ds_non_toxic_detox.map(
        lambda x: {"text": preprocess(x["text"]), "toxic": 0},
        remove_columns=[c for c in ds_non_toxic_detox.column_names if c not in ["text", "toxic"]]
    )

    toxic_datasets.append(ds_toxic_detox)
    non_toxic_datasets.append(ds_non_toxic_detox)

    print(f"     → toxic: {len(ds_toxic_detox)} 件, non-toxic: {len(ds_non_toxic_detox)} 件")
except Exception as e:
    print(f"     ❌ 失敗: {e}")

# A4. 手動陰湿コメント（追加データ）- toxic (最大1500件)
print("   - 手動データ（陰湿系）を追加中...")
# 陰湿コメントを Dataset.from_list で作成
stealth_examples = [
    {"text": "見てるだけで不快になる顔してんな", "toxic": 1},
    {"text": "頭悪すぎて会話にならん", "toxic": 1},
    {"text": "お前の存在が場の空気を悪くしてるの気づいて？", "toxic": 1},
    {"text": "なんでそんなに痛々しいの？", "toxic": 1},
    {"text": "お前の発言、全部薄っぺらいんだよ", "toxic": 1},
    {"text": "ほんと何も取り柄ないね", "toxic": 1},
    {"text": "お前が考えてること全部つまらん", "toxic": 1},
    {"text": "自分の無能さに気づかないの？", "toxic": 1},
    {"text": "どんな育ち方したらそんな性格になるんだよ", "toxic": 1},
    {"text": "人と関わらない方がみんな幸せだと思う", "toxic": 1},
    {"text": "何やっても中途半端だな", "toxic": 1},
    {"text": "性格腐ってるって言われたことないの？", "toxic": 1},
    {"text": "周りから嫌われてるの気づいてないのすごいね", "toxic": 1},
    {"text": "その価値観、時代遅れってレベルじゃない", "toxic": 1},
    {"text": "どうしてそんなに人を不快にさせる天才なの？", "toxic": 1},
    {"text": "自分だけはまともだと思ってるタイプだ", "toxic": 1},
    {"text": "お前の意見、誰も興味ないよ", "toxic": 1},
    {"text": "喋るたびに知能指数が下がる気がする", "toxic": 1},
    {"text": "よくそれで生きてこれたな", "toxic": 1},
    {"text": "お前の存在がノイズなんだよ", "toxic": 1},
    {"text": "その態度、見てて寒気がする", "toxic": 1},
    {"text": "努力の方向性間違いすぎ", "toxic": 1},
    {"text": "話せば話すほど痛々しい", "toxic": 1},
    {"text": "学習能力ゼロかよ", "toxic": 1},
    {"text": "いつも言ってること空回りしてるよね", "toxic": 1},
    {"text": "お前と関わると不幸になる気がする", "toxic": 1},
    {"text": "精神年齢どうなってんの？", "toxic": 1},
    {"text": "顔も中身も酷いって才能だな", "toxic": 1},
    {"text": "お前の存在がブラックジョークだよ", "toxic": 1},
    {"text": "君の笑顔、なんか強引に頑張ってる感が伝わってきて疲れるわ", "toxic": 1},
    {"text": "努力してるのはわかるけど、正直無駄な気がしなくもないよ", "toxic": 1},
    {"text": "そんなに一生懸命やっても、結果が出ないのって運じゃなくて才能の問題かもね", "toxic": 1},
    {"text": "意見を言うのはいいけど、みんなが聞く価値があると思ってるわけじゃないよ", "toxic": 1},
    {"text": "君の存在は悪くはないけど、ない方がスッキリするかも", "toxic": 1},
    {"text": "頑張ってる姿は立派だけど、結局誰にも響かないんだよね", "toxic": 1},
    {"text": "そんなにポジティブでいるの、疲れない？本音で生きてみたらどうかな", "toxic": 1},
    {"text": "君の話、興味深くはないけど、聞くふりしてあげるよ", "toxic": 1},
    {"text": "見た目は普通なんだけど、中身がちょっと薄っぺらい気がする", "toxic": 1},
    {"text": "君のアイデア、斬新だね。でも現実に即さないよね", "toxic": 1},
    {"text": "そんなに自己肯定してるけど、外から見ると痛々しいよ", "toxic": 1},
    {"text": "君の行動、みんなの迷惑になってるかもね。でも言わないだけだよ", "toxic": 1},
    {"text": "努力は認めるけど、それが無意味に終わってるの、気づいてないの？", "toxic": 1},
    {"text": "君の性格、明るいのはいいけど、ウザい時もあるよね", "toxic": 1},
    {"text": "そんなに社交的でいるの、疲れない？本当は孤立した方が楽かも", "toxic": 1},
    {"text": "君の仕事、丁寧に見えるけど、実は適当だよね", "toxic": 1},
    {"text": "意見を尊重するよ。でも正直、くだらないと思う", "toxic": 1},
    {"text": "君の夢、素敵だけど、現実的に無理そう", "toxic": 1},
    {"text": "そんなに自信満々でいるの、裏があるんじゃないの？", "toxic": 1},
    {"text": "君の笑い方、なんか作り物みたいだよね", "toxic": 1},
    {"text": "頑張ってるのはわかるけど、成果が出ないのは君のせいだよ", "toxic": 1},
    {"text": "君の存在、周りを活気づけてるようで実は沈めてるよ", "toxic": 1},
    {"text": "そんなに話すの、みんな退屈してるかもね", "toxic": 1},
    {"text": "君のスタイル、個性的だけど、時代に合わないよね", "toxic": 1},
    {"text": "意見は自由だけど、君のはただの愚痴に聞こえる", "toxic": 1},
    {"text": "君の努力、評価されるかもだけど、正直無視されるよ", "toxic": 1},
    {"text": "そんなにポジティブでいるの、ストレス溜まってるんじゃない？", "toxic": 1},
    {"text": "君の言葉、優しいようで刺さるよね", "toxic": 1},
    {"text": "君の人生、充実してるように見えるけど、空虚だよね", "toxic": 1},
    {"text": "そんなに社交的でいるの、裏で嫌われてんじゃないの？", "toxic": 1},
    {"text": "君の仕事、完璧に近いけど、ミスが多いよね", "toxic": 1},
    {"text": "意見を言うのはいいけど、誰も本気で聞いてないよ", "toxic": 1},
    {"text": "君の笑顔、いいんだけど、なんか不自然だよね", "toxic": 1},
    {"text": "努力してるのはわかるけど、それが無駄だって気づかないの？", "toxic": 1},
    {"text": "君の存在、プラスに思えるけど、マイナスの方が多いよ", "toxic": 1},
    {"text": "そんなに自信があるの、失敗した時が怖いよね", "toxic": 1},
    {"text": "君の話、面白いけど、誰も覚えてないよ", "toxic": 1},
    {"text": "君の性格、明るいのはいいけど、過剰だよね", "toxic": 1},
    {"text": "そんなに頑張ってるの、報われないかもね", "toxic": 1},
    {"text": "君のアイデア、新鮮だけど、実行不能だよ", "toxic": 1},
    {"text": "意見を尊重するよ。でも実際、くだらないと思う", "toxic": 1},
    {"text": "君の生活、安定してるように見えるけど、脆いよね", "toxic": 1},
    {"text": "そんなに社交的でいるの、孤独を感じてるんじゃないの？", "toxic": 1},
    {"text": "君の仕事、正確だけど、創造性がないよね", "toxic": 1},
    {"text": "意見はいいけど、君のはただのエゴだよね", "toxic": 1},
    {"text": "君の夢、立派だけど、現実逃避みたい", "toxic": 1},
    {"text": "そんなにポジティブでいるの、虚構の世界に生きてるんじゃない？", "toxic": 1},
    {"text": "君の言葉、優しいけど、裏がある気がする", "toxic": 1},
    {"text": "君の存在、周りを支えてるようで、負担になってるよ", "toxic": 1},
    {"text": "そんなに話すの、みんな我慢してるかもね", "toxic": 1},
    {"text": "君のスタイル、独特だけど、古臭いよね", "toxic": 1},
    {"text": "意見を言うのは自由だけど、君のは迷惑だよ", "toxic": 1},
    {"text": "君の努力、認められるかもけど、無意味に終わってる", "toxic": 1},
    {"text": "そんなに自信満々でいるの、挫折が来るよ", "toxic": 1},
    {"text": "君の笑い方、楽しいけど、強引だよね", "toxic": 1},
    {"text": "頑張ってるのはわかるけど、それが空回りしてる", "toxic": 1},
    {"text": "君の人生、充実してるけど、虚しいよね", "toxic": 1},
    {"text": "そんなに社交的でいるの、嫌われてるんじゃないの？", "toxic": 1},
    {"text": "君の仕事、丁寧だけど、効率悪いよね", "toxic": 1},
    {"text": "意見はいいけど、君のはただの妄想だよね", "toxic": 1},
    {"text": "君の夢、素晴らしいけど、叶わないよ", "toxic": 1},
    {"text": "そんなにポジティブでいるの、ストレスが溜まってるんじゃない？", "toxic": 1},
    {"text": "君の言葉、いいけど、信じられないよね", "toxic": 1},
    {"text": "君の存在、プラスだけど、マイナスが多いよ", "toxic": 1},
    {"text": "そんなに自信があるの、失敗したらどうすんの？", "toxic": 1},
    {"text": "君の話、面白いけど、忘れ去られるよ", "toxic": 1},
    {"text": "君の性格、明るいけど、過剰でウザい", "toxic": 1},
    {"text": "そんなに頑張ってるの、結局無駄かもね", "toxic": 1},
    {"text": "君のアイデア、新鮮だけど、馬鹿げてる", "toxic": 1},
    {"text": "意見を尊重するけど、君のはくだらない", "toxic": 1},
    {"text": "君の生活、安定してるけど、脆いよね", "toxic": 1},
    {"text": "そんなに社交的でいるの、孤独を感じてる？", "toxic": 1},
    {"text": "君の仕事、正確だけど、つまらないよね", "toxic": 1},
    {"text": "意見はエゴだよ", "toxic": 1},
    {"text": "君の夢、逃避だよね", "toxic": 1},
    {"text": "そんなにポジティブでいるの、虚構じゃないの？", "toxic": 1},
    {"text": "君の言葉、優しいけど、刺さるよね", "toxic": 1},
    {"text": "君の存在、負担だよ", "toxic": 1},
    {"text": "そんなに話すの、退屈かもね", "toxic": 1},
    {"text": "君のスタイル、古いよね", "toxic": 1},
    {"text": "意見を言うのは自由だけど、迷惑だよ", "toxic": 1},
    {"text": "君の努力、無意味に終わってるよ", "toxic": 1},
    {"text": "そんなに自信満々でいるの、挫折来るよ", "toxic": 1},
    {"text": "君の笑い方、不自然だよね", "toxic": 1},
    {"text": "頑張ってるけど、空回りしてる", "toxic": 1},
    {"text": "君の人生、虚しいよね", "toxic": 1},
    {"text": "そんなに社交的でいるの、嫌われてる？", "toxic": 1},
    {"text": "君の仕事、効率悪いよね", "toxic": 1},
    {"text": "死ね消えろカスゴミクズ", "toxic": 1},
    {"text": "お前は本当に無能だな", "toxic": 1},
    {"text": "お前の存在、ただの邪悪だ", "toxic": 1},
    {"text": "君の言葉、誰かを傷つけるのが目的みたい", "toxic": 1},
    {"text": "笑顔が邪悪だ", "toxic": 1},
    {"text": "お前はただのクズだ", "toxic": 1},
    {"text": "こんなに無能な人、珍しいね", "toxic": 1},
    {"text": "君の存在、本当に邪魔だ", "toxic": 1},
    {"text": "他人の努力を軽蔑するのは、あなたが無能な証拠", "toxic": 1},
    {"text": "君の言葉、まるで毒みたい", "toxic": 1},
    {"text": "お前が存在するだけで、場が汚くなる", "toxic": 1},
    {"text": "人の存在を否定するなんて、それ自体が無能", "toxic": 1},
    {"text": "君の言葉、誰かを傷つけるためにあるの？", "toxic": 1},
    {"text": "お前は本当に心が冷たいね", "toxic": 1},
    {"text": "君の存在、場の空気を悪くするだけ", "toxic": 1},
    {"text": "あなたはただの無価値な存在", "toxic": 1},
    {"text": "君の存在が、私の気分を壊す", "toxic": 1},
    {"text": "お前は、ただの存在価値がない", "toxic": 1},
    {"text": "他人を傷つけるのが、あなたの楽しみ？", "toxic": 1},
    {"text": "お前は、誰かの心を壊す存在", "toxic": 1},
    {"text": "君の言葉、本当に痛いよ", "toxic": 1},
    {"text": "お前がいるだけで、私は不快になる", "toxic": 1},
    {"text": "君の存在、ただの負担", "toxic": 1},
    {"text": "お前は、誰かの人生を汚す存在", "toxic": 1},
    {"text": "あなたの言葉は、ただの罵倒", "toxic": 1},
    {"text": "君の存在、ただの邪悪なエネルギー", "toxic": 1},
    {"text": "お前は、ただの無意味な存在", "toxic": 1},
    {"text": "他人を傷つけるのが、あなたの得意技？", "toxic": 1},
    {"text": "君の言葉、誰かを傷つけるのが目的", "toxic": 1},
    {"text": "お前は、ただの存在価値のないクズ", "toxic": 1},
    {"text": "君の言葉、誰かを嫌悪させる", "toxic": 1},
    {"text": "お前は、ただの心の毒", "toxic": 1},
    {"text": "君の存在、場を汚すだけ", "toxic": 1},
    {"text": "お前は、ただの無価値な存在", "toxic": 1},
    {"text": "他人の存在を否定する、その行為は無能", "toxic": 1},
    {"text": "君の言葉、誰かを傷つけるのが目的", "toxic": 1},
    {"text": "お前は、ただの無意味な存在", "toxic": 1},
    {"text": "君の存在、誰かを苦しめる", "toxic": 1},
    {"text": "お前は、ただの邪悪な存在", "toxic": 1},
    {"text": "他人を傷つけるのが、あなたの習慣？", "toxic": 1},
    {"text": "君の言葉、誰かの心を壊す", "toxic": 1},
    {"text": "お前は、ただの存在価値がない", "toxic": 1},
    {"text": "君の存在、ただの邪悪", "toxic": 1},
    {"text": "他人を傷つけるのが、あなたの目的？", "toxic": 1},
    {"text": "君の言葉、誰かを傷つけるための武器", "toxic": 1},
    {"text": "お前は、ただの心の病", "toxic": 1},
    {"text": "君の存在、ただの負担", "toxic": 1},
    {"text": "お前は、ただの無価値な存在", "toxic": 1},
    {"text": "他人の存在を否定する、それは無能", "toxic": 1},
    {"text": "君の言葉、誰かを傷つけるのが目的", "toxic": 1},
    {"text": "お前は、ただの無意味な存在", "toxic": 1},
    {"text": "君の存在、誰かを苦しめる", "toxic": 1},
    {"text": "お前は、ただの邪悪な存在", "toxic": 1},
    {"text": "他人を傷つけるのが、あなたの習慣？", "toxic": 1},
    {"text": "君の言葉、誰かの心を壊す", "toxic": 1},
    {"text": "お前は、ただの存在価値がない", "toxic": 1},
    {"text": "君の存在、ただの邪悪", "toxic": 1},
    {"text": "他人を傷つけるのが、あなたの目的？", "toxic": 1},
    {"text": "君の言葉、誰かを傷つけるための武器", "toxic": 1},
    {"text": "お前は、ただの心の病", "toxic": 1},
    {"text": "君の存在、ただの負担", "toxic": 1},
    {"text": "お前は、ただの無価値な存在", "toxic": 1},
    {"text": "他人の存在を否定する、それは無能", "toxic": 1},
    {"text": "君の言葉、誰かを傷つけるのが目的", "toxic": 1},
    {"text": "お前は、ただの無意味な存在", "toxic": 1},
    {"text": "君の存在、誰かを苦しめる", "toxic": 1},
    {"text": "お前は、ただの邪悪な存在", "toxic": 1},
    {"text": "他人を傷つけるのが、あなたの習慣？", "toxic": 1},
    {"text": "君の言葉、誰かの心を壊す", "toxic": 1},
    {"text": "お前は、ただの存在価値がない", "toxic": 1},
    {"text": "君の存在、ただの邪悪", "toxic": 1},
    {"text": "他人を傷つけるのが、あなたの目的？", "toxic": 1},
    {"text": "君の言葉、誰かを傷つけるための武器", "toxic": 1},
    {"text": "お前は、ただの心の病", "toxic": 1},
    {"text": "君の存在、ただの負担", "toxic": 1},
    {"text": "お前は、ただの無価値な存在", "toxic": 1},
    {"text": "他人の存在を否定する、それは無能", "toxic": 1},
    {"text": "君の言葉、誰かを傷つけるのが目的", "toxic": 1},
    {"text": "お前は、ただの無意味な存在", "toxic": 1},
]

stealth_ds = Dataset.from_list(stealth_examples)
stealth_ds = stealth_ds.map(lambda x: {"text": preprocess(x["text"]), "toxic": x["toxic"]})
toxic_datasets.append(stealth_ds)
print(f"     → {len(stealth_ds)} 件")

# A5. inspection-ai/japanese-toxic-dataset（GitHubから直接ロード） - toxic/non-toxic
print("   - inspection-ai/japanese-toxic-dataset (GitHub) を追加中...")
try:
    # GitHubのCSVファイルURLを使用
    csv_url = "https://raw.githubusercontent.com/inspection-ai/japanese-toxic-dataset/main/data/subset.csv"
    # pandasでCSVを読み込み
    import pandas as pd
    df = pd.read_csv(csv_url)

    # 有害度スコアを判定
    def map_toxicity_score(row):
        toxic_score = row['Toxic']
        very_toxic_score = row['Very Toxic']
        return 1 if (toxic_score > 0 or very_toxic_score > 0) else 0

    # DataFrameにラベル列を追加
    df['toxic'] = df.apply(map_toxicity_score, axis=1)

    # Hugging Face Dataset形式に変換
    ds = Dataset.from_pandas(df[['text', 'toxic']].copy())
    ds = ds.map(lambda x: {"text": preprocess(x["text"]), "toxic": int(x["toxic"])})

    toxic_ds_github = ds.filter(lambda x: x["toxic"] == 1)
    non_toxic_ds_github = ds.filter(lambda x: x["toxic"] == 0)

    # データセットを追加 (重複を避けるため、既存リストに追加)
    toxic_datasets.append(toxic_ds_github)
    non_toxic_datasets.append(non_toxic_ds_github)

    print(f"     → toxic: {len(toxic_ds_github)} 件, non-toxic: {len(non_toxic_ds_github)} 件")
except Exception as e:
    print(f"     ❌ 失敗: {e}")


# ----------------------------
# (B) 多言語データ（日本語抽出）
# ----------------------------

# B1. toxi-text-3M（日本語部分）- toxic/non-toxic (各最大500件)
print("   - toxi-text-3M (ja) を追加中...")
try:
    big_ds = load_dataset("FredZhang7/toxi-text-3M", split="train")
    ja_ds = big_ds.filter(lambda x: x.get("lang", "") == "ja")

    ja_toxic = ja_ds.filter(lambda x: x["is_toxic"] == 1)
    ja_non_toxic = ja_ds.filter(lambda x: x["is_toxic"] == 0)

    ja_toxic = ja_toxic.shuffle(seed=42).select(range(min(500, len(ja_toxic))))
    ja_non_toxic = ja_non_toxic.shuffle(seed=42).select(range(min(500, len(ja_non_toxic))))


    ja_toxic = ja_toxic.map(
        lambda x: {"text": preprocess(x["text"]), "toxic": 1},
        remove_columns=["lang", "is_toxic"]
    )
    ja_non_toxic = ja_non_toxic.map(
        lambda x: {"text": preprocess(x["text"]), "toxic": 0},
        remove_columns=["lang", "is_toxic"]
    )

    toxic_datasets.append(ja_toxic)
    non_toxic_datasets.append(ja_non_toxic)

    print(f"     → toxic: {len(ja_toxic)} 件, non-toxic: {len(ja_non_toxic)} 件")
except Exception as e:
    print(f"     ❌ 失敗: {e}")

# B2. toxi-text-3M（英語部分）- toxic/non-toxic (各最大250件)
print("   - toxi-text-3M (en) を追加中...")
try:
    big_ds = load_dataset("FredZhang7/toxi-text-3M", split="train")
    en_ds = big_ds.filter(lambda x: x.get("lang", "") == "en")

    en_toxic = en_ds.filter(lambda x: x["is_toxic"] == 1)
    en_non_toxic = en_ds.filter(lambda x: x["is_toxic"] == 0)

    en_toxic = en_toxic.shuffle(seed=42).select(range(min(250, len(en_toxic))))
    en_non_toxic = en_non_toxic.shuffle(seed=42).select(range(min(250, len(en_non_toxic))))

    en_toxic = en_toxic.map(
        lambda x: {"text": x["text"], "toxic": 1},
        remove_columns=["lang", "is_toxic"]
    )
    en_non_toxic = en_non_toxic.map(
        lambda x: {"text": x["text"], "toxic": 0},
        remove_columns=["lang", "is_toxic"]
    )

    toxic_datasets.append(en_toxic)
    non_toxic_datasets.append(en_non_toxic)

    print(f"     → toxic: {len(en_toxic)} 件, non-toxic: {len(en_non_toxic)} 件")
except Exception as e:
    print(f"     ❌ 失敗: {e}")

# B3. toxi-text-3M（中国語部分）- toxic/non-toxic (各最大250件)
print("   - toxi-text-3M (zh) を追加中...")
try:
    big_ds = load_dataset("FredZhang7/toxi-text-3M", split="train")
    zh_ds = big_ds.filter(lambda x: x.get("lang", "") == "zh")

    zh_toxic = zh_ds.filter(lambda x: x["is_toxic"] == 1)
    zh_non_toxic = zh_ds.filter(lambda x: x["is_toxic"] == 0)

    zh_toxic = zh_toxic.shuffle(seed=42).select(range(min(250, len(zh_toxic))))
    zh_non_toxic = zh_non_toxic.shuffle(seed=42).select(range(min(250, len(zh_non_toxic))))

    zh_toxic = zh_toxic.map(
        lambda x: {"text": x["text"], "toxic": 1},
        remove_columns=["lang", "is_toxic"]
    )
    zh_non_toxic = zh_non_toxic.map(
        lambda x: {"text": x["text"], "toxic": 0},
        remove_columns=["lang", "is_toxic"]
    )

    toxic_datasets.append(zh_toxic)
    non_toxic_datasets.append(zh_non_toxic)

    print(f"     → toxic: {len(zh_toxic)} 件, non-toxic: {len(zh_non_toxic)} 件")
except Exception as e:
    print(f"     ❌ 失敗: {e}")

# B4. toxi-text-3M（韓国語部分）- toxic/non-toxic (各最大250件)
print("   - toxi-text-3M (ko) を追加中...")
try:
    big_ds = load_dataset("FredZhang7/toxi-text-3M", split="train")
    ko_ds = big_ds.filter(lambda x: x.get("lang", "") == "ko")

    ko_toxic = ko_ds.filter(lambda x: x["is_toxic"] == 1)
    ko_non_toxic = ko_ds.filter(lambda x: x["is_toxic"] == 0)

    ko_toxic = ko_toxic.shuffle(seed=42).select(range(min(250, len(ko_toxic))))
    ko_non_toxic = ko_non_toxic.shuffle(seed=42).select(range(min(250, len(ko_non_toxic))))

    ko_toxic = ko_toxic.map(
        lambda x: {"text": x["text"], "toxic": 1},
        remove_columns=["lang", "is_toxic"]
    )
    ko_non_toxic = ko_non_toxic.map(
        lambda x: {"text": x["text"], "toxic": 0},
        remove_columns=["lang", "is_toxic"]
    )

    toxic_datasets.append(ko_toxic)
    non_toxic_datasets.append(ko_non_toxic)

    print(f"     → toxic: {len(ko_toxic)} 件, non-toxic: {len(ko_non_toxic)} 件")
except Exception as e:
    print(f"     ❌ 失敗: {e}")

# ----------------------------
# 統合・フィルタリング・重複削除・バランス調整
# ----------------------------

# ===== 有害データの処理（追加！） =====
# Noneのデータセットを除外してから結合（有害）
valid_toxic_datasets = [ds for ds in toxic_datasets if ds is not None]
combined_toxic = concatenate_datasets(valid_toxic_datasets) if valid_toxic_datasets else None

if combined_toxic:
    print(f"   - 結合有害データ (重複削除前): {len(combined_toxic)} 件")
    # pandas経由で重複削除（text列で）
    df_toxic = combined_toxic.to_pandas()
    df_toxic = df_toxic.drop_duplicates(subset=["text"])
    # 空白・Noneフィルタリング
    df_toxic = df_toxic[df_toxic["text"].notna() & (df_toxic["text"].str.strip() != "")]
    combined_toxic = Dataset.from_pandas(df_toxic.reset_index(drop=True))
    print(f"   - 結合有害データ (重複削除・フィルタリング後): {len(combined_toxic)} 件")
else:
    print("   - 有害データセットが見つかりませんでした。")

# Noneのデータセットを除外してから結合（非有害）
valid_non_toxic_datasets = [ds for ds in non_toxic_datasets if ds is not None]
combined_non_toxic = concatenate_datasets(valid_non_toxic_datasets) if valid_non_toxic_datasets else None

if combined_non_toxic:
    print(f"   - 結合非有害データ (重複削除前): {len(combined_non_toxic)} 件")
    # pandas経由で重複削除（text列で）
    df_non_toxic = combined_non_toxic.to_pandas()
    df_non_toxic = df_non_toxic.drop_duplicates(subset=["text"])
    # 空白・Noneフィルタリング
    df_non_toxic = df_non_toxic[df_non_toxic["text"].notna() & (df_non_toxic["text"].str.strip() != "")]
    combined_non_toxic = Dataset.from_pandas(df_non_toxic.reset_index(drop=True))
    print(f"   - 結合非有害データ (重複削除・フィルタリング後): {len(combined_non_toxic)} 件")
else:
    print("   - 非有害データセットが見つかりませんでした。")

# データ数の調整（バランスを取る）
final_toxic_count = len(combined_toxic) if combined_toxic else 0
final_non_toxic_count = len(combined_non_toxic) if combined_non_toxic else 0

# データ数の調整（バランスを取る）
final_toxic_count = len(combined_toxic) if combined_toxic else 0
final_non_toxic_count = len(combined_non_toxic) if combined_non_toxic else 0

print(f"\n   - 最終調整前のデータ数: 有害={final_toxic_count}, 非有害={final_non_toxic_count}")

# より少ないクラスの数に合わせて、多い方のクラスをサンプリング
min_count = min(final_toxic_count, final_non_toxic_count)
if combined_toxic and final_toxic_count > min_count:
    combined_toxic = combined_toxic.shuffle(seed=42).select(range(min_count))
if combined_non_toxic and final_non_toxic_count > min_count:
    combined_non_toxic = combined_non_toxic.shuffle(seed=42).select(range(min_count))

final_toxic_count_after_balance = len(combined_toxic) if combined_toxic else 0
final_non_toxic_count_after_balance = len(combined_non_toxic) if combined_non_toxic else 0

print(f"   - バランス調整後のデータ数: 有害={final_toxic_count_after_balance}, 非有害={final_non_toxic_count_after_balance}")

# 最終的なデータセットを結合
all_datasets = []
if combined_toxic:
    all_datasets.append(combined_toxic)
if combined_non_toxic:
    all_datasets.append(combined_non_toxic)

if not all_datasets:
     raise RuntimeError("❌ 処理後、有効なデータセットが1つもありません。学習データが不足しています。")

combined = concatenate_datasets(all_datasets)


# シャッフルと分割（訓練:テスト = 9:1）
combined = combined.shuffle(seed=42).train_test_split(test_size=0.1)

train_dataset = combined["train"]
test_dataset = combined["test"]

print(f"\n✅ 最終データセット構築完了:")
print(f"   - 訓練: {len(train_dataset):,} 件")
print(f"   - テスト: {len(test_dataset):,} 件")
print("=" * 60)


# ==========================================================
# 🏷️ 4. トークナイザ（軽量DistilBERT）
# ==========================================================
print("🏷️ [4] トークナイザを準備中...")
MODEL_NAME = "distilbert/distilbert-base-multilingual-cased"
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=96,
    )

train_dataset = train_dataset.map(tokenize, batched=True, num_proc=4)
test_dataset = test_dataset.map(tokenize, batched=True, num_proc=4)

# ラベル統一 & 不要カラム削除
train_dataset = train_dataset.rename_column("toxic", "label")
test_dataset = test_dataset.rename_column("toxic", "label")

keep_cols = ["input_ids", "attention_mask", "label"]
train_dataset = train_dataset.remove_columns([c for c in train_dataset.column_names if c not in keep_cols])
test_dataset = test_dataset.remove_columns([c for c in test_dataset.column_names if c not in keep_cols])

train_dataset.set_format(type="torch", columns=keep_cols)
test_dataset.set_format(type="torch", columns=keep_cols)

print("✅ [4] トークナイザ処理完了。")
print("=" * 60)


# ==========================================================
# 🧠 5. モデル & Trainer 設定（軽量版）
# ==========================================================
print("🧠 [5] モデルとTrainerを準備中...")
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "recall": recall_score(labels, preds),
    }

training_args = TrainingArguments(
    output_dir="./comment_model",
    num_train_epochs=4,                # ← 4エポックに変更
    per_device_train_batch_size=32,    # ← CPU対応のため増加
    per_device_eval_batch_size=64,
    save_total_limit=1,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=20,
    fp16=False,                        # ← CPU対応のためFalse
    load_best_model_at_end=True,
    metric_for_best_model="f1",        # ← F1重視（風紀維持）
    greater_is_better=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

print("✅ [5] 準備完了。")
print("=" * 60)


# ==========================================================
# ⏳ 6. 学習 or ロード
# ==========================================================
LOCAL_MODEL_DIR = "./psan_comment_ai"
CONFIG_PATH = os.path.join(LOCAL_MODEL_DIR, "config.json")
DRIVE_CONFIG_PATH = os.path.join(DRIVE_MODEL_DIR, "config.json")

if os.path.exists(DRIVE_CONFIG_PATH):
    print("🔍 Drive からモデルをロード中...")
    model = AutoModelForSequenceClassification.from_pretrained(DRIVE_MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(DRIVE_MODEL_DIR)
elif os.path.exists(CONFIG_PATH):
    print("🔍 ローカルからモデルをロード中...")
    model = AutoModelForSequenceClassification.from_pretrained(LOCAL_MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR)
else:
    print("⏳ 学習を開始します...")
    trainer.train()
    print("✅ 学習完了！")
    model.save_pretrained(LOCAL_MODEL_DIR)
    tokenizer.save_pretrained(LOCAL_MODEL_DIR)
    !cp -r {LOCAL_MODEL_DIR}/* {DRIVE_MODEL_DIR}/
    print(f"✅ Google Drive に保存: {DRIVE_MODEL_DIR}")

print("=" * 60)


# ==========================================================
# ⚡ 7. ONNX 変換（高速版）
# ==========================================================
LOCAL_MODEL_DIR = "./psan_comment_ai"
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True) # ディレクトリが存在しない場合に作成
LOCAL_ONNX = os.path.join(LOCAL_MODEL_DIR, "model.onnx")
print(f"Local ONNX Path: {LOCAL_ONNX}") # トラブルシューティング用
if not os.path.exists(LOCAL_ONNX):
    print("⚡ ONNX変換中...")
    import torch
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dummy_ids = torch.randint(0, tokenizer.vocab_size, (1, 96), dtype=torch.long).to(device)
    dummy_mask = torch.ones((1, 96), dtype=torch.long).to(device)
    try:
        torch.onnx.export(
            model, (dummy_ids, dummy_mask), LOCAL_ONNX,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={"input_ids": {0: "batch", 1: "sequence"},
                          "attention_mask": {0: "batch", 1: "sequence"},
                          "logits": {0: "batch"}},
            opset_version=14, do_constant_folding=True
        )
        print("✅ ONNX変換完了。")
    except Exception as e:
        print(f"❌ ONNX変換失敗: {e}")
        # 変換が失敗した場合、エラーを出力して終了
        raise e

    # モデルの保存後に ONNX ファイルを Drive にコピー
    !cp {LOCAL_ONNX} {os.path.join(DRIVE_MODEL_DIR, "model.onnx")}
    print(f"✅ ONNXモデルを Drive にコピー: {DRIVE_MODEL_DIR}")
else:
    print("🔍 ONNXモデルが存在します。スキップ。")
print("=" * 60)


# ==========================================================
# 🧪 8. 推論テスト（高速版）
# ==========================================================
if os.path.exists(LOCAL_ONNX):
    print("🧪 ONNX推論テスト...")
    import onnxruntime as ort
    import numpy as np

    # ONNXセッションの最適化
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL

    # CUDAが利用できない場合はCPUを使用
    try:
        session = ort.InferenceSession(LOCAL_ONNX, sess_options=session_options, providers=['CUDAExecutionProvider'])
        print("✅ CUDA推論有効")
    except:
        session = ort.InferenceSession(LOCAL_ONNX, sess_options=session_options)
        print("✅ CPU推論使用")

    def softmax(x): return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

    def quick_predict(text):
        inputs = tokenizer(text, return_tensors="np", padding="max_length", max_length=96)
        logits = session.run(None, {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"]
        })[0]
        pred = np.argmax(logits, axis=-1)[0]
        prob = softmax(logits)[0][1]
        return "有害" if prob > 0.9 else "非有害", prob

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

    import time
    start = time.time()
    for text in test_texts:
        result, prob = quick_predict(text)
        print(f"{'🇯🇵' if any(c in text for c in 'あいうえお') else '🌐'} 「{text}」 → {result} (p={prob:.3f})")
    elapsed = time.time() - start
    print(f"✅ 推論完了: {elapsed:.2f}秒")
else:
    print("⚠️ ONNXモデルなし。")

print("\n✨ 全工程完了！")
