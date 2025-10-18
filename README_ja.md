# BlockWordAI - 日本語毒性検出

BlockWordAIは、最先端の日本語コンテンツ毒性検出システムであり、高度なAIと日本語の言語文脈の微細な理解を組み合わせています。
インターフェースは日本語にローカライズされており、日本語ユーザーにより良いユーザーエクスペリエンスを提供します。

## ✨ 主な特徴

- **日本語最適化**: 文化的文脈を理解した日本語テキストに特化して訓練されています
- **軽量**: 効率的なデプロイと推論に最適化されています
- **高精度**: 日本語毒性検出用に微調整されたBERTベースのアーキテクチャを使用
- **簡単な統合**: 既存システムへの統合が容易なAPI
- **日本語ローカライズUI**: ユーザーインターフェース要素とコンソールメッセージは日本語で表示されます

## 🚀 クイックスタート

### ZIP配布版 (本番用)

1. ZIPファイルを希望のディレクトリに展開
2. Python 3.8+ をインストール (まだの場合は)
3. 仮想環境を作成し、依存関係をインストール:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windowsでは: venv\Scripts\activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. 学習済みモデルファイルを `./models/latest/` ディレクトリに配置
   - ディレクトリには以下のファイルが必要: `toxic-bert-jp.onnx`, `tokenizer_config.json`, `tokenizer.json`, およびその他のモデルファイル
   - 学習済みモデルがない場合は、最初にモデルを学習させる必要があります

5. 環境変数を設定 (オプション、以下を参照) またはデフォルトを使用
6. APIサーバーを起動:
   ```bash
   python api_server.py
   ```

### 前提条件

- Python 3.8以上 (pip付き)
- 少なくとも2GBの空きディスク容量
- モデル操作には4GB以上のRAMを推奨

### 環境変数

以下の環境変数を使用してアプリケーションを設定できます:

| 変数名 | 説明 | デフォルト |
|----------|-------------|---------|
| `BLOCKWORDAI_MODEL_PATH` | 推論用の学習済みモデルへのパス | `./models/latest` |
| `BLOCKWORDAI_LANGUAGE` | UIローカライズの言語 | `ja_JP` |
| `PORT` | APIサーバーのポート | `8000` |
| `HOST` | APIサーバーのホスト | `0.0.0.0` |

例: `PORT=9000 BLOCKWORDAI_MODEL_PATH=./my_model_dir python api_server.py`

### モデルファイルの配置

アプリケーションが正しく動作するためには、学習済みモデルファイルを指定のディレクトリに配置する必要があります:
- デフォルトモデルディレクトリ: `./models/latest/`
- モデルディレクトリに必要なファイル:
  - `toxic-bert-jp.onnx` - ONNXモデルファイル
  - `tokenizer_config.json` - トークナイザー設定
  - `tokenizer.json` - トークナイザーファイル
  - モデルと一緒に保存されたその他のファイル

## 📚 API利用方法

### コマンドライン利用方法

```bash
# デモを実行
python app.py --demo

# 単一テキストの毒性を予測
python app.py --predict "素晴らしいですね！"
```

### Web API利用方法

BlockWordAIはREST APIサーバーも提供します：

```bash
# APIサーバーを起動
python api_server.py

# またはDockerで
docker-compose up -d
```

APIサーバーは `http://localhost:8000` で利用可能です

#### エンドポイント:

- `GET /` - API情報と利用可能なエンドポイント
- `POST /api/toxicity` - 単一テキストの毒性検出
- `POST /api/batch-toxicity` - 一括毒性検出
- `GET /api/health` - ヘルスチェック

### Python利用方法

```python
from app import predict_toxicity

# 毒性を予測 (モデルは自動的に読み込まれます)
result = predict_toxicity("素晴らしいですね！")
print(result)
# 出力: {'text': '素晴らしいですね！', 'toxic_probability': 0.1234, 'is_toxic': False, 'classification': 'non-toxic'}

# または複数の予測用に事前に読み込まれたモデルで
from app import load_model
tokenizer, session = load_model()
result = predict_toxicity("素晴らしいですね！", tokenizer, session)
print(result)
```

## 🏗️ プロジェクト構造

```
BlockWordAI/
├── app.py                 # メインアプリケーションインターフェース
├── api_server.py          # Web APIサーバー
├── localization.py        # ローカライゼーションユーティリティ
├── translations/          # 翻訳ファイル
│   ├── messages.pot      # 翻訳テンプレート
│   └── ja_JP/            # 日本語翻訳
├── Dockerfile            # Docker設定
├── docker-compose.yml    # Docker Compose設定
├── requirements.txt      # Python依存関係
├── VERSION.json          # バージョン情報
└── README.md             # 英語版README
```

## 🌐 環境変数

| 変数名 | 説明 | デフォルト |
|----------|-------------|---------|
| `BLOCKWORDAI_MODEL_PATH` | 推論用の学習済みモデルへのパス | `./models/latest` |
| `BLOCKWORDAI_LANGUAGE` | UIローカライズの言語 | `ja_JP` |

## 🌍 ローカライゼーション

BlockWordAIはAPIレスポンスとコンソールメッセージの両方で日本語ローカライズをサポートしています。アプリケーションはFlask-Babelを国際化に使用しており、将来追加の言語を追加することが容易です。

### サポートされている言語
- 日本語 (ja_JP) - デフォルト
- 英語 (en_US) - フォールバック

### 新しい言語を追加するには
1. `translations`ディレクトリに新しい翻訳ファイルを作成
2. 既存のパターンに従って翻訳を追加
3. `pybabel compile -d translations -l [language_code]` を使用して翻訳ファイルをコンパイル
4. 新しい言語を使用するように `BLOCKWORDAI_LANGUAGE` 環境変数を更新

## 📦 サンプルコード

### Python
```python
from app import predict_toxicity

result = predict_toxicity("素晴らしいですね！")
print(result)
```

### JavaScript (fetch)
```javascript
async function checkToxicity(text) {
  // これはAPIエンドポイントを呼び出します
  const response = await fetch('/api/toxicity', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ text: text })
  });
  
  const result = await response.json();
  return result;
}

checkToxicity("素晴らしいですね！").then(result => {
  console.log(result);
});
```

### curl
```bash
curl -X POST http://localhost:8000/api/toxicity \
  -H "Content-Type: application/json" \
  -d '{"text": "素晴らしいですね！"}'
```

## 🏷️ バージョン管理とリリース

- 現在のバージョン: 1.0.0 ([VERSION.json](VERSION.json)を参照)
- リリースにはセマンティックバージョニングを使用
- 安定版リリースは[Releases](https://github.com/your-repo/releases)ページを参照
- 各リリースには学習済みモデルとバイナリが含まれます

## 📄 ライセンス

このプロジェクトは専有ソフトウェアです - 詳細は[LICENSE](LICENSE)ファイルを参照してください。

- **商用利用**: ✅ 許可されています (購入が必要)
- **修正**: ❌ 禁止されています
- **配布**: ❌ 禁止されています
- **個人利用**: ✅ 許可されています
- **販売**: ✅ 著者のみ許可

## 🌟 競争優位性

- **日本語のニュアンス理解**: 日本語の文化的および言語的ニュアンスに最適化されています
- **辞書 + AIハイブリッド**: 包括的な検出のために従来の辞書と現代のAIを組み合わせています
- **軽量モデル**: より高速なデプロイと推論のための小型モデル
- **WordPressプラグイン互換性**: コンテンツ管理システムへの簡単な統合を設計

## ❗ サポートポリシー

このソフトウェアは「現状のまま」提供され、サポート、保証、またはメンテナンスの約束は一切ありません。
購入後はサポートを提供せず、今後の更新や修正も行いません。
ユーザーには、一切のクレームなく現在の状態でソフトウェアを受け入れることが期待されます。

## 🙏 謝辞

- cl-tohokuの日本語BERTモデル
- モデル最適化のためのONNX
- Transformersライブラリ

---

**BlockWordAI** - 先進のAIで日本語コンテンツモデレーションにより安全なオンライン空間を