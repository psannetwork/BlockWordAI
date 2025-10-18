# BlockWordAI - Japanese Toxicity Detection

BlockWordAI is a state-of-the-art Japanese content toxicity detection system that combines advanced AI with nuanced understanding of Japanese language context.
The interface has been localized in Japanese to provide a better user experience for Japanese users.

## ✨ Key Features

- **Japanese Language Optimized**: Specifically trained on Japanese text with cultural context awareness
- **Lightweight**: Optimized for efficient deployment and inference
- **High Accuracy**: Uses BERT-based architecture fine-tuned for Japanese toxicity detection
- **Easy Integration**: Simple API for integration into existing systems
- **Japanese Localized UI**: User interface elements and console messages are available in Japanese

## 🚀 Quick Start

### For ZIP Distribution (Production)

1. Extract the ZIP file to your desired directory
2. Install Python 3.8+ if not already installed
3. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Place your trained model files in the `./models/latest/` directory
   - The directory should contain: `toxic-bert-jp.onnx`, `tokenizer_config.json`, `tokenizer.json`, and other model files
   - If you don't have a trained model yet, you'll need to train one first

5. Set environment variables (optional, see below) or use defaults
6. Start the API server:
   ```bash
   python api_server.py
   ```

### Prerequisites

- Python 3.8+ with pip
- At least 2GB free disk space
- 4GB+ RAM recommended for model operations

### Environment Variables

You can configure the application using these environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `BLOCKWORDAI_MODEL_PATH` | Path to trained model for inference | `./models/latest` |
| `BLOCKWORDAI_LANGUAGE` | Language for UI localization | `ja_JP` |
| `PORT` | Port for the API server | `8000` |
| `HOST` | Host for the API server | `0.0.0.0` |

Example: `PORT=9000 BLOCKWORDAI_MODEL_PATH=./my_model_dir python api_server.py`

### Model Files Placement

For the application to work correctly, you need to place your trained model files in the designated directory:
- Default model directory: `./models/latest/`
- Required files in the model directory:
  - `toxic-bert-jp.onnx` - The ONNX model file
  - `tokenizer_config.json` - Tokenizer configuration
  - `tokenizer.json` - Tokenizer file
  - Any other files that were saved with the model

### Web API Usage

BlockWordAI also provides a REST API server:

```bash
# Start the API server
python api_server.py

# The API server will be available at http://localhost:8000
```

## 📚 API Usage

### Command Line Usage

```bash
# Run demonstration
python app.py --demo

# Predict toxicity of a single text
python app.py --predict "素晴らしいですね！"
```

### Web API Usage

BlockWordAI also provides a REST API server:

```bash
# Start the API server
python api_server.py

# Or with Docker
docker-compose up -d
```

The API server will be available at `http://localhost:8000`

#### Endpoints:

- `GET /` - API information and available endpoints
- `POST /api/toxicity` - Single text toxicity detection
- `POST /api/batch-toxicity` - Batch toxicity detection
- `GET /api/health` - Health check

### Python Usage

```python
from app import predict_toxicity

# Predict toxicity (model loads automatically if not provided)
result = predict_toxicity("素晴らしいですね！")
print(result)
# Output: {'text': '素晴らしいですね！', 'toxic_probability': 0.1234, 'is_toxic': False, 'classification': 'non-toxic'}

# Or with pre-loaded model for multiple predictions
from app import load_model
tokenizer, session = load_model()
result = predict_toxicity("素晴らしいですね！", tokenizer, session)
print(result)
```

## 🏗️ Project Structure

```
BlockWordAI/
├── app.py                 # Main application interface
├── api_server.py          # Web API server
├── localization.py        # Localization utilities
├── translations/          # Translation files
│   ├── messages.pot      # Translation template
│   └── ja_JP/            # Japanese translations
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose configuration
├── requirements.txt      # Python dependencies
├── VERSION.json          # Version information
├── README.md             # English documentation
└── README_ja.md          # Japanese documentation
```

## 🌐 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `BLOCKWORDAI_MODEL_PATH` | Path to trained model for inference | `./models/latest` |
| `BLOCKWORDAI_LANGUAGE` | Language for UI localization | `ja_JP` |

## 🌍 Localization

BlockWordAI now supports Japanese localization for both API responses and console messages. The application uses a custom localization system, making it easy to add additional languages in the future.

### Supported Languages
- Japanese (ja_JP) - default
- English (en_US) - fallback

### Adding New Languages
1. Add your translations to the `localization.py` file
2. Update the translations dictionary with your new language
3. Update the `BLOCKWORDAI_LANGUAGE` environment variable to use your new language

## 📦 Sample Code

### Python
```python
from app import predict_toxicity

result = predict_toxicity("素晴らしいですね！")
print(result)
```

### JavaScript (fetch)
```javascript
async function checkToxicity(text) {
  // This would call your API endpoint
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

## 🏷️ Versioning & Releases

- Current version: 1.0.0 (see [VERSION.json](VERSION.json))
- We use semantic versioning for releases
- Check the [Releases](https://github.com/your-repo/releases) page for stable versions
- Each release includes pre-trained models and binaries

## 📄 License

This project is proprietary software - see the [LICENSE](LICENSE) file for details.

- **Commercial Use**: ✅ Allowed (with purchase)
- **Modification**: ❌ Prohibited
- **Distribution**: ❌ Prohibited 
- **Private Use**: ✅ Allowed
- **Sale**: ✅ Allowed (by author only)

## 🌟 競争優位性

- **日本語のニュアンス理解**: 日本語の文化的および言語的ニュアンスに最適化されています
- **辞書 + AIハイブリッド**: 包括的な検出のために従来の辞書と現代のAIを組み合わせています
- **軽量モデル**: より高速なデプロイと推論のための小型モデル
- **WordPressプラグイン互換性**: コンテンツ管理システムへの簡単な統合を設計

## ❗ Support Policy

This software is provided "as is" without any support, warranty, or maintenance commitment. 
After purchase, no support will be provided and no further updates or fixes will be issued.
Users are expected to accept the software in its current condition without any complaints.

## 🙏 謝辞

- cl-tohokuの日本語BERTモデル
- モデル最適化のためのONNX
- Transformersライブラリ

---

**BlockWordAI** - Making online spaces safer with advanced AI for Japanese content moderation.