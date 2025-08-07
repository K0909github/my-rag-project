# RAGシステム with Gemini API

PDFドキュメントから情報を抽出し、質問応答を行うRAG（Retrieval Augmented Generation）システムです。

## 🚀 特徴

- **Google Gemini API** を使用した高精度な質問応答
- **LangChain** を活用したRAGパイプライン
- **FAISS** による高速ベクトル検索
- **Docker** による簡単な環境構築

## 📋 前提条件

- Docker & Docker Compose
- Google AI Studio APIキー ([取得方法](https://makersuite.google.com/app/apikey))

## 🛠️ セットアップ

### 1. リポジトリのクローン
```bash
git clone <repository-url>
cd my-rag-project
```

### 2. 環境変数の設定
```bash
# .env.example をコピーして .env を作成
cp .env.example .env

# .env ファイルを編集してAPIキーを設定
# GOOGLE_API_KEY=your_actual_api_key_here
```

### 3. Dockerコンテナのビルドと起動
```bash
# イメージをビルド
docker build -t my-rag-app .

# コンテナを起動（インタラクティブモード）
docker run -it --rm -v $(pwd):/app my-rag-app bash
```

## 🏃‍♂️ 実行方法

### Docker内でPythonスクリプトを実行
```bash
# コンテナ内で実行
python main.py
```

### または、ローカル環境で実行
```bash
# 依存関係をインストール
pip install -r requirements.txt

# スクリプトを実行
python main.py
```

## 📁 プロジェクト構造

```
my-rag-project/
├── main.py                 # メインのRAGスクリプト
├── requirements.txt        # Python依存関係
├── Dockerfile             # Docker設定
├── .env.example           # 環境変数テンプレート
├── README.md              # このファイル
└── 深層学習とIITと自由エネルギー原理_.pdf  # 対象PDFドキュメント
```

## 🔧 技術スタック

- **Python 3.11**
- **LangChain** - RAGパイプライン構築
- **Google Generative AI** - Gemini API接続
- **FAISS** - ベクトルデータベース
- **PyPDF** - PDF処理
- **Docker** - コンテナ化

## 💡 使用方法

1. システムが起動すると、PDFドキュメントを自動で読み込み
2. テキストをチャンクに分割し、ベクトル化
3. 事前定義された質問に対して回答を生成
4. 参考にした情報源も表示

質問内容は `main.py` の68行目で変更可能です：
```python
question = "統合情報理論の公理は何ですか？"
```

## 🐛 トラブルシューティング

### APIキーエラー
```
❌ エラー: 環境変数 'GOOGLE_API_KEY' が設定されていません。
```
→ `.env` ファイルにAPIキーが正しく設定されているか確認

### ImportError
→ Dockerコンテナ内で実行しているか、または `pip install -r requirements.txt` が完了しているか確認

## 📄 ライセンス

このプロジェクトはMITライセンスの下で公開されています。
