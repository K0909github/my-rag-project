# ベースとなる公式Pythonイメージを指定
FROM python:3.11-slim

# コンテナ内での作業ディレクトリを設定
WORKDIR /app

# まず要件定義ファイルだけをコピーする（キャッシュを有効活用するため）
COPY requirements.txt .

# pipをアップグレードし、要件定義ファイルを使ってライブラリをインストール
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# プロジェクトの全てのファイルをコンテナにコピー
COPY . .

# コンテナが起動したときに実行されるコマンド (開発中は不要なことが多い)
# CMD ["python", "main.py"]