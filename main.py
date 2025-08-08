import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

def main():
    """
    RAGシステムを実行するメイン関数
    """
    # --- 0. APIキーの確認 ---
    # .envファイルから環境変数を読み込み
    load_dotenv()
    key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        print("❌ エラー: 環境変数 'GOOGLE_API_KEY' が設定されていません。")
        print("📝 .env ファイルを作成し、GOOGLE_API_KEY=your_api_key を設定してください。")
        return

    print("✅ APIキーを読み込みました。")

    # --- 1. ドキュメントの読み込みと分割 ---
    print("🔄 ドキュメントを読み込んでいます...")
    try:
        loader = PyPDFLoader("深層学習とIITと自由エネルギー原理_.pdf")
        documents = loader.load()
        
        if not documents:
            print("❌ エラー: PDFファイルが見つからないか、内容が空です。")
            return
            
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)
        print(f"✅ ドキュメントを {len(chunks)} 個のチャンクに分割しました。")
        
    except FileNotFoundError:
        print("❌ エラー: PDFファイル '深層学習とIITと自由エネルギー原理_.pdf' が見つかりません。")
        return
    except Exception as e:
        print(f"❌ ドキュメント読み込みエラー: {e}")
        return

    # --- 2. ベクトル化とベクトルストアの構築 ---
    print("🔄 チャンクをベクトル化し、データベースを構築しています...")
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=key)
        vector_store = FAISS.from_documents(chunks, embeddings)
        print("✅ ベクトルストアの準備が完了しました。")
    except Exception as e:
        print(f"❌ ベクトル化エラー: {e}")
        print("💡 APIキーが正しいか確認してください。")
        return

    # --- 3. RAGチェーンの構築 ---
    print("🔄 RAGチェーンを構築しています...")
    try:
        # LLMとしてGemini Proを準備
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", google_api_key=key, convert_system_message_to_human=True)

        # プロンプトのテンプレートを定義
        prompt_template = """
        提供された「コンテキスト情報」だけを参考にして、以下の「質問」に日本語で具体的に回答してください。
        コンテキスト情報に答えが見つからない場合は、「その情報は見つかりませんでした」と回答してください。

        コンテキスト情報:
        {context}

        質問: {question}
        """
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        # RAGチェーンを作成
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        print("✅ RAGチェーンの準備が完了しました。")
    except Exception as e:
        print(f"❌ RAGチェーン構築エラー: {e}")
        return

    # --- 4. 質問応答の実行 ---
    print("\n--- 質問応答を開始します ---")
    question = "このドキュメントに記載されている、統合情報理論とは何ですか?"
    
    print(f"質問: {question}")
    try:
        result = qa_chain.invoke({"query": question})

        print("\n--- 回答 ---")
        print(result["result"])
        

        def clean_text(text: str) -> str:
            # 改行や連続スペースを除去して読みやすくする
            import re
            # 改行をスペースに置換
            text = text.replace("\n", " ")
            # 複数スペースは1つに
            text = re.sub(r"\s+", " ", text)
            # 前後の空白を削除
            text = text.strip()
            return text

        # 質問応答結果の出力部分の一部を修正
        print("\n--- 参考にした情報源 ---")
        for i, doc in enumerate(result["source_documents"], 1):
            cleaned = clean_text(doc.page_content)
            print(f"{i}. 抜粋: {cleaned[:300]}...")  # 300文字まで表示

            
    except Exception as e:
        print(f"❌ 質問応答エラー: {e}")
        return
        
    print("\n🎉 RAGシステムの実行が完了しました！")

if __name__ == "__main__":
    main()