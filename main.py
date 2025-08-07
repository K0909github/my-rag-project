import os
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
    # Dev Containerの環境変数からAPIキーを読み込む
    api_key = os.environ.get("")
    if not api_key:
        print("エラー: 環境変数 'GOOGLE_API_KEY' が設定されていません。")
        return

    print("✅ APIキーを読み込みました。")

    # --- 1. ドキュメントの読み込みと分割 ---
    print("🔄 ドキュメントを読み込んでいます...")
    loader = PyPDFLoader("深層学習とIITと自由エネルギー原理_.pdf")
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    print(f"✅ ドキュメントを {len(chunks)} 個のチャンクに分割しました。")

    # --- 2. ベクトル化とベクトルストアの構築 ---
    print("🔄 チャンクをベクトル化し、データベースを構築しています...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_documents(chunks, embeddings)
    print("✅ ベクトルストアの準備が完了しました。")

    # --- 3. RAGチェーンの構築 ---
    # LLMとしてGemini Proを準備
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key, convert_system_message_to_human=True)

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

    # --- 4. 質問応答の実行 ---
    print("\n--- 質問応答を開始します ---")
    question = "このドキュメントに記載されている、製品の主な機能は何ですか？"
    
    print(f"質問: {question}")
    result = qa_chain.invoke({"query": question})

    print("\n--- 回答 ---")
    print(result["result"])
    
    print("\n--- 参考にした情報源 ---")
    for doc in result["source_documents"]:
        print(f"抜粋: {doc.page_content[:100]}...")

if __name__ == "__main__":
    main()