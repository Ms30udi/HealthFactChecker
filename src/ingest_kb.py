import os
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings  # hugging face embedder (FAST) 
from langchain_chroma import Chroma #database
from bs4 import BeautifulSoup

def load_html_simple(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')
        for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
            tag.decompose()
        text = soup.get_text(separator=' ', strip=True)
    return [Document(page_content=text[:10000], metadata={"source": file_path.name})]

# Load + chunk
docs_path = Path("data/raw")
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
all_docs = []

print("📚 Loading documents...")
for file in docs_path.glob("*.html"):
    print(f"📄 Processing {file.name}...")
    raw_docs = load_html_simple(file)
    chunks = splitter.split_documents(raw_docs)
    for chunk in chunks:
        chunk.metadata["source"] = file.name
    all_docs.extend(chunks)
    print(f"   → {len(chunks)} chunks")

print(f"TOTAL: {len(all_docs)} chunks ready")

# FAST embeddings + ChromaDB
print("⚡ FAST embedding (30s)...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
os.makedirs("data/processed", exist_ok=True)

print("💾 Building ChromaDB...")
vectorstore = Chroma.from_documents(
    all_docs, 
    embeddings, 
    persist_directory="data/processed/chroma_db"
)

print("✅ ChromaDB built! Testing...")
test_db = Chroma(persist_directory="data/processed/chroma_db", embedding_function=embeddings)
print(f"🎉 FINAL COUNT: {len(test_db.get()['ids'])} chunks stored!")
print("✅ KB READY FOR LangGraph!")
