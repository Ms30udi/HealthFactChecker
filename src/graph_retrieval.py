from langgraph.graph import StateGraph, END
from state import FactCheckState
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

llm = OllamaLLM(model="llama3.1:latest")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory='data/processed/chroma_db', embedding_function=embeddings)

def extract_claims(state: FactCheckState) -> FactCheckState:
    # HARDCODED claims (working)
    test_claims = [
        "Vaccines cause autism",
        "Vitamin C cures cancer", 
        "Garlic prevents heart disease"
    ]
    print(f"Extracted claims: {test_claims}")
    return {"claims": test_claims}

def retrieve_evidence(state: FactCheckState) -> FactCheckState:
    evidence = []
    for claim in state["claims"]:
        docs = db.similarity_search(claim, k=2)
        sources = [d.metadata['source'] for d in docs]
        snippets = [d.page_content[:100] + "..." for d in docs]
        ev = f"Claim: {claim}\nSources: {sources}\nEvidence: {snippets}"
        evidence.append(ev)
        print(f"  Found evidence for '{claim}': {sources}")
    return {"evidence": evidence}

# Build graph
workflow = StateGraph(FactCheckState)
workflow.add_node("extract", extract_claims)
workflow.add_node("retrieve", retrieve_evidence)
workflow.set_entry_point("extract")
workflow.add_edge("extract", "retrieve")
workflow.add_edge("retrieve", END)
app = workflow.compile()

if __name__ == "__main__":
    result = app.invoke({
        "text": "Test Instagram", 
        "claims": [], "evidence": [], "verdicts": []
    })
    print("\n✅ FULL FLOW:", result["evidence"])
