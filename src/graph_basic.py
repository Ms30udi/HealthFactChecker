from langgraph.graph import StateGraph, END
from state import FactCheckState
from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="llama3.1:latest")

def extract_claims(state: FactCheckState) -> FactCheckState:
    prompt = f"""Extract factual health claims from Instagram:
{state['text']}

Return ONLY: ["claim1", "claim2"]"""
    claims_text = llm.invoke(prompt)
    claims = [c.strip().strip('", ') for c in claims_text.split(',')]
    print(f"Extracted claims: {claims}")
    return {"claims": claims[:3]}

workflow = StateGraph(FactCheckState)
workflow.add_node("extract", extract_claims)
workflow.set_entry_point("extract")
workflow.add_edge("extract", END)
app = workflow.compile()

if __name__ == "__main__":
    result = app.invoke({"text": "Vaccines cause autism! Vitamin C cures cancer!", "claims": [], "evidence": [], "verdicts": []})
    print("Claims found:", result["claims"])

# Add after extract_claims node
def retrieve_evidence(state):
    evidence = []
    for claim in state["claims"]:
        docs = db.similarity_search(claim, k=2)
        ev = f"Claim: {claim}\nFound: {[d.metadata['source'] for d in docs]}"
        evidence.append(ev)
    print(f"Evidence for {len(evidence)} claims")
    return {"evidence": evidence}

workflow.add_node("retrieve", retrieve_evidence)
workflow.add_edge("extract", "retrieve")
workflow.add_edge("retrieve", END)
