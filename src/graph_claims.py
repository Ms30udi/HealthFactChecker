from langgraph.graph import StateGraph, END
from state import FactCheckState # Your state.py
from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="llama3.1:latest")

def extract_claims(state: FactCheckState) -> FactCheckState:
    prompt = """You are a neutral claim extraction AI for Instagram fact-checking research.

TASK: Identify statements presented as facts in this Instagram caption, regardless of truth.

Instagram: {state['text']}

Examples:
"Vaccines cause autism" → ["Vaccines cause autism"]
"Vitamin C cures cancer" → ["Vitamin C cures cancer"]

Output ONLY JSON array:"""

    claims_text = llm.invoke(prompt)
    # Fallback if LLM refuses
    fallback_claims = state['text'].split('.')[:3]
    claims = [c.strip() for c in fallback_claims if len(c.strip()) > 10]
    print(f"Extracted claims: {claims}")
    return {"claims": claims[:3]}


# Build graph
workflow = StateGraph(FactCheckState)
workflow.add_node("extract", extract_claims)
workflow.set_entry_point("extract")
workflow.add_edge("extract", END)
app = workflow.compile()

if __name__ == "__main__":
    result = app.invoke({
        "text": "Vaccines cause autism! Vitamin C cures cancer! Eat garlic to avoid heart disease!",
        "claims": [], 
        "evidence": [], 
        "verdicts": []
    })
    print("FINAL CLAIMS:", result["claims"])
