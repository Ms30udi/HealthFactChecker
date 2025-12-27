"""
HealthFactCheck - LangChain + LangGraph Implementation
Refactored from custom Python orchestration to LangGraph StateGraph workflow.

Workflow: START → transcribe → extract_claims → verify_claims (loop) → generate_summary → END
"""

import os
import json
from typing import TypedDict, Annotated, Optional
from dotenv import load_dotenv

# LangChain imports
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Load environment variables
load_dotenv()


# ==============================================================================
# STATE DEFINITION
# ==============================================================================

class FactCheckState(TypedDict):
    """State for the fact-checking workflow."""
    url: str
    transcript: str
    claims: list[str]
    verified_claims: list[dict]  # [{"claim": str, "verdict": str, "explanation": str}]
    summary: str
    language: str  # ar, en, fr
    error: Optional[str]


# ==============================================================================
# LANGCHAIN COMPONENTS
# ==============================================================================

# Initialize ChatGroq LLM (consistent settings)
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.0,
    api_key=os.getenv("GROQ_API_KEY"),
    model_kwargs={"seed": 42}
)

# Summary LLM (slightly higher temperature for natural summaries)
summary_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
    max_tokens=300
)

# Embeddings (same as ingest_kb.py)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ChromaDB vectorstore
CHROMA_PATH = "data/processed/chroma_db"
vectorstore = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings
)

# Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# ==============================================================================
# PROMPT TEMPLATES
# ==============================================================================

EXTRACT_CLAIMS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "أنت خبير في استخراج المعلومات الصحية من النصوص العربية والدارجة المغربية."),
    ("human", """استخرج كل المعلومات الصحية والطبية من هذا النص.

النص:
{transcript}

أعطني قائمة بالادعاءات الصحية فقط، كل ادعاء في سطر واحد.
لا تضف أي تعليقات أو شروحات، فقط الادعاءات.

مثال للصيغة:
- الفيتامين د مهم للعظام
- الماء يساعد على الهضم
""")
])


def get_verify_claim_messages(claim: str, context: str, lang_instruction: str) -> list:
    """Generate verification messages with context (avoids template variable issues)."""
    human_message = f"""You are a medical expert. Verify this health claim:

Claim: {claim}

Medical reference information:
{context if context else "No reference information available"}

{lang_instruction}

Respond ONLY with valid JSON in this exact format:
{{
    "verdict": "correct",
    "explanation": "brief explanation in one sentence"
}}

OR

{{
    "verdict": "incorrect",
    "explanation": "brief explanation in one sentence"
}}

Do not include any text outside the JSON object."""
    
    return [
        ("system", "أنت طبيب خبير متخصص في التحقق من المعلومات الصحية. أجب دائماً بصيغة JSON صحيحة."),
        ("human", human_message)
    ]


def get_summary_prompt(transcript: str, language: str) -> ChatPromptTemplate:
    """Generate summary prompt based on language."""
    lang_prompts = {
        "ar": "لخص هذا النص في 2-3 جمل، مع التركيز على المواضيع الصحية الرئيسية:",
        "en": "Summarize this text in 2-3 sentences, focusing on the main health topics:",
        "fr": "Résumez ce texte en 2-3 phrases, en vous concentrant sur les principaux sujets de santé:"
    }
    
    return ChatPromptTemplate.from_messages([
        ("system", "أنت خبير في تلخيص المحتوى الصحي بشكل واضح ومختصر."),
        ("human", f"{lang_prompts.get(language, lang_prompts['ar'])}\n\n{transcript}")
    ])


# ==============================================================================
# LANGGRAPH NODE FUNCTIONS
# ==============================================================================

def extract_claims_node(state: FactCheckState) -> dict:
    """
    Node: Extract health claims from transcript.
    Uses LangChain ChatGroq with structured prompt.
    """
    print("[DEBUG] Extracting claims...")
    
    try:
        transcript = state.get("transcript", "")
        
        if not transcript:
            return {"claims": [], "error": "No transcript provided"}
        
        # Create chain
        chain = EXTRACT_CLAIMS_PROMPT | llm | StrOutputParser()
        
        # Invoke chain
        response = chain.invoke({"transcript": transcript})
        
        # Parse claims (each line starting with - or number)
        claims = []
        for line in response.split('\n'):
            line = line.strip()
            if line and (line.startswith('-') or line.startswith('*') or (line and line[0].isdigit())):
                # Remove bullet points and numbers
                claim = line.lstrip('-*0123456789. ').strip()
                if claim:
                    claims.append(claim)
        
        print(f"[DEBUG] Found {len(claims)} claims")
        return {"claims": claims}
        
    except Exception as e:
        print(f"[ERROR] Claim extraction failed: {e}")
        return {"claims": [], "error": str(e)}


def verify_claims_node(state: FactCheckState) -> dict:
    """
    Node: Verify each claim against knowledge base using RAG.
    Uses LangChain retriever + ChatGroq.
    """
    print("[DEBUG] Verifying claims...")
    
    claims = state.get("claims", [])
    language = state.get("language", "ar")
    
    if not claims:
        return {"verified_claims": []}
    
    # Language instructions
    lang_instructions = {
        "ar": "أجب بالعربية فقط",
        "en": "Respond in English only",
        "fr": "Répondez en français uniquement"
    }
    lang_instruction = lang_instructions.get(language, lang_instructions["ar"])
    
    # Error messages
    error_messages = {
        "ar": "غير قادر على التحقق من هذا الادعاء حالياً",
        "en": "Unable to verify this claim at the moment",
        "fr": "Impossible de vérifier cette affirmation pour le moment"
    }
    
    verified_claims = []
    
    for idx, claim_text in enumerate(claims):
        print(f"[DEBUG] Verifying claim {idx+1}/{len(claims)}...")
        
        try:
            # RAG: Retrieve relevant documents
            print(f"[DEBUG] Searching KB for: {claim_text[:50]}...")
            docs = retriever.invoke(claim_text)
            
            # Build context from retrieved documents
            context = ""
            if docs:
                context = "\n\n".join(doc.page_content for doc in docs)
            
            # Create verification messages (avoids template variable issues)
            messages = get_verify_claim_messages(claim_text, context, lang_instruction)
            
            # Invoke LLM directly with messages and JSON response format
            json_llm = llm.bind(response_format={"type": "json_object"})
            response = json_llm.invoke(messages)
            response_text = response.content
            print(f"[DEBUG] Groq response: {response_text[:100]}")
            
            # Parse JSON response
            result = json.loads(response_text)
            
            verified_claims.append({
                "claim": claim_text,
                "verdict": result.get("verdict", "incorrect"),
                "explanation": result.get("explanation", "غير متأكد")
            })
            
        except Exception as e:
            print(f"[ERROR] Verification failed for claim: {e}")
            verified_claims.append({
                "claim": claim_text,
                "verdict": "incorrect",
                "explanation": error_messages.get(language, error_messages["ar"])
            })
    
    return {"verified_claims": verified_claims}


def generate_summary_node(state: FactCheckState) -> dict:
    """
    Node: Generate summary of the transcript.
    Uses LangChain ChatGroq.
    """
    print("[DEBUG] Generating summary...")
    
    transcript = state.get("transcript", "")
    language = state.get("language", "ar")
    
    if not transcript:
        return {"summary": "No transcript available."}
    
    try:
        # Create chain
        summary_prompt = get_summary_prompt(transcript, language)
        chain = summary_prompt | summary_llm | StrOutputParser()
        
        # Invoke chain
        summary = chain.invoke({})
        
        return {"summary": summary.strip()}
        
    except Exception as e:
        print(f"[ERROR] Summary generation failed: {e}")
        return {"summary": "Could not generate summary."}


# ==============================================================================
# LANGGRAPH WORKFLOW
# ==============================================================================

def build_fact_check_graph() -> StateGraph:
    """
    Build the LangGraph StateGraph workflow.
    
    Flow:
    START → extract_claims → verify_claims → generate_summary → END
    """
    # Create graph builder
    builder = StateGraph(FactCheckState)
    
    # Add nodes
    builder.add_node("extract_claims", extract_claims_node)
    builder.add_node("verify_claims", verify_claims_node)
    builder.add_node("generate_summary", generate_summary_node)
    
    # Define edges
    builder.add_edge(START, "extract_claims")
    builder.add_edge("extract_claims", "verify_claims")
    builder.add_edge("verify_claims", "generate_summary")
    builder.add_edge("generate_summary", END)
    
    # Compile graph
    return builder.compile()


# Compile the workflow (module-level for reuse)
fact_check_workflow = build_fact_check_graph()


# ==============================================================================
# PUBLIC API (backwards compatible)
# ==============================================================================

def fact_check_claims(transcript: str, language: str = "ar") -> dict:
    """
    Main function to fact-check health claims from a transcript.
    Backwards compatible with original API.
    
    Args:
        transcript: Video transcript in Arabic/Darija
        language: Output language (ar, en, fr)
        
    Returns:
        dict with 'claims' list containing verified claims
    """
    try:
        # Initialize state
        initial_state: FactCheckState = {
            "url": "",
            "transcript": transcript,
            "claims": [],
            "verified_claims": [],
            "summary": "",
            "language": language,
            "error": None
        }
        
        # Run only extract + verify nodes (summary handled separately in UI)
        # But we can invoke the partial graph or just call nodes directly
        
        # Extract claims
        extract_result = extract_claims_node(initial_state)
        initial_state["claims"] = extract_result.get("claims", [])
        
        if not initial_state["claims"]:
            return {"claims": [], "message": "No health claims found"}
        
        # Verify claims
        verify_result = verify_claims_node(initial_state)
        verified_claims = verify_result.get("verified_claims", [])
        
        return {"claims": verified_claims}
        
    except Exception as e:
        print(f"[ERROR] Fact checking failed: {e}")
        return {"claims": [], "error": str(e)}


def run_full_workflow(url: str, transcript: str, language: str = "ar") -> dict:
    """
    Run the complete LangGraph workflow.
    
    Args:
        url: Instagram Reel URL
        transcript: Video transcript
        language: Output language (ar, en, fr)
        
    Returns:
        Complete state with all results
    """
    initial_state: FactCheckState = {
        "url": url,
        "transcript": transcript,
        "claims": [],
        "verified_claims": [],
        "summary": "",
        "language": language,
        "error": None
    }
    
    # Invoke the compiled workflow
    result = fact_check_workflow.invoke(initial_state)
    
    return result


# ==============================================================================
# STANDALONE SUMMARY FUNCTION (for UI compatibility)
# ==============================================================================

def generate_summary(transcript: str, language: str = "ar") -> str:
    """
    Generate a summary of the video transcript.
    Standalone function for backwards compatibility.
    
    Args:
        transcript: Video transcript
        language: Output language (ar, en, fr)
        
    Returns:
        Summary string
    """
    state: FactCheckState = {
        "url": "",
        "transcript": transcript,
        "claims": [],
        "verified_claims": [],
        "summary": "",
        "language": language,
        "error": None
    }
    
    result = generate_summary_node(state)
    return result.get("summary", "Could not generate summary.")


# ==============================================================================
# TEST
# ==============================================================================

if __name__ == "__main__":
    # Test the workflow
    test_transcript = """
    الفيتامين د مهم بزاف للعظام ديالك.
    خاصك تشرب الما بزاف كل يوم، على الأقل 8 كيسان.
    الثوم كيساعد على خفض ضغط الدم.
    """
    
    print("=" * 60)
    print("Testing LangGraph Fact-Check Workflow")
    print("=" * 60)
    
    # Test backwards compatible function
    result = fact_check_claims(test_transcript, language="ar")
    print("\nResults:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    
    # Test full workflow
    print("\n" + "=" * 60)
    print("Testing Full Workflow")
    print("=" * 60)
    
    full_result = run_full_workflow(
        url="https://instagram.com/reel/test",
        transcript=test_transcript,
        language="ar"
    )
    print("\nFull Result:")
    print(f"Claims: {len(full_result.get('claims', []))}")
    print(f"Verified: {len(full_result.get('verified_claims', []))}")
    print(f"Summary: {full_result.get('summary', 'N/A')[:100]}...")
