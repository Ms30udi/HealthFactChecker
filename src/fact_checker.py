import os
import time
import json
from dotenv import load_dotenv
from groq import Groq
from chromadb import PersistentClient
from chromadb.utils import embedding_functions

# Load environment variables
load_dotenv()

# Configure Groq
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ChromaDB setup
CHROMA_PATH = "data/chroma_db"
chroma_client = PersistentClient(path=CHROMA_PATH)

# Embedding function
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

# Get or create collection
try:
    collection = chroma_client.get_collection(
        name="health_knowledge_base",
        embedding_function=embedding_function
    )
except:
    collection = chroma_client.create_collection(
        name="health_knowledge_base",
        embedding_function=embedding_function
    )

def fact_check_claims(transcript: str, language: str = "ar") -> dict:
    """
    Main function to fact-check health claims from a transcript.
    
    Args:
        transcript: Video transcript in Arabic/Darija
        language: Output language (ar, en, fr)
        
    Returns:
        dict with 'claims' list containing verified claims
    """
    try:
        # Step 1: Extract claims
        print("[DEBUG] Extracting claims...")
        claims = extract_claims(transcript)
        
        if not claims:
            return {"claims": [], "message": "No health claims found"}
        
        # Process all claims (no rate limit with Groq!)
        print(f"[DEBUG] Found {len(claims)} claims, verifying...")
        verified_claims = []
        
        for idx, claim_text in enumerate(claims):
            print(f"[DEBUG] Verifying claim {idx+1}/{len(claims)}...")
            verdict = verify_claim(claim_text, language)
            verified_claims.append({
                "claim": claim_text,
                "verdict": verdict["verdict"],
                "explanation": verdict["explanation"]
            })
        
        return {"claims": verified_claims}
        
    except Exception as e:
        print(f"[ERROR] Fact checking failed: {e}")
        return {"claims": [], "error": str(e)}

def extract_claims(transcript: str) -> list:
    """Extract health-related claims from transcript using Groq."""
    
    prompt = f"""استخرج كل المعلومات الصحية والطبية من هذا النص.

النص:
{transcript}

أعطني قائمة بالادعاءات الصحية فقط، كل ادعاء في سطر واحد.
لا تضف أي تعليقات أو شروحات، فقط الادعاءات.

مثال للصيغة:
- الفيتامين د مهم للعظام
- الماء يساعد على الهضم
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "أنت خبير في استخراج المعلومات الصحية من النصوص العربية والدارجة المغربية."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=1000,
            seed=42
        )
        
        text = response.choices[0].message.content.strip()
        
        # Parse claims (each line starting with - or number)
        claims = []
        for line in text.split('\n'):
            line = line.strip()
            if line and (line.startswith('-') or line.startswith('*') or (line and line[0].isdigit())):
                # Remove bullet points and numbers
                claim = line.lstrip('-*0123456789. ').strip()
                if claim:
                    claims.append(claim)
        
        return claims
        
    except Exception as e:
        print(f"[ERROR] Claim extraction failed: {e}")
        return []

def verify_claim(claim: str, language: str = "ar") -> dict:
    """Verify a single claim against knowledge base."""
    try:
        # Search knowledge base
        print(f"[DEBUG] Searching KB for: {claim[:50]}...")
        results = collection.query(
            query_texts=[claim],
            n_results=3
        )
        
        # Get context from KB
        context = ""
        if results and results.get('documents'):
            docs_list = results['documents']
            if docs_list and len(docs_list) > 0:
                first_result = docs_list[0]
                if first_result:
                    context = "\n\n".join(str(doc) for doc in first_result)
        
        # Language instruction
        lang_instructions = {
            "ar": "أجب بالعربية فقط",
            "en": "Respond in English only",
            "fr": "Répondez en français uniquement"
        }
        lang_instruction = lang_instructions.get(language, "أجب بالعربية فقط")
        
        prompt = f"""You are a medical expert. Verify this health claim:

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

        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "أنت طبيب خبير متخصص في التحقق من المعلومات الصحية. أجب دائماً بصيغة JSON صحيحة."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=500,
                response_format={"type": "json_object"},
                seed=42
            )
            
            text = response.choices[0].message.content.strip()
            
            print(f"[DEBUG] Groq response: {text[:100]}")
            
            # Parse JSON response
            result = json.loads(text)
            
            return {
                "verdict": result.get("verdict", "incorrect"),
                "explanation": result.get("explanation", "غير متأكد")
            }
                
        except Exception as e:
            print(f"[ERROR] Verification API call failed: {e}")
            raise e
        
    except Exception as e:
        error_msg = str(e)
        print(f"[ERROR] Verification failed for claim: {error_msg}")
        
        # Return error message in selected language
        error_messages = {
            "ar": "غير قادر على التحقق من هذا الادعاء حالياً",
            "en": "Unable to verify this claim at the moment",
            "fr": "Impossible de vérifier cette affirmation pour le moment"
        }
        
        return {
            "verdict": "incorrect",
            "explanation": error_messages.get(language, error_messages["ar"])
        }
