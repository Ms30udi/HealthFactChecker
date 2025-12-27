"""
HealthFactCheck - LangGraph State Definition
Centralized state for the fact-checking workflow.
"""

from typing import TypedDict, Optional


class FactCheckState(TypedDict):
    """
    State for the fact-checking workflow.
    
    Flow: START → extract_claims → verify_claims → generate_summary → END
    
    Attributes:
        url: Instagram Reel URL being analyzed
        transcript: Transcribed text from the video
        claims: List of extracted health claims
        verified_claims: List of verified claims with verdicts
        summary: Generated summary of the transcript
        language: Output language (ar, en, fr)
        error: Error message if any step fails
    """
    url: str
    transcript: str
    claims: list[str]
    verified_claims: list[dict]  # [{"claim": str, "verdict": str, "explanation": str}]
    summary: str
    language: str  # ar, en, fr
    error: Optional[str]


# Legacy compatibility - keep old fields for existing graph_*.py files
class LegacyFactCheckState(TypedDict):
    """Legacy state for backwards compatibility with existing graph files."""
    text: str
    claims: list[str]
    evidence: list[str]
    verdicts: str
    language: str
