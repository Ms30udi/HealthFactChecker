from typing import TypedDict

class FactCheckState(TypedDict):
    text: str
    claims: list[str]
    evidence: list[str]
    verdicts: str
    language: str  # Add this
