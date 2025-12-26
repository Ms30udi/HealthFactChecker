# Instagram Health Fact-Checker
Detects & verifies health claims in IG videos using LangGraph + local Llama RAG.

## Stack
- LLM: Llama 7B/8B (llama.cpp/Ollama)
- Orchestration: LangGraph
- Vector DB: ChromaDB
- Embeddings: sentence-transformers/all-MiniLM-L6-v2
- Input: Instagram video URL
- Output: Claims + Verdicts + Sources

## Health Sources (KB)
1. WHO fact sheets
2. CDC guidelines 
3. Mayo Clinic overviews


# HealthFactCheck 🏥

AI-powered fact-checker for health-related Instagram Reels in Moroccan Darija. Automatically transcribes Arabic/Darija videos and verifies medical claims against a scientific knowledge base.

## Features

- 🎥 **Instagram Reel Transcription** - Extracts audio from Instagram reels using Gemini 1.5 Flash
- 🔍 **Medical Claim Extraction** - Identifies health/nutrition claims in Darija
- ✅ **Fact Verification** - Cross-references claims with scientific papers
- 📊 **Interactive Dashboard** - Streamlit UI with claim analysis history
- 🧠 **RAG System** - ChromaDB vector database with medical literature

## Tech Stack

- **Frontend**: Streamlit
- **LLM**: Google Gemini 2.5 Flash (transcription + analysis)
- **Vector DB**: ChromaDB
- **Frameworks**: LangGraph, LangChain
- **Media Processing**: yt-dlp, ffmpeg

## Installation

### 1. Prerequisites

