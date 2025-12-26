import json
import os
from datetime import datetime

HISTORY_FILE = "analysis_history.json"

def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_analysis(url, transcript, language, explanations, claims_count):
    history = load_history()
    
    # Check if URL already analyzed
    existing = next((h for h in history if h['url'] == url), None)
    if existing:
        # Update existing entry
        existing['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        existing['transcript'] = transcript
        existing['language'] = language
        existing['explanations'] = explanations
        existing['claims_count'] = claims_count
    else:
        # Add new entry
        history.append({
            "url": url,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "transcript": transcript,
            "language": language,
            "explanations": explanations,
            "claims_count": claims_count
        })
    
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def get_analysis_by_url(url):
    history = load_history()
    return next((h for h in history if h['url'] == url), None)
