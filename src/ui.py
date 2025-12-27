"""
HealthFactCheck - Streamlit UI
Refactored to use LangGraph workflow for fact-checking.
"""

import streamlit as st
import json
from pathlib import Path
from datetime import datetime
from instagram_audio_extractor import extract_reel_transcript
from fact_checker import fact_check_claims, generate_summary, run_full_workflow
import plotly.graph_objects as go


# Page config
st.set_page_config(
    page_title="HealthFactCheck",
    page_icon="🏥",
    layout="wide"
)


# Title
st.title("🏥 HealthFactCheck")
st.markdown("AI-powered fact-checker for health-related Instagram Reels in Moroccan Darija")


# Language selection
language = st.selectbox(
    "🌐 Output Language / لغة العرض",
    options=["العربية (Arabic)", "English", "Français"],
    index=0
)


# Extract language code
lang_map = {
    "العربية (Arabic)": "ar",
    "English": "en",
    "Français": "fr"
}
selected_lang = lang_map[language]


# Load history
HISTORY_FILE = Path("analysis_history.json")


def load_history():
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


# Main interface
with st.container():
    url = st.text_input("🔗 Instagram Reel URL", placeholder="https://www.instagram.com/reel/...")
    
    col1, col2 = st.columns([1, 5])
    with col1:
        analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)


if analyze_btn and url:
    with st.spinner("⏳ Processing..."):
        try:
            # Step 1: Extract transcript (uses Gemini/Groq Whisper)
            st.info("📝 Extracting transcript...")
            transcript = extract_reel_transcript(url)
            
            # Step 2: Run LangGraph workflow
            # Option A: Use individual functions (more control)
            st.info("📋 Generating summary...")
            summary = generate_summary(transcript, language=selected_lang)
            
            st.info("🔍 Fact-checking claims...")
            result = fact_check_claims(transcript, language=selected_lang)
            
            # Option B: Use full workflow (uncomment to use)
            # st.info("🔄 Running LangGraph workflow...")
            # workflow_result = run_full_workflow(url, transcript, language=selected_lang)
            # summary = workflow_result.get("summary", "")
            # result = {"claims": workflow_result.get("verified_claims", [])}
            
            # Save to history
            history = load_history()
            history.insert(0, {
                "url": url,
                "transcript": transcript,
                "summary": summary,
                "result": result,
                "language": selected_lang,
                "timestamp": datetime.now().isoformat()
            })
            save_history(history[:50])  # Keep last 50
            
            # Display results
            st.success("✅ Analysis Complete!")
            
            # Video Summary Section
            st.markdown("## 📝 Video Summary")
            st.markdown(f"**{summary}**")
            st.divider()
            
            # Transcript
            with st.expander("📄 Full Transcript", expanded=False):
                st.text_area("Transcript", transcript, height=150, disabled=True, label_visibility="hidden")
            
            # Claims and fact-check
            if "claims" in result and result["claims"]:
                st.markdown("## 📋 Extracted Claims")
                
                # Display claims count
                total_claims = len(result["claims"])
                correct_count = sum(1 for c in result["claims"] if c.get("verdict") == "correct")
                incorrect_count = total_claims - correct_count
                
                # Accuracy pie chart with inverted colors
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    if total_claims > 0:
                        # Create pie chart data
                        fig = go.Figure(data=[go.Pie(
                            labels=['Correct', 'Incorrect'],
                            values=[correct_count, incorrect_count],
                            marker=dict(colors=['#22c55e', '#ef4444']),  # Green for correct, Red for incorrect
                            hole=0.3
                        )])
                        
                        fig.update_layout(
                            title="Accuracy Distribution",
                            showlegend=True,
                            height=300,
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.metric("Total Claims", total_claims)
                    st.metric("✅ Correct", correct_count)
                    st.metric("❌ Incorrect", incorrect_count)
                
                st.divider()
                
                # Separate correct and incorrect claims
                correct_claims = [c for c in result["claims"] if c.get("verdict") == "correct"]
                incorrect_claims = [c for c in result["claims"] if c.get("verdict") == "incorrect"]
                
                # Display correct claims
                if correct_claims:
                    st.markdown("### ✅ What's Correct")
                    for claim in correct_claims:
                        claim_text = claim.get("claim", "")
                        explanation = claim.get("explanation", "")
                        
                        # Clean up explanation to remove repetitive claim
                        if "(CLAIM:" in explanation or "(Claim:" in explanation:
                            explanation = explanation.split("(CLAIM:")[0].split("(Claim:")[0].strip()
                        
                        with st.container():
                            st.markdown(f"**{claim_text}**")
                            st.markdown(f"_{explanation}_")
                            st.divider()
                
                # Display incorrect claims
                if incorrect_claims:
                    st.markdown("### ❌ What's Incorrect")
                    for claim in incorrect_claims:
                        claim_text = claim.get("claim", "")
                        explanation = claim.get("explanation", "")
                        
                        # Clean up explanation to remove repetitive claim
                        if "(CLAIM:" in explanation or "(Claim:" in explanation:
                            explanation = explanation.split("(CLAIM:")[0].split("(Claim:")[0].strip()
                        
                        with st.container():
                            st.markdown(f"**{claim_text}**")
                            st.markdown(f"_{explanation}_")
                            st.divider()
            else:
                st.warning("⚠️ No health claims detected in this video")
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")


# Sidebar - History
with st.sidebar:
    st.markdown("## 📚 Analysis History")
    
    history = load_history()
    
    if history:
        for idx, item in enumerate(history[:10]):  # Show last 10
            with st.expander(f"📊 Analysis {idx + 1}", expanded=False):
                st.markdown(f"**🔗 URL:** [{item['url'][:30]}...]({item['url']})")
                st.markdown(f"**📅 Date:** {item.get('timestamp', 'N/A')[:19].replace('T', ' ')}")
                st.markdown(f"**🌐 Language:** {item.get('language', 'ar').upper()}")
                
                # Show summary if available
                if "summary" in item:
                    st.markdown(f"**📝 Summary:**")
                    st.markdown(f"_{item['summary'][:120]}..._")
                
                # Show stats
                if "result" in item and "claims" in item["result"]:
                    total = len(item["result"]["claims"])
                    correct = sum(1 for c in item["result"]["claims"] if c.get("verdict") == "correct")
                    incorrect = total - correct
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Total", total)
                    with col_b:
                        accuracy = int((correct / total * 100)) if total > 0 else 0
                        st.metric("Accuracy", f"{accuracy}%")
                
                st.divider()
                
                # View Full Analysis Button
                if st.button(f"👁️ View Full Analysis", key=f"view_{idx}", use_container_width=True):
                    # Store selected analysis in session state
                    st.session_state['selected_analysis'] = item
                    st.rerun()
                    
    else:
        st.info("No analysis history yet")
    
    # Clear history button
    if st.button("🗑️ Clear History", use_container_width=True):
        save_history([])
        st.rerun()


# Display selected analysis from history
if 'selected_analysis' in st.session_state and st.session_state['selected_analysis']:
    item = st.session_state['selected_analysis']
    
    st.markdown("---")
    st.markdown("## 📂 Viewing Saved Analysis")
    
    # Close button
    if st.button("❌ Close", key="close_analysis"):
        st.session_state['selected_analysis'] = None
        st.rerun()
    
    # Display the full analysis
    st.markdown(f"**🔗 URL:** [{item['url']}]({item['url']})")
    st.markdown(f"**📅 Date:** {item.get('timestamp', 'N/A')[:19].replace('T', ' ')}")
    st.markdown(f"**🌐 Language:** {item.get('language', 'ar').upper()}")
    
    # Video Summary
    if "summary" in item:
        st.markdown("## 📝 Video Summary")
        st.markdown(f"**{item['summary']}**")
        st.divider()
    
    # Full Transcript
    if "transcript" in item:
        with st.expander("📄 Full Transcript", expanded=False):
            st.text_area("Transcript", item['transcript'], height=150, disabled=True, label_visibility="hidden")
    
    # Claims
    if "result" in item and "claims" in item["result"]:
        result = item["result"]
        total_claims = len(result["claims"])
        correct_count = sum(1 for c in result["claims"] if c.get("verdict") == "correct")
        incorrect_count = total_claims - correct_count
        
        st.markdown("## 📋 Extracted Claims")
        
        # Pie chart
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if total_claims > 0:
                fig = go.Figure(data=[go.Pie(
                    labels=['Correct', 'Incorrect'],
                    values=[correct_count, incorrect_count],
                    marker=dict(colors=['#22c55e', '#ef4444']),
                    hole=0.3
                )])
                
                fig.update_layout(
                    title="Accuracy Distribution",
                    showlegend=True,
                    height=300,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.metric("Total Claims", total_claims)
            st.metric("✅ Correct", correct_count)
            st.metric("❌ Incorrect", incorrect_count)
        
        st.divider()
        
        # Claims list
        correct_claims = [c for c in result["claims"] if c.get("verdict") == "correct"]
        incorrect_claims = [c for c in result["claims"] if c.get("verdict") == "incorrect"]
        
        if correct_claims:
            st.markdown("### ✅ What's Correct")
            for claim in correct_claims:
                claim_text = claim.get("claim", "")
                explanation = claim.get("explanation", "")
                
                if "(CLAIM:" in explanation or "(Claim:" in explanation:
                    explanation = explanation.split("(CLAIM:")[0].split("(Claim:")[0].strip()
                
                with st.container():
                    st.markdown(f"**{claim_text}**")
                    st.markdown(f"_{explanation}_")
                    st.divider()
        
        if incorrect_claims:
            st.markdown("### ❌ What's Incorrect")
            for claim in incorrect_claims:
                claim_text = claim.get("claim", "")
                explanation = claim.get("explanation", "")
                
                if "(CLAIM:" in explanation or "(Claim:" in explanation:
                    explanation = explanation.split("(CLAIM:")[0].split("(Claim:")[0].strip()
                
                with st.container():
                    st.markdown(f"**{claim_text}**")
                    st.markdown(f"_{explanation}_")
                    st.divider()
