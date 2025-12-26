import os
import hashlib
import yt_dlp
from groq import Groq
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Configure both APIs
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def download_reel_video(url: str, out_dir: str = "tmp") -> str:
    """Download full video (audio + visual for better context)"""
    os.makedirs(out_dir, exist_ok=True)
    
    url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
    out_tmpl = os.path.join(out_dir, f"reel_{url_hash}")

    ydl_opts = {
        "format": "best[ext=mp4]",  # Full video with visual context
        "outtmpl": out_tmpl + ".%(ext)s",
        "quiet": False,
        "noplaylist": True,
    }

    print(f"[DEBUG] Downloading video from: {url}")
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
    
    print(f"[DEBUG] Video saved to: {filename}")
    
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Video file not found at {filename}")
    
    return filename

def transcribe_with_gemini_multimodal(video_path: str) -> str:
    """
    Transcribe video using Gemini 2.0 Flash (better for Darija with visual context).
    Gemini sees both video + audio = better understanding.
    """
    print(f"[DEBUG] Transcribing with Gemini (multimodal): {video_path}")
    
    try:
        video_file = genai.upload_file(path=video_path)
        
        import time
        while video_file.state.name == "PROCESSING":
            print("[DEBUG] Processing video...")
            time.sleep(2)
            video_file = genai.get_file(video_file.name)
        
        if video_file.state.name == "FAILED":
            raise ValueError("Video processing failed")
        
        model = genai.GenerativeModel("gemini-2.0-flash-exp")
        
        prompt = """أنت خبير في الدارجة المغربية. استخرج النص المنطوق في هذا الفيديو بدقة عالية.

انتبه إلى:
- الكلمات الدارجة المغربية الصحيحة
- السياق البصري (ما يظهر في الفيديو)
- المصطلحات الطبية والصحية

أعطني النص الكامل فقط، بدون أي إضافات."""

        response = model.generate_content([video_file, prompt], request_options={"timeout": 600})
        
        genai.delete_file(video_file.name)
        
        return response.text.strip()
        
    except Exception as e:
        print(f"[ERROR] Gemini transcription failed: {e}")
        print("[DEBUG] Falling back to Groq Whisper...")
        return transcribe_with_groq_whisper(video_path)

def transcribe_with_groq_whisper(video_path: str) -> str:
    """Fallback to Groq Whisper if Gemini fails"""
    print(f"[DEBUG] Transcribing with Groq Whisper: {video_path}")
    
    try:
        with open(video_path, "rb") as file:
            transcription = groq_client.audio.transcriptions.create(
                file=(video_path, file.read()),
                model="whisper-large-v3-turbo",
                language="ar",
                response_format="text",
                temperature=0.0,
                prompt="هذا فيديو عن الصحة والتغذية بالدارجة المغربية"  # Context hint
            )
        
        return transcription.strip()
        
    except Exception as e:
        print(f"[ERROR] Groq Whisper transcription failed: {e}")
        raise

def extract_reel_transcript(url: str) -> str:
    try:
        video_path = download_reel_video(url)
        # Try Gemini first (better for Darija + visual context)
        text = transcribe_with_gemini_multimodal(video_path)
        return text
    except Exception as e:
        print(f"[ERROR] {e}")
        raise
