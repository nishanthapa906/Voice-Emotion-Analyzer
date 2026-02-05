

import os
import pickle
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import librosa
import librosa.display
import sounddevice as sd
from datetime import datetime
from io import BytesIO
import soundfile as sf
from scipy.signal import butter, filtfilt
import pyttsx3
import tempfile
import firebase_admin
from firebase_admin import credentials, firestore
import pyrebase
import pytz
import base64
from PIL import Image

st.set_page_config(
    page_title="VibeCheck AI",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="collapsed"
)




SR = 22050
DURATION = 2.5
OFFSET = 0.6

EMOTIONS = {
    'angry': {'emoji': '😠', 'color': '#ef4444'},
    'disgust': {'emoji': '🤢', 'color': '#22c55e'},
    'fear': {'emoji': '😨', 'color': '#a855f7'},
    'happy': {'emoji': '😄', 'color': '#fbbf24'},
    'neutral': {'emoji': '😐', 'color': '#6b7280'},
    'sad': {'emoji': '😢', 'color': '#3b82f6'},
    'surprise': {'emoji': '😲', 'color': '#ec4899'}
}

GENDERS = {
    'male': {'emoji': '👨', 'label': 'Male'},
    'female': {'emoji': '👩', 'label': 'Female'},
    'uncertain': {'emoji': '❓', 'label': 'Uncertain'}
}

JOKES = [
    "Why don't scientists trust atoms? Because they make up everything! 😄",
    "Did you hear about the mathematician who's afraid of negative numbers? He will stop at nothing! 🤣",
    "I told my computer I needed a break, and now it won't stop sending me Kit-Kat ads. 😆",
    "Why did the scarecrow win an award? He was outstanding in his field! 🌾😄",
    "What do you call a fake noodle? An impasta! 🍝😆",
    "Why don't eggs tell jokes? They'd crack each other up! 🥚😄",
    "What did the ocean say to the beach? Nothing, it just waved! 👋😆"
]

EMOTION_SUGGESTIONS = {
    'angry': {
        'messages': [
            "Take a deep breath. You're stronger than this anger.",
            "Your feelings are valid. Channel this energy positively.",
            "Count to 10 slowly. Things will feel better soon.",
        ],
        'suggestions': [
            '🏃 Go for a workout - https://www.youtube.com/results?search_query=workout+videos',
            '🎮 Play a game - https://www.miniclip.com/games/en/',
            '🎵 Listen to calming music - https://www.spotify.com/search/calm',
            '🧘 Try breathing exercises - https://www.headspace.com/work/breathing',
            '✍️ Journal your feelings'
        ]
    },
    'disgust': {
        'messages': [
            "Your instincts are protecting you. Trust them.",
            "It's okay to avoid what doesn't feel right.",
            "Set boundaries and prioritize yourself.",
        ],
        'suggestions': [
            '🌿 Explore nature - https://www.alltrails.com/',
            '🎨 Try creative activities - https://www.skillshare.com/browse/art',
            '🧼 Organize and refresh your space',
            '🎮 Play puzzle games - https://www.aarp.org/games/',
            '☕ Enjoy something pleasant'
        ]
    },
    'fear': {
        'messages': [
            "Courage isn't the absence of fear, it's acting despite it.",
            "You've overcome fears before. You can do it again.",
            "Small steps lead to big changes.",
        ],
        'suggestions': [
            '💪 Watch motivational content - https://www.ted.com/',
            '🎮 Play confidence-building games - https://www.miniclip.com/games/en/',
            '🧘 Practice meditation - https://www.calm.com/',
            '📱 Connect with someone you trust',
            '🌟 Remember your past victories'
        ]
    },
    'happy': {
        'messages': [
            "Keep spreading this joy! You're amazing!",
            "This is the perfect moment. Enjoy it fully!",
            "Your happiness is contagious!",
        ],
        'suggestions': [
            '🎮 Play uplifting games - https://www.miniclip.com/games/en/',
            '🎵 Create a happiness playlist - https://www.spotify.com/search/happy',
            '📸 Capture this moment',
            '🤗 Share joy with others',
            '🎉 Do something fun'
        ]
    },
    'neutral': {
        'messages': [
            "Perfect state for productivity and focus.",
            "Use this calm energy to accomplish your goals.",
            "This is ideal for clear thinking.",
        ],
        'suggestions': [
            '🎯 Work on important tasks',
            '🎮 Play strategic games - https://www.miniclip.com/games/en/',
            '📚 Learn something new - https://www.coursera.org/',
            '🧠 Plan your day effectively',
            '📖 Read an interesting article'
        ]
    },
    'sad': {
        'messages': [
            "It's okay to feel sad. You're not alone.",
            "This sadness will pass. Better days are coming.",
            "Be kind to yourself. You deserve compassion.",
        ],
        'suggestions': [
            '💙 Talk to someone - https://www.betterhelp.com/',
            '🎵 Listen to healing music - https://www.spotify.com/search/sad+but+hopeful',
            '🎮 Play relaxing games - https://www.aarp.org/games/',
            '🌳 Spend time in nature',
            '📖 Read uplifting stories'
        ]
    },
    'surprise': {
        'messages': [
            "What an interesting surprise!",
            "Take time to process what happened.",
            "This teaches you something valuable.",
        ],
        'suggestions': [
            '🎮 Play mystery games - https://www.miniclip.com/games/en/',
            '💭 Reflect and journal',
            '🎵 Explore new music - https://www.spotify.com/search/discover',
            '🗣️ Share your experience',
            '😲 Embrace the moment'
        ]
    }
}

st.markdown("""
<style>
            
            <link rel="icon" href="why-is-khusi-ban-everywhere-in-kathmandu-v0-fqpcf3l7tl2g1.webp">
    /* ============================================================
       FONTS
       ============================================================ */
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=Inter:wght@300;400;500;600&display=swap');

    /* ============================================================
       RESET + BASE
       ============================================================ */
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }

    body, .stApp {
        font-family: 'Inter', sans-serif;
        background: #080810;
        color: #e2e2e8;
        min-height: 100vh;
        overflow-x: hidden;
    }

    /* ============================================================
       HIDE STREAMLIT CHROME
       ============================================================ */
    #MainMenu, footer, .stDeployButton { display: none !important; }
    .stApp { padding: 0 !important; }

    /* ============================================================
       AMBIENT BACKGROUND — subtle floating orbs, no full-page noise
       ============================================================ */
    .stApp::before {
        content: '';
        position: fixed;
        inset: 0;
        z-index: 0;
        background:
            radial-gradient(ellipse 340px 340px at 18% 22%, rgba(99,102,241,0.13) 0%, transparent 70%),
            radial-gradient(ellipse 280px 280px at 82% 75%, rgba(168,85,247,0.10) 0%, transparent 70%),
            radial-gradient(ellipse 200px 200px at 55% 10%, rgba(236,72,153,0.07) 0%, transparent 70%);
        pointer-events: none;
    }

            
            .bg-overlay {
    pointer-events: none;
}

    /* ============================================================
       AUTH PAGE — vertically + horizontally centered compact card
       ============================================================ */
    .auth-wrapper {
            margin-top:-200px;
        position: relative;
        z-index: 1;
        min-height: 100vh;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 2rem 1rem;
    }

    /* SVG logo area */
    .auth-logo {
        margin-bottom: 1.4rem;
    }

    .auth-logo svg {
        width: 56px;
        height: 56px;
        filter: drop-shadow(0 0 18px rgba(99,102,241,0.45));
    }

    /* Title */
    .auth-title {
        font-family: 'Syne', sans-serif, bold;
        font-size: 1.95rem;
        font-weight: 900;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #a78bfa 0%, #c084fc 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.25rem;
    }

    .auth-subtitle {
        font-size: 0.78rem;
        font-weight: 400;
        color: #52525e;
        text-align: center;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 1.8rem;
    }
/* Auth box with background image */
/* Auth box with background image */
.auth-box {
    width: 100%;
    max-width: 420px;
    height: 400px;

    /* Background image + gradient overlay */
    background: 
        linear-gradient(
            rgba(20, 20, 28, 0.75),
            rgba(20, 20, 28, 0.75)
        ),
        url("https://i.redd.it/fqpcf3l7tl2g1.jpeg"); /* Use reliable Imgur link */

    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;

    backdrop-filter: blur(28px);
    -webkit-backdrop-filter: blur(28px);
    border: 1px solid rgba(99,102,241,0.14);
    border-radius: 20px;
    padding: 2.4rem 2.2rem 1.8rem;

    box-shadow:
        0 2px 6px rgba(0,0,0,0.25),
        0 16px 48px rgba(0,0,0,0.35),
        inset 0 1px 0 rgba(255,255,255,0.04);

    position: relative; 
    z-index: 20; /* Important: above .stApp::before */
}


    /* ============================================================
       TABS  (Login / Sign Up)
       ============================================================ */
    .stTabs {
        margin-top:  0 !important;
        padding: 0 !important;
        width: 600px;
        margin: auto;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 3px !important;
        background: rgba(0,0,0,0.28) !important;
        border-radius: 8px !important;
        padding: 3px !important;
        margin-top: -150px !important;
        margin-bottom: 1.2rem !important;
        width: fit-content !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        border-radius: 6px !important;
        color: #52525e !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.76rem !important;
        font-weight: 500 !important;
        padding: 6px 18px !important;
        transition: all 0.2s ease !important;
        border: none !important;
        white-space: nowrap !important;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(99,102,241,0.18) !important;
        color: #a78bfa !important;
        font-weight: 600 !important;
    }

    .stTabs [data-baseweb="tab-panel"] {
        padding: 0 !important;
        margin-top: 0 !important;
    }

    /* ============================================================
       INPUT FIELDS
       ============================================================ */
    .stTextInput label {
        font-size: 0.72rem !important;
        font-weight: 700 !important;
        color: #a1a1aa !important;
        text-transform: uppercase !important;
        letter-spacing: 0.07em !important;
        margin-bottom: 4px !important;
    }

    .stTextInput > div > div > input {
        background: rgba(0,0,0,0.3) !important;
        border: 1px solid rgba(99,102,241,0.15) !important;
        border-radius: 9px !important;
        color: #e2e2e8 !important;
        font-size: 0.88rem !important;
        font-family: 'Inter', sans-serif !important;
        padding: 0.7rem 0.85rem !important;
        transition: border-color 0.2s, box-shadow 0.2s !important;
    }

    .stTextInput > div > div > input::placeholder {
        color: #3a3a46 !important;
    }

    .stTextInput > div > div > input:focus {
        border-color: rgba(99,102,241,0.5) !important;
        box-shadow: 0 0 0 3px rgba(99,102,241,0.12) !important;
        outline: none !important;
    }

    

    /* ============================================================
       BUTTONS
       ============================================================ */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #7c3aed 100%) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 9px !important;
        padding: 0.72rem 1.2rem !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.86rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.04em !important;
        width: 100% !important;
        cursor: pointer !important;
        transition: transform 0.18s ease, box-shadow 0.18s ease !important;
        margin-top: 0.3rem !important;
        box-shadow: 0 4px 14px rgba(99,102,241,0.3) !important;
    }

    .stButton > button:hover {
        transform: translateY(-1.5px) !important;
        box-shadow: 0 6px 22px rgba(99,102,241,0.45) !important;
    }

    .stButton > button:active {
        transform: translateY(0px) !important;
    }

    /* ============================================================
       FILE UPLOADER (signup avatar)
       ============================================================ */
    .stFileUploader > div {
        background: rgba(0,0,0,0.2) !important;
        border: 1.5px dashed rgba(99,102,241,0.22) !important;
        border-radius: 9px !important;
        padding: 0.8rem !important;
    }

    .stFileUploader span {
        color: #52525e !important;
        font-size: 0.78rem !important;
    }

    /* ============================================================
       ALERTS / ERRORS inside auth
       ============================================================ */
    .stAlert {
        border-radius: 8px !important;
        font-size: 0.8rem !important;
        padding: 0.6rem 0.8rem !important;
        margin-top: 0.5rem !important;
    }

    /* ============================================================
       MAIN APP (after login) — keeps original dark feel
       ============================================================ */
    .hero {
        text-align: center;
        padding: 2.4rem 2rem 2rem;
        background: linear-gradient(135deg, rgba(99,102,241,0.06), rgba(168,85,247,0.06));
        border-radius: 20px;
        margin: 1.2rem 1rem 1.6rem;
        border: 1px solid rgba(99,102,241,0.12);
    }

    .hero-title {
        font-family: 'Syne', sans-serif;
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #a78bfa, #c084fc, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.3rem;
    }

    .hero-subtitle {
        font-size: 0.82rem;
        color: #52525e;
        font-weight: 400;
        letter-spacing: 0.06em;
    }

    /* Glass card — main app */
    .glass-card {
        background: rgba(20,20,28,0.55);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        border-radius: 18px;
        border: 1px solid rgba(99,102,241,0.1);
        padding: 1.6rem;
        margin: 1rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.28);
    }

    .section-title {
            margin-top: 100px;
        font-family: 'Syne', sans-serif;
        font-size: 0.95rem;
        font-weight: 700;
        color: #a1a1aa;
        letter-spacing: 0.02em;
        margin-bottom: 0.9rem;
    }

    /* Result card */
    .result-card {
        text-align: center;
        padding: 2.2rem 1.5rem;
        background: linear-gradient(135deg, rgba(99,102,241,0.08), rgba(168,85,247,0.08));
        border-radius: 20px;
        border: 1.5px solid rgba(99,102,241,0.18);
        margin: 1.2rem 0;
    }

    .result-emoji {
        font-size: 4.5rem;
        margin-bottom: 0.6rem;
        animation: float 3s ease-in-out infinite;
    }

    @keyframes float {
        0%, 100% { transform: translateY(0); }
        50%      { transform: translateY(-8px); }
    }

    .result-label {
        font-family: 'Syne', sans-serif;
        font-size: 2rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.2rem;
    }

    .result-conf {
        font-size: 0.95rem;
        color: #7c3aed;
        font-weight: 600;
    }

    /* Metric boxes */
    .metric-box {
        background: rgba(0,0,0,0.3);
        padding: 0.9rem 0.6rem;
        border-radius: 12px;
        text-align: center;
        border: 1px solid rgba(99,102,241,0.1);
    }

    .metric-label {
        font-size: 0.68rem;
        color: #52525e;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 500;
        margin-bottom: 0.2rem;
    }

    .metric-value {
        font-family: 'Syne', sans-serif;
        font-size: 1.35rem;
        font-weight: 700;
        background: linear-gradient(135deg, #a78bfa, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    /* Emotion bars */
    .emotion-bar { margin: 0.7rem 0; }

    .bar-header {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.25rem;
        font-size: 0.8rem;
        font-weight: 500;
    }

    .bar-track {
        height: 6px;
        background: rgba(255,255,255,0.06);
        border-radius: 3px;
        overflow: hidden;
    }

    .bar-fill {
        height: 100%;
        border-radius: 3px;
        transition: width 0.7s cubic-bezier(.4,0,.2,1);
    }

    /* Suggestion links */
    .suggestion-link {
        background: rgba(99,102,241,0.06);
        padding: 0.7rem 0.85rem;
        border-radius: 9px;
        margin: 0.45rem 0;
        border-left: 3px solid rgba(99,102,241,0.4);
        font-size: 0.82rem;
    }

    .suggestion-link a {
        color: #7c3aed !important;
        text-decoration: none;
        font-weight: 500;
    }

    .suggestion-link a:hover { text-decoration: underline; }

    /* Profile avatar */
    .profile-avatar {
        width: 52px;
        height: 52px;
        border-radius: 50%;
        object-fit: cover;
        border: 2px solid rgba(99,102,241,0.4);
        margin: 0 auto;
        display: block;
    }

    /* Recording pulse */
    .recording-status {
        text-align: center;
        padding: 1rem;
        background: rgba(239,68,68,0.08);
        border: 1.5px solid rgba(239,68,68,0.25);
        border-radius: 10px;
        margin: 0.8rem 0;
    }

    .rec-dot {
        display: inline-block;
        width: 10px;
        height: 10px;
        background: #ef4444;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 1.4s infinite;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50%      { opacity: 0.4; transform: scale(1.3); }
    }

    /* History items */
    .history-item {
        padding: 0.7rem 0.85rem;
        background: rgba(0,0,0,0.2);
        border-radius: 8px;
        margin: 0.4rem 0;
        font-size: 0.82rem;
    }



            
.stApp {
    position: relative;
    z-index: 1;
}

.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    z-index: -1;
    pointer-events: none;
}

.auth-wrapper,
.auth-box {
    position: relative;
    z-index: 20;
}

.stButton button,
.stTextInput input,
.stTabs button {
    position: relative;
    z-index: 30;
    pointer-events: auto;
    cursor: pointer;
}

</style>
""", unsafe_allow_html=True)


if 'user' not in st.session_state:
    st.session_state.user = None
if 'history' not in st.session_state:
    st.session_state.history = []
if 'voice_enabled' not in st.session_state:
    st.session_state.voice_enabled = True
if 'volume_level' not in st.session_state:
    st.session_state.volume_level = 1.0
if 'profile_pic' not in st.session_state:
    st.session_state.profile_pic = None
if 'user_name' not in st.session_state:
    st.session_state.user_name = 'User'

try:
    if not firebase_admin._apps:
        cred_path = "voice-emotion-analyzer-84149-firebase-adminsdk-fbsvc-a4b01a6e8b.json"
        if os.path.exists(cred_path):
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred)
            db = firestore.client()
    else:
         db = firestore.client()
except Exception as e:
    pass

config = {
    "apiKey": "AIzaSyD2tf_5r_-j9ZFyRuqVKWtP1R0gJMS5JS8",
    "authDomain": "voice-emotion-analyzer-84149.firebaseapp.com",
    "databaseURL": "",
    "storageBucket":"voice-emotion-analyzer-84149.firebasestorage.app",
}

firebase = pyrebase.initialize_app(config)
auth = firebase.auth()

    if db:
        try:
            db.collection("users").document(uid).collection("history").add(data)
        except:
            pass

def get_history(uid):
    if db:
        try:
            docs = db.collection("users").document(uid).collection("history").order_by("Timestamp", direction=firestore.Query.DESCENDING).stream()
            return [doc.to_dict() for doc in docs]
        except:
            return []
    return []

def firebase_login(email, password):
    try:
        user = auth.sign_in_with_email_and_password(email, password)
        return user
    except:
        return None

def firebase_signup(email, password, name):
    try:
        user = auth.create_user_with_email_and_password(email, password)
        uid = user['localId']
        if db:
            db.collection("users").document(uid).set({
                "name": name,
                "email": email,
                "created_at": datetime.now()
            })
        return user
    except Exception as e:
        st.error(f"Signup failed: {str(e)}")
        return None


def speak_text(text, volume_level=1.0, rate=140):
    if not st.session_state.voice_enabled:
        return
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', rate)
        volume = min(1.0, volume_level * 1.3)
        engine.setProperty('volume', volume)
        engine.say(text)
        engine.runAndWait()
    except:
        pass


def load_model():
    try:
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        model = tf.keras.models.load_model('CNN_model_IMPROVED_FINAL.h5')
        scaler = pickle.load(open('scaler_final.pkl', 'rb'))
        encoder = pickle.load(open('encoder_final.pkl', 'rb'))
        return model, scaler, encoder
    except:
        return None, None, None


def detect_gender(audio, sr):
    """
    FIXED: Balanced gender detection - NO BIAS toward female
    Uses REAL acoustic science thresholds
    """
    try:
        # === LAYER 1: Fundamental Frequency (F0) ===
        f0, voiced_flag, voiced_prob = librosa.pyin(
            audio,
            fmin=50,
            fmax=400,
            sr=sr,
            frame_length=2048
        )
        
        voiced_f0 = f0[voiced_flag & (voiced_prob > 0.5)]
        
        if len(voiced_f0) < 5:
            return _gender_fallback(audio, sr)
        
        median_f0 = float(np.median(voiced_f0))
        mean_f0 = float(np.mean(voiced_f0))
        std_f0 = float(np.std(voiced_f0))
        
        # === LAYER 2: Formants (F1, F2, F3) ===
        f1, f2, f3 = _estimate_formants(audio, sr)
        
        # === LAYER 3: MFCC (Most important for gender!) ===
        mfcc_mean = _estimate_mfcc(audio, sr)
        
        # === LAYER 4: Spectral Features ===
        spectral_centroid = _estimate_spectral_centroid(audio, sr)
        spectral_tilt = _estimate_spectral_tilt(audio, sr)
        
        
        score = 0.0
        
       
        pitch_score = 0.0
        
        if median_f0 < 85:
            pitch_score = -3.0  # STRONG MALE
        elif median_f0 < 110:
            pitch_score = -2.2  # MALE
        elif median_f0 < 135:
            pitch_score = -0.8  # SLIGHTLY MALE
        elif median_f0 < 155:
            pitch_score = 0.0   # NEUTRAL
        elif median_f0 < 180:
            pitch_score = 0.8   # SLIGHTLY FEMALE
        elif median_f0 < 210:
            pitch_score = 2.2   # FEMALE
        else:
            pitch_score = 3.0   # STRONG FEMALE
        
        score += pitch_score * 3.0
        
        # FORMANT SCORE (Weight: 3.0)
        # F1: Male 500-600Hz, Female 650-750Hz
        # F2: Male 1400-1600Hz, Female 1700-2000Hz
        formant_score = 0.0
        formant_count = 0
        
        if f1 > 0:
            if f1 < 480:
                formant_score += -1.5  # MALE
            elif f1 < 550:
                formant_score += -0.5  # SLIGHTLY MALE
            elif f1 < 650:
                formant_score += 0.0   # NEUTRAL
            elif f1 < 720:
                formant_score += 1.2   # FEMALE
            else:
                formant_score += 1.8   # STRONG FEMALE
            formant_count += 1
        
        if f2 > 0:
            if f2 < 1300:
                formant_score += -1.5  # MALE
            elif f2 < 1500:
                formant_score += -0.8  # SLIGHTLY MALE
            elif f2 < 1650:
                formant_score += 0.0   # NEUTRAL
            elif f2 < 1850:
                formant_score += 1.0   # FEMALE
            else:
                formant_score += 1.8   # STRONG FEMALE
            formant_count += 1
        
        if f3 > 0:
            if f3 < 2300:
                formant_score += -0.8
            elif f3 < 2700:
                formant_score += 0.0
            else:
                formant_score += 0.8
            formant_count += 1
        
        if formant_count > 0:
            formant_score /= formant_count
            score += formant_score * 3.0
        
        # MFCC SCORE (Weight: 3.5 - MOST IMPORTANT!)
        mfcc_score = 0.0
        if mfcc_mean < 10:
            mfcc_score = -2.5  # STRONG MALE
        elif mfcc_mean < 12:
            mfcc_score = -1.5  # MALE
        elif mfcc_mean < 13.5:
            mfcc_score = -0.5  # SLIGHTLY MALE
        elif mfcc_mean < 15:
            mfcc_score = 0.0   # NEUTRAL
        elif mfcc_mean < 16.5:
            mfcc_score = 1.2   # SLIGHTLY FEMALE
        elif mfcc_mean < 18:
            mfcc_score = 2.0   # FEMALE
        else:
            mfcc_score = 2.8   # STRONG FEMALE
        
        score += mfcc_score * 3.5
        
        # SPECTRAL CENTROID (Weight: 1.5)
        # Male: Lower centroid (2000-3000Hz)
        # Female: Higher centroid (3500-4500Hz)
        centroid_score = 0.0
        if spectral_centroid < 2500:
            centroid_score = -1.2  # MALE
        elif spectral_centroid < 3000:
            centroid_score = -0.3
        elif spectral_centroid < 3500:
            centroid_score = 0.0
        elif spectral_centroid < 4000:
            centroid_score = 0.8
        else:
            centroid_score = 1.2   # FEMALE
        
        score += centroid_score * 1.5
        
        # SPECTRAL TILT (Weight: 1.0)
        # Male: Steep tilt (negative slope, deeper voice)
        # Female: Gentle tilt
        tilt_score = 0.0
        if spectral_tilt < -8:
            tilt_score = -1.0  # MALE
        elif spectral_tilt < -6:
            tilt_score = -0.3
        elif spectral_tilt < -4:
            tilt_score = 0.0
        else:
            tilt_score = 0.8   # FEMALE
        
        score += tilt_score * 1.0
        
    
        max_possible_score = (3.0 * 3.0) + (1.8 * 3.0) + (2.8 * 3.5) + (1.2 * 1.5) + (0.8 * 1.0)
        min_possible_score = (-3.0 * 3.0) + (-1.5 * 3.0) + (-2.5 * 3.5) + (-1.2 * 1.5) + (-1.0 * 1.0)
        
        # Normalize score to -1 to +1 range
        if score < 0:
            normalized = score / abs(min_possible_score)
        else:
            normalized = score / max_possible_score
        
        normalized = np.clip(normalized, -1.0, 1.0)
        
        # DECISION THRESHOLD (at 0, not skewed)
        if normalized < -0.15:
            gender = 'male'
        elif normalized > 0.15:
            gender = 'female'
        else:
            gender = 'uncertain'
        
        # CONFIDENCE (0.50 to 0.95)
        confidence = 0.50 + (abs(normalized) * 0.45)
        confidence = np.clip(confidence, 0.50, 0.95)
        
        return gender, round(float(confidence), 2)
    
    except Exception as e:
        print(f"[ERROR] detect_gender: {str(e)}")
        return _gender_fallback(audio, sr)


def _estimate_mfcc(audio, sr):
    """
    Extract MFCC mean - STRONGEST indicator of gender
    Male: Lower MFCC values (deeper spectral shape)
    Female: Higher MFCC values
    """
    try:
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfcc_mean = float(np.mean(mfcc[0, :]))  # Use first MFCC coefficient
        return mfcc_mean
    except Exception:
        return 14.0  # Neutral default


def _estimate_formants(audio, sr):
    """
    Estimate first 3 formants using LPC
    """
    try:
        # Pre-emphasis to enhance formants
        audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])
        
        frame_length = 512
        hop = 256
        
        # Use librosa to split into frames
        frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop).T
        
        if len(frames) == 0:
            return 0, 0, 0
        
        # Filter by voiced frames (higher energy)
        rms_vals = np.array([np.sqrt(np.mean(f ** 2)) for f in frames])
        threshold = np.percentile(rms_vals, 60)
        voiced_frames = frames[rms_vals >= threshold]
        
        if len(voiced_frames) == 0:
            return 0, 0, 0
        
        all_f1, all_f2, all_f3 = [], [], []
        
        for frame in voiced_frames[:min(50, len(voiced_frames))]:  # Limit frames for speed
            try:
                # Apply Hamming window
                windowed_frame = frame * np.hamming(len(frame))
                
                # LPC analysis
                lpc_coeffs = librosa.core.lpc(windowed_frame, order=14)
                
                # Find roots of LPC polynomial
                roots = np.roots(lpc_coeffs)
                roots = roots[np.abs(roots) <= 1.0]
                
                if len(roots) == 0:
                    continue
                
                # Convert to frequencies
                angles = np.angle(roots)
                freqs = np.abs(angles) * sr / (2.0 * np.pi)
                freqs = np.sort(freqs[freqs > 0])
                
                # Extract formants (F1, F2, F3)
                f1_cands = freqs[(freqs >= 200) & (freqs <= 900)]
                f2_cands = freqs[(freqs >= 700) & (freqs <= 3000)]
                f3_cands = freqs[(freqs >= 1800) & (freqs <= 4500)]
                
                if len(f1_cands) > 0:
                    all_f1.append(f1_cands[0])
                if len(f2_cands) > 0:
                    all_f2.append(f2_cands[0])
                if len(f3_cands) > 0:
                    all_f3.append(f3_cands[0])
            
            except Exception:
                continue
        
        f1 = float(np.median(all_f1)) if len(all_f1) > 2 else 0
        f2 = float(np.median(all_f2)) if len(all_f2) > 2 else 0
        f3 = float(np.median(all_f3)) if len(all_f3) > 2 else 0
        
        return f1, f2, f3
    
    except Exception:
        return 0, 0, 0


def _estimate_spectral_centroid(audio, sr):
    """
    Spectral centroid - higher for female, lower for male
    """
    try:
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        return float(np.mean(centroid))
    except Exception:
        return 3000.0


def _estimate_spectral_tilt(audio, sr):
    """
    Spectral slope - steeper (more negative) for male
    """
    try:
        S = np.abs(librosa.stft(audio)) ** 2
        freqs = librosa.fft_frequencies(sr=sr, n_fft=(S.shape[0] - 1) * 2)
        avg_spectrum = np.mean(S, axis=1)
        avg_db = 10 * np.log10(avg_spectrum + 1e-10)
        
        # Focus on speech range
        mask = (freqs >= 200) & (freqs <= 4000)
        freqs_masked = freqs[mask]
        db_masked = avg_db[mask]
        
        if len(freqs_masked) < 10:
            return -6.0
        
        log_freqs = np.log2(freqs_masked)
        slope, _ = np.polyfit(log_freqs, db_masked, 1)
        
        return float(slope)
    
    except Exception:
        return -6.0


def _gender_fallback(audio, sr):
    """
    Simple fallback using spectral centroid
    """
    try:
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)))
        if centroid < 2800:
            return 'male', 0.60
        elif centroid > 3500:
            return 'female', 0.60
        else:
            return 'uncertain', 0.55
    except Exception:
        return 'uncertain', 0.50

def add_noise(data, factor=0.005):
    return (data + factor * np.random.randn(len(data))).astype(np.float32)

def pitch_shift(data, sr, steps):
    return librosa.effects.pitch_shift(y=data, sr=sr, n_steps=steps)


def process_live(audio, sr):
  
    # 1. Remove DC
    audio = audio - np.mean(audio)

    # 2. High-pass filter at 80Hz
    try:
        b, a = butter(4, 80 / (sr / 2), btype='high')
        audio = filtfilt(b, a, audio)
    except:
        pass

    # 3. Soft noise gate — reduces noise without wiping signal
    # Uses 15th percentile as threshold, applies smooth reduction
    threshold = np.percentile(np.abs(audio), 15)
    # Soft ratio: anything below threshold gets scaled down, not zeroed
    soft_mask = np.where(
        np.abs(audio) > threshold,
        np.ones_like(audio),
        np.abs(audio) / (threshold + 1e-8)  # smooth fade to 0
    )
    audio = audio * soft_mask

    # 4. Find speech segments
    intervals = librosa.effects.split(audio, top_db=22)
    if len(intervals) > 0:
        speech = np.concatenate([audio[s:e] for s, e in intervals])
        if len(speech) > sr * 0.3:
            audio = speech

    # 5. Take loudest 2.5s segment
    target = int(DURATION * sr)

    if len(audio) > target:
        hop = 512
        rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=hop)[0]
        win = target // hop
        if len(rms) > win:
            best_i = max(range(len(rms) - win), key=lambda i: np.sum(rms[i:i+win]), default=0)
            audio = audio[best_i * hop:(best_i * hop) + target]
        else:
            audio = audio[:target]

    # 6. Pad if shorter
    if len(audio) < target:
        audio = np.pad(audio, (0, target - len(audio)))

    # 7. Gentle pre-emphasis (0.85 not 0.97)
    # 0.97 is too aggressive for live mic — boosts noise too much
    # 0.85 gives a slight high-freq lift without amplifying noise
    pre_emphasized = np.append(audio[0], audio[1:] - 0.85 * audio[:-1])
    audio = pre_emphasized

    # 8. RMS normalize to training level
    rms = np.sqrt(np.mean(audio ** 2))
    if rms > 1e-5:
        target_rms = 0.06
        scale = min(target_rms / rms, 25.0)
        audio = np.clip(audio * scale, -1.0, 1.0)

    return audio.astype(np.float32)   


def extract_features(audio, sr):
    f = []
    f.extend(np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0))
    f.extend(np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0))
    f.extend(np.mean(librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128).T, axis=0))
    f.extend(np.mean(librosa.feature.spectral_contrast(y=audio, sr=sr).T, axis=0))
    f.extend(np.mean(librosa.feature.tonnetz(y=audio, sr=sr).T, axis=0))
    f.append(np.mean(librosa.feature.zero_crossing_rate(y=audio)))
    f.append(np.mean(librosa.feature.rms(y=audio)))
    f.append(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)))
    f.append(np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr)))
    return np.array(f)


def predict_ensemble(audio, sr, model, scaler, encoder):
    """
    EXACT ENSEMBLE METHOD:
    - Original
    - Noise (0.005)
    - Noise (0.003)
    - Pitch +1
    - Pitch -1
    - Noise + Pitch
    """
    versions = [audio, add_noise(audio, 0.005), add_noise(audio, 0.003)]
    
    try:
        versions.append(pitch_shift(audio, sr, 1))
        versions.append(pitch_shift(audio, sr, -1))
        versions.append(pitch_shift(add_noise(audio, 0.003), sr, 1))
    except:
        pass
    
    preds = []
    for v in versions:
        try:
            f = extract_features(v, sr)
            if len(f) == 197:
                f_scaled = scaler.transform([f])
                p = model.predict(f_scaled, verbose=0)
                if len(p.shape) == 3:
                    p = p[0]
                preds.append(p[0] if len(p.shape) == 2 else p)
        except:
            continue
    
    if not preds:
        return None, None
    
    avg = np.mean(preds, axis=0)
    labels = encoder.classes_
    result = {labels[i]: float(avg[i]) for i in range(len(labels))}
    return max(result, key=result.get), result


def predict_file(path, model, scaler, encoder):
    try:
        audio, sr = librosa.load(path, duration=DURATION, offset=OFFSET, sr=SR)
        f = extract_features(audio, sr)
        if len(f) != 197:
            return None, None, None
        
        f_scaled = scaler.transform([f])
        p = model.predict(f_scaled, verbose=0)
        if len(p.shape) == 3:
            p = p[0]
        p = p[0] if len(p.shape) == 2 else p
        
        labels = encoder.classes_
        result = {labels[i]: float(p[i]) for i in range(len(labels))}
        return max(result, key=result.get), result, audio
    except:
        return None, None, None


def plot_spectrogram(audio, sr):
    fig, ax = plt.subplots(figsize=(10, 3))
    fig.patch.set_facecolor('#0a0a0f')
    ax.set_facecolor('#0a0a0f')
    mel = librosa.power_to_db(librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128), ref=np.max)
    librosa.display.specshow(mel, sr=sr, x_axis='time', y_axis='mel', ax=ax, cmap='magma')
    ax.tick_params(colors='#6b7280', labelsize=8)
    ax.set_xlabel('Time (s)', color='#9ca3af', fontsize=9)
    ax.set_ylabel('Frequency (Hz)', color='#9ca3af', fontsize=9)
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png', facecolor='#0a0a0f', dpi=120)
    buf.seek(0)
    plt.close()
    return buf

def plot_waveform(audio, sr):
    fig, ax = plt.subplots(figsize=(10, 3))
    fig.patch.set_facecolor('#0a0a0f')
    ax.set_facecolor('#0a0a0f')
    librosa.display.waveshow(audio, sr=sr, ax=ax, color='#6366f1', alpha=0.8)
    ax.tick_params(colors='#6b7280', labelsize=8)
    ax.set_xlabel('Time (s)', color='#9ca3af', fontsize=9)
    ax.set_ylabel('Amplitude', color='#9ca3af', fontsize=9)
    for spine in ax.spines.values():
        spine.set_color('#27272a')
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png', facecolor='#0a0a0f', dpi=120)
    buf.seek(0)
    plt.close()
    return buf


def show_emotion_result(emotion, scores, audio, sr, user_name, uid):
    gender, gender_conf = detect_gender(audio, sr)
    cfg = EMOTIONS[emotion]
    suggestions = EMOTION_SUGGESTIONS[emotion]
    
    st.markdown(f"""
        <div class="result-card">
            <div class="result-emoji">{cfg['emoji']}</div>
            <div class="result-label" style="color: {cfg['color']};">{emotion.upper()}</div>
            <div class="result-conf">{scores[emotion]:.1%} CONFIDENCE</div>
            <div style="margin-top: 1rem; font-size: 1.1rem; font-weight: 700; color: #a5b4fc;">
{GENDERS[gender]['emoji']} {GENDERS[gender]['label']} ({gender_conf:.1%} confidence)
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    speak_text(f"Emotion detected: {emotion}. Confidence: {scores[emotion]:.0%}. Gender: {gender}.", st.session_state.volume_level)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-box"><div class="metric-label">Emotion</div><div class="metric-value">{emotion.upper()}</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-box"><div class="metric-label">Confidence</div><div class="metric-value">{scores[emotion]:.0%}</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-box"><div class="metric-label">Gender</div><div class="metric-value">{gender.upper()}</div></div>', unsafe_allow_html=True)
    with col4:
        rms = np.sqrt(np.mean(audio ** 2))
        st.markdown(f'<div class="metric-box"><div class="metric-label">Audio RMS</div><div class="metric-value">{rms:.4f}</div></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-title">💬 Personalized Message</div>', unsafe_allow_html=True)
    message = np.random.choice(suggestions['messages'])
    if emotion == 'sad':
        joke = np.random.choice(JOKES)
        st.info(f"💙 {message}\n\n😄 **Funny Moment:** {joke}")
        speak_text(f"{message} Here's a joke: {joke}", st.session_state.volume_level)
    else:
        st.info(f"✨ {message}")
        speak_text(message, st.session_state.volume_level)
    
    st.markdown('<div class="section-title">🎯 Suggestions For You</div>', unsafe_allow_html=True)
    for suggestion in suggestions['suggestions']:
        if 'http' in suggestion:
            emoji, rest = suggestion.split(' ', 1)
            text, link = rest.rsplit(' - ', 1)
            st.markdown(f'<div class="suggestion-link"><strong>{emoji} {text}</strong><br><a href="{link}" target="_blank" style="color: #6366f1; text-decoration: none;">👉 Click here</a></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="suggestion-link">✨ {suggestion}</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-title">📊 All Emotions</div>', unsafe_allow_html=True)
    for e in sorted(scores, key=scores.get, reverse=True):
        p = scores[e]
        c = EMOTIONS[e]['color']
        em = EMOTIONS[e]['emoji']
        st.markdown(f"""
            <div class="emotion-bar">
                <div class="bar-header">
                    <span>{em} {e.upper()}</span>
                    <span>{p:.1%}</span>
                </div>
                <div class="bar-track">
                    <div class="bar-fill" style="width: {p*100:.0f}%; background: {c};"></div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-title">🎵 Audio Visualizations</div>', unsafe_allow_html=True)
    tab1, tab2 = st.tabs(['Waveform', 'Spectrogram'])
    with tab1:
        st.image(plot_waveform(audio, sr), use_container_width=True)
    with tab2:
        st.image(plot_spectrogram(audio, sr), use_container_width=True)
    
    # Audio playback
    st.markdown('<div class="section-title">🔊 Audio Playback</div>', unsafe_allow_html=True)
    audio_bytes = BytesIO()
    sf.write(audio_bytes, audio, sr, format='WAV')
    audio_bytes.seek(0)
    st.audio(audio_bytes, format='audio/wav')
    
    # Save to history
    st.session_state.history.insert(0, {
        'emotion': emotion,
        'gender': gender,
        'conf': scores[emotion],
        'time': datetime.now().strftime('%H:%M:%S')
    })
    st.session_state.history = st.session_state.history[:20]
    
    save_history(uid, {
        'Timestamp': datetime.now(pytz.utc).strftime('%Y-%m-%d %H:%M:%S UTC'),
        'Emotion': emotion,
        'Confidence': float(scores[emotion]),
        'Gender': gender,
        'Gender_Confidence': float(gender_conf),
        'Source': 'VibeCheck Analysis'
    })


def record_audio(duration=6):
    try:
        audio = sd.rec(int(duration * SR), samplerate=SR, channels=1, dtype='float32')
        sd.wait()
        return audio.flatten()
    except:
        return None


def auth_ui():
    # Wrap everything in the centered auth-wrapper div
    st.markdown("""
        <div class="auth-wrapper">
            <div class="auth-logo">
                <!-- SVG LOGO: a stylised speech-bubble with a subtle wave inside -->
                <svg viewBox="0 0 56 56" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <defs>
                        <linearGradient id="g1" x1="0" y1="0" x2="56" y2="56">
                            <stop offset="0%" stop-color="#6366f1"/>
                            <stop offset="50%" stop-color="#a855f7"/>
                            <stop offset="100%" stop-color="#ec4899"/>
                        </linearGradient>
                        <linearGradient id="g2" x1="0" y1="0" x2="56" y2="0">
                            <stop offset="0%" stop-color="#a78bfa" stop-opacity="0.9"/>
                            <stop offset="100%" stop-color="#f472b6" stop-opacity="0.9"/>
                        </linearGradient>
                    </defs>
                    <!-- Bubble body -->
                    <path d="M6 8 C6 4.686 8.686 2 12 2 L44 2 C47.314 2 50 4.686 50 8 L50 36
                             C50 39.314 47.314 42 44 42 L18 42 L10 50 L10 42 L12 42
                             C8.686 42 6 39.314 6 36 Z"
                          fill="url(#g1)" fill-opacity="0.18" stroke="url(#g1)" stroke-width="1.8"/>
                    <!-- Wave line inside bubble -->
                    <path d="M14 24 Q18 17 22 24 T30 24 T38 24 T42 24"
                          fill="none" stroke="url(#g2)" stroke-width="2.2" stroke-linecap="round"/>
                </svg>
            </div>
            <div class="auth-title">VibeCheck AI</div>
            <div class="auth-subtitle">Voice · Emotion · Identity</div>
            <div class="auth-box">
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(['Login', 'Sign Up'])

    with tab1:
        email = st.text_input('Email', key='login_email', placeholder='you@email.com')
        password = st.text_input('Password', type='password', key='login_pass', placeholder='••••••••')

        if st.button('Sign In', use_container_width=True, type='primary'):
            if email and password:
                user = firebase_login(email, password)
                if user:
                    st.session_state.user = user
                    if db:
                        try:
                            user_doc = db.collection("users").document(user['localId']).get()
                            if user_doc.exists:
                                st.session_state.user_name = user_doc.to_dict().get('name', 'User')
                            else:
                                st.session_state.user_name = 'User'
                        except:
                            st.session_state.user_name = 'User'
                    else:
                        st.session_state.user_name = 'User'
                    speak_text(f"Welcome back, {st.session_state.user_name}!", st.session_state.volume_level)
                    st.rerun()
                else:
                    st.error('Invalid email or password.')
            else:
                st.error('Please fill both fields.')

    with tab2:
        name = st.text_input('Name', key='signup_name', placeholder='Your name')

        st.markdown("<p style='font-size:0.7rem; color:#3a3a46; margin:0.6rem 0 0.3rem; letter-spacing:0.06em; text-transform:uppercase;'>Avatar (optional)</p>", unsafe_allow_html=True)
        prof_file = st.file_uploader("Upload", type=["jpg","png","jpeg"], label_visibility="collapsed")
        if prof_file:
            encoded = base64.b64encode(prof_file.read()).decode()
            st.session_state.profile_pic = f"data:image/png;base64,{encoded}"
            st.success("Avatar saved ✓")

        email2   = st.text_input('Email', key='signup_email', placeholder='you@email.com')
        password2= st.text_input('Password', type='password', key='signup_pass', placeholder='Min 6 characters')
        confirm2 = st.text_input('Confirm', type='password', key='signup_confirm', placeholder='Re-enter password')

        if st.button('Create Account', use_container_width=True, type='primary'):
            if name and email2 and password2 and confirm2:
                if password2 != confirm2:
                    st.error('Passwords do not match.')
                elif len(password2) < 6:
                    st.error('Password needs at least 6 characters.')
                else:
                    user = firebase_signup(email2, password2, name)
                    if user:
                        st.session_state.user = user
                        st.session_state.user_name = name
                        speak_text(f"Welcome, {name}!", st.session_state.volume_level)
                        st.rerun()
            else:
                st.error('Please fill all fields.')

    # Close the card + wrapper
    st.markdown('</div></div>', unsafe_allow_html=True)



def main_app():
    model, scaler, encoder = load_model()
    
    if model is None:
        st.error(' Model files not found!')
        st.stop()
    
    st.markdown('<div class="hero"><div class="hero-title">🎭 VibeCheck AI</div><div class="hero-subtitle">Advanced Voice Emotion Recognition System</div></div>', unsafe_allow_html=True)
    
    # Header with profile
    avatar_url = st.session_state.get('profile_pic', "https://www.gravatar.com/avatar/00000000000000000000000000000000?d=mp&f=y")
    user_name = st.session_state.get('user_name', 'User')
    
    col_h1, col_h2, col_h3 = st.columns([1, 6, 1])
    with col_h1:
        st.markdown(f'<img src="{avatar_url}" class="profile-avatar">', unsafe_allow_html=True)
    with col_h2:
        st.markdown(f'<p style="font-size: 1.3rem; font-weight: 700; color: #a855f7; margin: 1.5rem 0;">Welcome, {user_name}! 👋</p>', unsafe_allow_html=True)
    with col_h3:
        if st.button('🚪 Logout', use_container_width=True):
            st.session_state.user = None
            st.session_state.history = []
            st.session_state.user_name = 'User'
            st.session_state.profile_pic = None
            speak_text('Goodbye!', st.session_state.volume_level)
            st.rerun()
    
    # Voice toggle
    st.session_state.voice_enabled = st.checkbox('🔊 Voice Feedback', value=st.session_state.voice_enabled)
    
    # Main controls
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎤 Record Audio")
        if st.button('🔴 Start Recording ', use_container_width=True, type='primary'):
            st.markdown("""
                <div class="recording-status">
                    <span class="rec-dot"></span>
                    <span style="color: #f87171; font-weight: 600;">RECORDING... Speak now!</span>
                </div>
            """, unsafe_allow_html=True)
            
            audio = record_audio(6)   
            if audio is not None and len(audio) > 0:
                # Process with EXACT logic from best file
                processed = process_live(audio, SR)
                
                # Predict with EXACT ensemble from best file
                emo, scores = predict_ensemble(processed, SR, model, scaler, encoder)
                
                if emo and scores:
                    show_emotion_result(emo, scores, processed, SR, user_name, st.session_state.user['localId'])
                else:
                    st.warning('Could not analyze. Please speak louder and try again.')
            else:
                st.error('Recording failed. Check microphone.')
    
    with col2:
        st.markdown("### 📁 Upload Audio File")
        uploaded = st.file_uploader('Select audio file', type=['wav', 'mp3', 'ogg', 'flac', 'm4a'], label_visibility='collapsed')
        if uploaded:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]) as tmp:
                tmp.write(uploaded.getbuffer())
                tmp_path = tmp.name
            
            # Use EXACT file prediction from best file
            emo, scores, audio = predict_file(tmp_path, model, scaler, encoder)
            os.unlink(tmp_path)
            
            if emo and scores:
                show_emotion_result(emo, scores, audio, SR, user_name, st.session_state.user['localId'])
            else:
                st.error('Could not process file.')
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # History
    if st.session_state.history:
        st.markdown('<div class="section-title">📜 Recent Detections</div>', unsafe_allow_html=True)
        for h in st.session_state.history[:5]:
            emoji = EMOTIONS[h['emotion']]['emoji']
            color = EMOTIONS[h['emotion']]['color']
            gender_emoji = GENDERS[h['gender']]['emoji']
            st.markdown(f"""
                <div style="padding: 0.8rem; background: rgba(0,0,0,0.2); border-radius: 8px; margin: 0.5rem 0; border-left: 3px solid {color};">
                    <span style="font-size: 1.2rem;">{emoji}</span>
                    <span style="color: #e5e5e5; margin-left: 0.5rem; font-weight: 600;">{h['emotion'].capitalize()}</span>
                    <span style="margin-left: 0.5rem;">{gender_emoji} {h['gender'].capitalize()}</span>
                    <div style="color: #9ca3af; font-size: 0.75rem; margin-top: 0.2rem;">{h['time']} • {h['conf']:.0%}</div>
                </div>
            """, unsafe_allow_html=True)


if st.session_state.user is None:
    auth_ui()
else:
    main_app()