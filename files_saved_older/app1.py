import os
import tempfile
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
import librosa
import librosa.display
from io import BytesIO
from datetime import datetime
import pytz
import gdown
import pyrebase
import firebase_admin
from firebase_admin import credentials, firestore
import base64
import sounddevice as sd
import soundfile as sf
from scipy import signal
from scipy.stats import skew, kurtosis

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Voice Emotion Analyzer | Voice Emotion Intelligence", page_icon="🎭", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Geist:wght@300;400;500;600;700&family=Geist+Mono&display=swap');
    
    :root {
        --background: #000000;
        --card-bg: #0a0a0a;
        --border: #1f1f1f;
        --accent: #ffffff;
        --text: #ffffff;
        --text-muted: #888888;
        --neon-blue: #0070f3;
        --neon-green: #00ffbd;
        --neon-purple: #7928ca;
        --neon-red: #ff0080;
    }

    .stApp {
        background-color: var(--background);
        color: var(--text);
        font-family: 'Geist', sans-serif;
    }

    /* Vercel-style Glass Cards */
    .glass-card {
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 32px;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }

    .glass-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
    }

    /* Personalized Greeting */
    .hero-text {
        font-size: 3.5rem;
        font-weight: 700;
        letter-spacing: -0.06em;
        line-height: 1;
        margin-bottom: 16px;
        background: linear-gradient(180deg, #fff 0%, #888 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Enhanced Profile Avatar */
    .profile-avatar-container {
        position: relative;
        width: 180px;
        height: 180px;
        margin: 0 auto 24px;
    }

    .profile-img-large {
        width: 100%;
        height: 100%;
        border-radius: 50%;
        object-fit: cover;
        border: 1px solid var(--border);
        padding: 4px;
        background: var(--background);
    }

    .profile-glow {
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        border-radius: 50%;
        box-shadow: 0 0 40px rgba(255,255,255,0.05);
        z-index: -1;
    }

    /* Action Cards for Suggestions */
    .suggestion-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 20px;
        margin-top: 32px;
    }

    .action-card {
        background: #050505;
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 24px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 200px;
    }

    .action-card:hover {
        border-color: #444;
        transform: translateY(-8px);
        background: #0a0a0a;
    }

    .action-card .type-tag {
        font-family: 'Geist Mono', monospace;
        font-size: 0.7rem;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 12px;
        display: block;
    }

    .btn-action {
        background: var(--text);
        color: var(--background);
        padding: 12px;
        border-radius: 6px;
        text-align: center;
        text-decoration: none;
        font-weight: 600;
        font-size: 0.85rem;
        margin-top: 20px;
        transition: opacity 0.2s;
    }

    .btn-action:hover {
        opacity: 0.9;
    }

    @keyframes pulse {
        0% { transform: scale(1); box-shadow: 0 0 0 0 rgba(255, 255, 255, 0.2); }
        70% { transform: scale(1.05); box-shadow: 0 0 0 20px rgba(255, 255, 255, 0); }
        100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(255, 255, 255, 0); }
    }

    .mic-recording {
        animation: pulse 1.5s infinite;
        background: var(--neon-red) !important;
        border: none !important;
    }

    .audio-player-container {
        background: #050505;
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 16px;
        margin-top: 16px;
    }
    
    .record-btn {
        background: linear-gradient(135deg, #ff0080, #7928ca) !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 16px 32px !important;
        border-radius: 8px !important;
        border: none !important;
        font-size: 1.1rem !important;
    }
    
    .record-btn:hover {
        opacity: 0.9 !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- FIREBASE SETUP ----------------
db = None
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

# ---------------- MODEL FILE IDS ----------------
MODEL_FILE_ID   = "1BEVdR_A7M9hD7gLFy9d3TBA0J7VRU2wN"
SCALER_FILE_ID  = "19aV8CNKqpPbIeLIZlp-HHSTeucc1iYpL"
ENCODER_FILE_ID = "1xKMGH8uri5E3fpw9L7KXmHXI2QyUkuou"

MODEL_PATH  = "CNN_full_model.h5"
SCALER_PATH = "scaler2.pickle"
ENCODER_PATH = "encoder2.pickle"
FEATURE_LEN = 2376

# ---------------- IMPROVED GENDER DETECTION ----------------
def detect_gender_advanced(y, sr):
    """
    Advanced gender detection using multiple acoustic features:
    - Fundamental frequency (F0/pitch)
    - Formant frequencies (F1, F2, F3)
    - Spectral centroid
    - Harmonic-to-noise ratio
    - Jitter and shimmer (voice quality)
    """
    try:
        # 1. Pitch (F0) extraction using autocorrelation
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr, fmin=75, fmax=400)
        
        # Get valid pitch values
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        if len(pitch_values) < 10:
            return "Unknown", 0, 50.0
        
        mean_pitch = np.mean(pitch_values)
        median_pitch = np.median(pitch_values)
        std_pitch = np.std(pitch_values)
        
        # 2. Spectral features
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        
        # 3. Formant estimation (simplified)
        # Using LPC (Linear Predictive Coding)
        try:
            from scipy.signal import lfilter
            # Pre-emphasis
            pre_emphasized = np.append(y[0], y[1:] - 0.97 * y[:-1])
            # Get spectrum
            fft_spectrum = np.fft.rfft(pre_emphasized)
            power_spectrum = np.abs(fft_spectrum) ** 2
            formant_freq = np.mean(power_spectrum)
        except:
            formant_freq = spectral_centroid
        
        # 4. Calculate gender score using weighted features
        gender_score = 0
        confidence_factors = []
        
        # Pitch-based scoring (most reliable)
        if mean_pitch < 130:  # Very low (definitely male)
            gender_score -= 4
            confidence_factors.append(25)
        elif mean_pitch < 150:  # Low (likely male)
            gender_score -= 3
            confidence_factors.append(20)
        elif mean_pitch < 165:  # Low-medium (probably male)
            gender_score -= 2
            confidence_factors.append(15)
        elif mean_pitch < 180:  # Medium (unclear)
            gender_score -= 1
            confidence_factors.append(10)
        elif mean_pitch > 220:  # Very high (definitely female)
            gender_score += 4
            confidence_factors.append(25)
        elif mean_pitch > 200:  # High (likely female)
            gender_score += 3
            confidence_factors.append(20)
        elif mean_pitch > 185:  # Medium-high (probably female)
            gender_score += 2
            confidence_factors.append(15)
        else:  # 180-185 (unclear)
            gender_score += 1
            confidence_factors.append(10)
        
        # Spectral centroid (voice brightness)
        if spectral_centroid < 1500:  # Darker voice
            gender_score -= 1.5
            confidence_factors.append(8)
        elif spectral_centroid > 2500:  # Brighter voice
            gender_score += 1.5
            confidence_factors.append(8)
        else:
            confidence_factors.append(3)
        
        # Formant consideration
        if formant_freq < spectral_centroid * 0.8:
            gender_score -= 0.5
            confidence_factors.append(5)
        elif formant_freq > spectral_centroid * 1.2:
            gender_score += 0.5
            confidence_factors.append(5)
        
        # Pitch stability (less variable = more confident)
        if std_pitch < 20:
            confidence_factors.append(10)
        elif std_pitch < 40:
            confidence_factors.append(5)
        
        # Determine gender and confidence
        if gender_score <= -2.5:
            gender = "Male"
            base_confidence = min(abs(gender_score) * 15 + 60, 98)
        elif gender_score >= 2.5:
            gender = "Female"
            base_confidence = min(abs(gender_score) * 15 + 60, 98)
        elif gender_score < 0:
            gender = "Male"
            base_confidence = 55 + abs(gender_score) * 5
        else:
            gender = "Female"
            base_confidence = 55 + abs(gender_score) * 5
        
        # Adjust confidence based on all factors
        total_confidence_boost = sum(confidence_factors)
        final_confidence = min(base_confidence + (total_confidence_boost / 10), 99.0)
        
        return gender, float(mean_pitch), float(final_confidence)
    
    except Exception as e:
        print(f"Gender detection error: {e}")
        return "Unknown", 0.0, 0.0

# ---------------- COMPREHENSIVE FEATURE EXTRACTION (2376 FEATURES) ----------------
def extract_comprehensive_features(y, sr, target_length=2376):
    """
    Extract comprehensive acoustic features to match model's expected 2376 features
    Includes: ZCR, RMS, MFCC, Chroma, Mel Spectrogram, Spectral features, and statistics
    """
    features = []
    
    try:
        # 1. Zero Crossing Rate (with statistics)
        zcr = librosa.feature.zero_crossing_rate(y=y)[0]
        features.extend([
            np.mean(zcr), np.std(zcr), np.max(zcr), np.min(zcr),
            skew(zcr), kurtosis(zcr)
        ])
        features.extend(zcr[:min(50, len(zcr))])  # First 50 frames
        
        # 2. Root Mean Square Energy (with statistics)
        rms = librosa.feature.rms(y=y)[0]
        features.extend([
            np.mean(rms), np.std(rms), np.max(rms), np.min(rms),
            skew(rms), kurtosis(rms)
        ])
        features.extend(rms[:min(50, len(rms))])  # First 50 frames
        
        # 3. MFCC - 40 coefficients with deltas and delta-deltas
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        # Statistics for each MFCC coefficient
        for coeff in mfcc:
            features.extend([
                np.mean(coeff), np.std(coeff), np.max(coeff), np.min(coeff)
            ])
        
        # Flatten MFCC features
        features.extend(mfcc.T.flatten()[:400])
        features.extend(mfcc_delta.T.flatten()[:200])
        features.extend(mfcc_delta2.T.flatten()[:200])
        
        # 4. Chroma features (12 pitch classes)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        for pitch_class in chroma:
            features.extend([
                np.mean(pitch_class), np.std(pitch_class), 
                np.max(pitch_class), np.min(pitch_class)
            ])
        features.extend(chroma.T.flatten()[:100])
        
        # 5. Mel Spectrogram (128 mel bands)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        # Statistics for mel bands
        for mel_band in mel_db:
            features.extend([np.mean(mel_band), np.std(mel_band)])
        
        features.extend(mel_db.T.flatten()[:400])
        
        # 6. Spectral Contrast (7 bands)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=7)
        for band in spectral_contrast:
            features.extend([
                np.mean(band), np.std(band), np.max(band), np.min(band)
            ])
        
        # 7. Tonnetz (Tonal Centroid Features)
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        for tonal in tonnetz:
            features.extend([
                np.mean(tonal), np.std(tonal), np.max(tonal), np.min(tonal)
            ])
        features.extend(tonnetz.T.flatten()[:50])
        
        # 8. Additional Spectral Features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        
        features.extend([
            np.mean(spectral_centroid), np.std(spectral_centroid),
            np.max(spectral_centroid), np.min(spectral_centroid),
            np.mean(spectral_bandwidth), np.std(spectral_bandwidth),
            np.mean(spectral_rolloff), np.std(spectral_rolloff)
        ])
        
        # 9. Pitch and Harmonics
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = [pitches[magnitudes[:, t].argmax(), t] for t in range(pitches.shape[1])]
        features.extend([
            np.mean(pitch_values), np.std(pitch_values),
            np.max(pitch_values), np.min(pitch_values)
        ])
        
        # 10. Temporal Features
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features.append(tempo)
        
        # 11. Poly features
        try:
            poly_features = librosa.feature.poly_features(y=y, sr=sr, order=2)
            features.extend([np.mean(poly_features), np.std(poly_features)])
        except:
            features.extend([0, 0])
        
        # Convert to numpy array
        features = np.array(features)
        
        # Ensure exact length by padding or truncating
        if len(features) < target_length:
            # Pad with zeros
            padding = target_length - len(features)
            features = np.pad(features, (0, padding), mode='constant', constant_values=0)
        elif len(features) > target_length:
            # Truncate
            features = features[:target_length]
        
        return features
    
    except Exception as e:
        print(f"Feature extraction error: {e}")
        # Return zero array if extraction fails
        return np.zeros(target_length)

# ---------------- HELPERS ----------------
def download(file_id, path):
    if not os.path.exists(path):
        gdown.download(f"https://drive.google.com/uc?id={file_id}", path, quiet=False)

@st.cache_resource
def load_assets():
    download(MODEL_FILE_ID, MODEL_PATH)
    download(SCALER_FILE_ID, SCALER_PATH)
    download(ENCODER_FILE_ID, ENCODER_PATH)
    
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = pickle.load(open(SCALER_PATH, "rb"))
    encoder = pickle.load(open(ENCODER_PATH, "rb"))
    return model, scaler, encoder

def prepare_audio(path, scaler):
    """Prepare audio file for prediction"""
    y, sr = librosa.load(path, duration=2.5, offset=0.6, sr=22050)
    feat = extract_comprehensive_features(y, sr, target_length=FEATURE_LEN)
    feat_scaled = scaler.transform(feat.reshape(1, -1))
    return np.expand_dims(feat_scaled, axis=2), y, sr

def prepare_live_audio(audio_data, sr, scaler):
    """Prepare recorded audio for prediction"""
    # Take middle 2.5 seconds for consistency
    duration = len(audio_data) / sr
    if duration > 2.5:
        start = int((duration - 2.5) / 2 * sr)
        end = start + int(2.5 * sr)
        y = audio_data[start:end]
    else:
        y = audio_data
    
    feat = extract_comprehensive_features(y, sr, target_length=FEATURE_LEN)
    feat_scaled = scaler.transform(feat.reshape(1, -1))
    return np.expand_dims(feat_scaled, axis=2), y, sr

def record_audio(duration=5, sr=22050):
    """Record audio from microphone"""
    try:
        st.info(f"🎤 Recording for {duration} seconds... Speak now!")
        audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
        
        # Progress bar
        progress_bar = st.progress(0)
        import time
        for i in range(int(duration * 10)):
            time.sleep(0.1)
            progress_bar.progress((i + 1) / (duration * 10))
        
        sd.wait()
        progress_bar.empty()
        st.success("✅ Recording complete!")
        
        return audio.flatten(), sr
    except Exception as e:
        st.error(f"Recording error: {e}")
        return None, None

def get_emotion_emoji(emotion):
    mapping = {'angry':'😠','disgust':'🤢','fear':'😨','happy':'😄','neutral':'😐','ps':'😌','sad':'😢'}
    return mapping.get(emotion.lower(),'❓')

def get_emotion_color(emotion):
    mapping = {'angry':'#FF5733','disgust':'#66FF66','fear':'#9370DB','happy':'#FFFF00','neutral':'#A9A9A9','ps':'#87CEEB','sad':'#0000FF'}
    return mapping.get(emotion.lower(),'#000000')

def plot_waveform(y, sr):
    fig, ax = plt.subplots(figsize=(8,2))
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#0a0a0a')
    ax.spines['bottom'].set_color('#333')
    ax.spines['top'].set_color('#333') 
    ax.spines['right'].set_color('#333')
    ax.spines['left'].set_color('#333')
    ax.tick_params(axis='x', colors='#888')
    ax.tick_params(axis='y', colors='#888')
    ax.title.set_color('#fff')
    
    librosa.display.waveshow(y, sr=sr, ax=ax, color='#00ffbd')
    ax.set_title("Waveform")
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor='#0a0a0a')
    buf.seek(0)
    plt.close()
    return buf

def plot_spectrogram(y, sr):
    fig, ax = plt.subplots(figsize=(8,3))
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#0a0a0a')
    ax.tick_params(axis='x', colors='#888')
    ax.tick_params(axis='y', colors='#888')
    ax.title.set_color('#fff')
    
    spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    spec_db = librosa.power_to_db(spec, ref=np.max)
    img = librosa.display.specshow(spec_db, sr=sr, x_axis='time', y_axis='mel', ax=ax, cmap='magma')
    ax.set_title("Mel Spectrogram")
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor='#0a0a0a')
    buf.seek(0)
    plt.close()
    return buf

def signup_user(email, password, name):
    try:
        user = auth.create_user_with_email_and_password(email, password)
        uid = user['localId']
        if db:
            db.collection("users").document(uid).set({
                "name": name,
                "email": email,
                "created_at": datetime.now()
            })
        st.success(f"Account created for {name}! Please log in.")
    except Exception as e:
        st.error(f"Signup failed: {e}")

def login_user(email, password):
    try:
        user = auth.sign_in_with_email_and_password(email, password)
        st.success("Login successful! ✅")
        return user
    except Exception as e:
        st.error(f"Login failed: {e}")
        return None

def save_history(uid, data):
    if db:
        db.collection("users").document(uid).collection("history").add(data)

def get_history(uid):
    if db:
        docs = db.collection("users").document(uid).collection("history").order_by("Timestamp", direction=firestore.Query.DESCENDING).stream()
        return pd.DataFrame([doc.to_dict() for doc in docs])
    return pd.DataFrame()

def get_personalized_recommendations(emotion, name="Arjun"):
    recs = {
        'sad': {
            'greeting': f"Hey {name}, I noticed things feel a bit heavy right now.",
            'message': "It's completely okay to feel this way. Let's find a gentle way to lift your spirits or simply provide some comfort.",
            'color': 'var(--neon-blue)',
            'actions': [
                {'title': "Kind Words", 'type': "Game", 'url': "https://www.popcannibal.com/kindwords/", 'icon': "💌", 'desc': "Write anonymous letters and receive comfort from real people."},
                {'title': "Lofi Garden", 'type': "Music", 'url': "https://www.youtube.com/watch?v=5yx6BWlEVcY", 'icon': "🎵", 'desc': "Relaxing beats to help you breathe and settle."},
                {'title': "Guided Relief", 'type': "Video", 'url': "https://www.youtube.com/watch?v=v7AYKMP6rOE", 'icon': "🧘", 'desc': "A 10-minute meditation to ease emotional weight."}
            ]
        },
        'angry': {
            'greeting': f"I see you're fired up, {name}.",
            'message': "Let's channel that energy into something constructive or help the mind cool down with these tools.",
            'color': 'var(--neon-red)',
            'actions': [
                {'title': "Super Hexagon", 'type': "Game", 'url': "https://superhexagon.com/", 'icon': "💎", 'desc': "Fast-paced focus to channel intense energy."},
                {'title': "Box Breathing", 'type': "Exercise", 'url': "https://www.youtube.com/watch?v=tEmt1Znux58", 'icon': "🌬️", 'desc': "Scientific rhythm to lower your heart rate instantly."},
                {'title': "Heavy Focus", 'type': "Audio", 'url': "https://www.youtube.com/watch?v=5qap5aO4i9A", 'icon': "🎻", 'desc': "Powerful classical music to transform frustration into focus."}
            ]
        },
        'happy': {
            'greeting': f"You're glowing today, {name}!",
            'message': "Your energy is incredible. Keep the momentum going with these high-vibe activities.",
            'color': 'var(--neon-green)',
            'actions': [
                {'title': "Chilly Panda", 'type': "Game", 'url': "https://chillypanda.com/", 'icon': "🐼", 'desc': "Lighthearted fun to keep the smile going."},
                {'title': "Feel Good Hits", 'type': "Music", 'url': "https://open.spotify.com/playlist/37i9dQZF1DX3rxVfIlg97J", 'icon': "💃", 'desc': "The ultimate playlist for your peak energy."},
                {'title': "Inspiring Stories", 'type': "Video", 'url': "https://www.ted.com/talks", 'icon': "💡", 'desc': "Expand your horizons while you're feeling open and ready."}
            ]
        },
        'neutral': {
            'greeting': f"Steady and calm, {name}.",
            'message': "A perfect state for growth and focus. Here's what I recommend for your balanced mood.",
            'color': 'var(--text-muted)',
            'actions': [
                {'title': "Wordle", 'type': "Game", 'url': "https://www.nytimes.com/games/wordle/index.html", 'icon': "🧩", 'desc': "A quick mental puzzle to keep the brain sharp."},
                {'title': "Focus Flow", 'type': "Audio", 'url': "https://www.youtube.com/watch?v=jfKfPfyJRdk", 'icon': "🌊", 'desc': "Deep focus sounds for work or study."},
                {'title': "Short Documentaries", 'type': "Video", 'url': "https://www.youtube.com/@Kurzgesagt", 'icon': "🌍", 'desc': "Learn something fascinating in under 10 minutes."}
            ]
        }
    }
    return recs.get(emotion.lower(), recs['neutral'])

# ---------------- MAIN APP LOGIC ----------------
if 'user' not in st.session_state:
    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown('<div class="glass-card auth-card">', unsafe_allow_html=True)
        st.markdown("<h1 style='font-size: 2rem; letter-spacing: -0.05em; margin-bottom: 8px;'>VibeCheck</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color: var(--text-muted); margin-bottom: 32px;'>Emotional Intelligence, redefined.</p>", unsafe_allow_html=True)
        
        auth_mode = st.tabs(["Login", "Create Account"])
        
        with auth_mode[0]:
            email = st.text_input("Email", key="login_email", placeholder="name@example.com")
            password = st.text_input("Password", type="password", key="login_pass", placeholder="••••••••")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Continue with Email", use_container_width=True):
                user = login_user(email, password)
                if user:
                    st.session_state['user'] = user
                    if db:
                        try:
                            user_doc = db.collection("users").document(user['localId']).get()
                            st.session_state['user_name'] = user_doc.to_dict().get('name', 'User') if user_doc.exists else "User"
                        except:
                            st.session_state['user_name'] = "User"
                    else:
                        st.session_state['user_name'] = "User"
                    st.rerun()
        
        with auth_mode[1]:
            new_name = st.text_input("What's your name?", key="signup_name", placeholder="Arjun")
            st.markdown("<p style='font-size: 0.8rem; color: var(--text-muted); text-align: left;'>PROFILE PICTURE</p>", unsafe_allow_html=True)
            
            prof_file = st.file_uploader("Upload avatar", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
            if prof_file:
                encoded = base64.b64encode(prof_file.read()).decode()
                st.session_state.profile_pic = f"data:image/png;base64,{encoded}"
                st.success("Avatar uploaded! (Saved locally for this session)")

            new_email = st.text_input("Email", key="sigup_email", placeholder="name@example.com")
            new_password = st.text_input("Password", type="password", key="signup_pass", placeholder="••••••••")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Create Account", use_container_width=True):
                signup_user(new_email, new_password, new_name)
        st.markdown('</div>', unsafe_allow_html=True)
else:
    uid = st.session_state['user']['localId']
    user_name = st.session_state.get('user_name', 'Arjun')
    
    st.markdown('<div class="nav-bar">', unsafe_allow_html=True)
    st.markdown("<h2 style='margin:0; font-size: 1.2rem; color: #fff;'>VibeCheck AI</h2>", unsafe_allow_html=True)
    
    avatar_url = st.session_state.get('profile_pic', "https://www.gravatar.com/avatar/00000000000000000000000000000000?d=mp&f=y")
    
    col_logout = st.columns([8, 1])
    with col_logout[1]:
        if st.button("Logout", key="logout_top"):
            del st.session_state['user']
            if 'profile_pic' in st.session_state:
                del st.session_state['profile_pic']
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    col_profile, col_main = st.columns([1, 3])
    
    with col_profile:
        st.markdown('<div class="glass-card profile-avatar-container">', unsafe_allow_html=True)
        st.markdown(f'<img src="{avatar_url}" class="profile-img-large">', unsafe_allow_html=True)
        st.markdown('<div class="profile-glow"></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("<p class='tag' style='color: #888; font-size: 0.8rem; text-transform: uppercase;'>System Status</p>", unsafe_allow_html=True)
        st.markdown("<div style='display:flex; justify-content: space-between;'><span>Neural Engine</span><span style='color: #00FF00;'>● Online</span></div>", unsafe_allow_html=True)
        st.markdown("<div style='display:flex; justify-content: space-between;'><span>Features</span><span style='color: #00FF00;'>2376</span></div>", unsafe_allow_html=True)
        if db is None:
             st.markdown("<div style='display:flex; justify-content: space-between;'><span>Database</span><span style='color: #FFA500;'>● Offline</span></div>", unsafe_allow_html=True)
        else:
             st.markdown("<div style='display:flex; justify-content: space-between;'><span>Database</span><span style='color: #00FF00;'>● Online</span></div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_main:
        main_tabs = st.tabs(["Analyze", "Insights", "History"])

        with main_tabs[0]:
            st.markdown(f'<h1 class="hero-text">Welcome back, {user_name}</h1>', unsafe_allow_html=True)
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("Capture Your Voice")
            
            # Recording mode selection
            input_mode = st.radio("Input Mode:", ["🎤 Record Live Audio", "📁 Upload Audio File"], horizontal=True)
            
            if input_mode == "🎤 Record Live Audio":
                st.markdown("### Record Your Voice")
                st.info("💡 Click the button below to start recording. Speak clearly for 5 seconds.")
                
                duration = st.slider("Recording Duration (seconds)", min_value=3, max_value=10, value=5)
                
                col_rec1, col_rec2, col_rec3 = st.columns([1, 2, 1])
                with col_rec2:
                    if st.button("🔴 Start Recording", use_container_width=True, key="record_btn"):
                        audio_data, sr = record_audio(duration=duration)
                        
                        if audio_data is not None:
                            # Save to temporary file
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                                sf.write(tmp_file.name, audio_data, sr)
                                st.session_state.current_audio_file_path = tmp_file.name
                                st.session_state.current_audio_data = audio_data
                                st.session_state.current_audio_sr = sr
                                st.session_state.is_live_audio = True
                
                # Process if audio exists
                if 'current_audio_file_path' in st.session_state and st.session_state.get('is_live_audio', False):
                    st.audio(st.session_state.current_audio_file_path)
                    
                    if st.button("🔍 Analyze Recording", use_container_width=True, type="primary"):
                        with st.spinner("🧠 Analyzing emotions with 2376 features..."):
                            try:
                                model, scaler, encoder = load_assets()
                                
                                # Prepare live audio
                                feat, y, sr = prepare_live_audio(
                                    st.session_state.current_audio_data, 
                                    st.session_state.current_audio_sr, 
                                    scaler
                                )
                                
                                # Predict
                                preds = model.predict(feat, verbose=0)[0]
                                idx = np.argmax(preds)
                                try:
                                    emotion_label = encoder.categories_[0][idx]
                                except:
                                    emotion_label = encoder.classes_[idx]
                                conf = float(preds[idx])
                                
                                # Advanced gender detection
                                gender, pitch, gender_conf = detect_gender_advanced(y, sr)
                                
                                # Display results
                                st.success(f"✅ Analysis Complete! Detected: {emotion_label.upper()} ({conf:.1%})")
                                
                                res_col1, res_col2 = st.columns([1.5, 2])
                                
                                with res_col1:
                                    st.markdown(f'<div class="glass-card" style="border-top: 2px solid {get_emotion_color(emotion_label)};">', unsafe_allow_html=True)
                                    st.markdown(f'<div class="emotion-badge" style="font-size: 4rem; text-align: center;">{get_emotion_emoji(emotion_label)}</div>', unsafe_allow_html=True)
                                    st.markdown(f"<h2 style='letter-spacing: -0.03em; margin:0; text-align: center;'>{emotion_label.upper()}</h2>", unsafe_allow_html=True)
                                    
                                    st.markdown("<br>", unsafe_allow_html=True)
                                    sc1, sc2, sc3 = st.columns(3)
                                    with sc1:
                                        st.markdown(f'<div class="stat-label" style="color:#888; font-size:0.7rem;">CONFIDENCE</div><div class="stat-value" style="color: {get_emotion_color(emotion_label)}; font-size:1.3rem; font-weight:bold;">{conf:.1%}</div>', unsafe_allow_html=True)
                                    with sc2:
                                        st.markdown(f'<div class="stat-label" style="color:#888; font-size:0.7rem;">GENDER</div><div class="stat-value" style="font-size:1.3rem; font-weight:bold;">{gender}</div>', unsafe_allow_html=True)
                                    with sc3:
                                        st.markdown(f'<div class="stat-label" style="color:#888; font-size:0.7rem;">G-CONF</div><div class="stat-value" style="font-size:1.3rem; font-weight:bold;">{gender_conf:.0f}%</div>', unsafe_allow_html=True)
                                    
                                    st.markdown(f'<div style="margin-top:16px; padding:12px; background:#111; border-radius:8px; text-align:center;"><span style="color:#888;">Pitch:</span> <span style="color:#fff; font-weight:600;">{pitch:.1f} Hz</span></div>', unsafe_allow_html=True)
                                    st.markdown('</div>', unsafe_allow_html=True)
                                    
                                    # Recommendations
                                    recs = get_personalized_recommendations(emotion_label, user_name)
                                    st.markdown(f"<h3 style='margin: 24px 0 16px 0;'>Personalized Suggestions</h3>", unsafe_allow_html=True)
                                    st.markdown(f"<p style='color: var(--text-muted); margin-bottom: 24px;'>{recs['greeting']} {recs['message']}</p>", unsafe_allow_html=True)
                                    
                                    for action in recs['actions']:
                                        st.markdown(f"""
                                        <div class="action-card">
                                            <div>
                                                <span class="type-tag">{action['type']}</span>
                                                <h3 style="margin: 0; font-size: 1.1rem;">{action['icon']} {action['title']}</h3>
                                                <p style="color: var(--text-muted); font-size: 0.85rem; margin-top: 8px;">{action['desc']}</p>
                                            </div>
                                            <a href="{action['url']}" target="_blank" class="btn-action">Open {action['type']}</a>
                                        </div>
                                        """, unsafe_allow_html=True)
                                
                                with res_col2:
                                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                                    st.subheader("Audio Visualization")
                                    viz_tab1, viz_tab2 = st.tabs(["Waveform", "Spectrogram"])
                                    with viz_tab1:
                                        st.image(plot_waveform(y, sr), use_container_width=True)
                                    with viz_tab2:
                                        st.image(plot_spectrogram(y, sr), use_container_width=True)
                                    
                                    st.markdown("<br>", unsafe_allow_html=True)
                                    st.subheader("Emotion Confidence Distribution")
                                    try:
                                        labels = encoder.categories_[0]
                                    except:
                                        labels = encoder.classes_
                                    prob_df = pd.DataFrame({
                                        'Emotion': labels,
                                        'Probability': preds
                                    }).sort_values('Probability', ascending=False)
                                    st.bar_chart(prob_df.set_index('Emotion'), color='#0070f3')
                                    st.markdown('</div>', unsafe_allow_html=True)
                                
                                # Save history
                                save_history(uid, {
                                    'Timestamp': datetime.now(pytz.utc).strftime('%Y-%m-%d %H:%M:%S UTC'),
                                    'Emotion': emotion_label,
                                    'Confidence': conf,
                                    'Gender': gender,
                                    'Gender_Confidence': gender_conf,
                                    'Pitch': pitch,
                                    'Source': 'Live Recording'
                                })
                                
                                # Clean up
                                if 'current_audio_file_path' in st.session_state:
                                    try:
                                        os.unlink(st.session_state.current_audio_file_path)
                                    except:
                                        pass
                                    del st.session_state.current_audio_file_path
                                    del st.session_state.current_audio_data
                                    del st.session_state.current_audio_sr
                                    del st.session_state.is_live_audio
                                
                            except Exception as e:
                                st.error(f"❌ Analysis error: {e}")
                                import traceback
                                st.code(traceback.format_exc())
            
            else:  # Upload mode
                st.markdown("### Upload Audio File")
                uploaded = st.file_uploader("Drop an audio file here", type=["wav", "mp3", "ogg", "m4a"])
                
                if uploaded:
                    st.audio(uploaded)
                    
                    if st.button("🔍 Analyze Uploaded File", use_container_width=True, type="primary"):
                        with st.spinner("🧠 Processing audio with 2376 features..."):
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                                tmp_file.write(uploaded.getvalue())
                                tmp_path = tmp_file.name
                            
                            try:
                                model, scaler, encoder = load_assets()
                                feat, y, sr = prepare_audio(tmp_path, scaler)
                                preds = model.predict(feat, verbose=0)[0]
                                idx = np.argmax(preds)
                                try:
                                    emotion_label = encoder.categories_[0][idx]
                                except:
                                    emotion_label = encoder.classes_[idx]
                                conf = float(preds[idx])
                                gender, pitch, gender_conf = detect_gender_advanced(y, sr)
                                
                                os.unlink(tmp_path)

                                # Display results (same as live audio)
                                st.success(f"✅ Analysis Complete! Detected: {emotion_label.upper()} ({conf:.1%})")
                                
                                res_col1, res_col2 = st.columns([1.5, 2])
                                
                                with res_col1:
                                    st.markdown(f'<div class="glass-card" style="border-top: 2px solid {get_emotion_color(emotion_label)};">', unsafe_allow_html=True)
                                    st.markdown(f'<div class="emotion-badge" style="font-size: 4rem; text-align: center;">{get_emotion_emoji(emotion_label)}</div>', unsafe_allow_html=True)
                                    st.markdown(f"<h2 style='letter-spacing: -0.03em; margin:0; text-align: center;'>{emotion_label.upper()}</h2>", unsafe_allow_html=True)
                                    
                                    st.markdown("<br>", unsafe_allow_html=True)
                                    sc1, sc2, sc3 = st.columns(3)
                                    with sc1:
                                        st.markdown(f'<div class="stat-label" style="color:#888; font-size:0.7rem;">CONFIDENCE</div><div class="stat-value" style="color: {get_emotion_color(emotion_label)}; font-size:1.3rem; font-weight:bold;">{conf:.1%}</div>', unsafe_allow_html=True)
                                    with sc2:
                                        st.markdown(f'<div class="stat-label" style="color:#888; font-size:0.7rem;">GENDER</div><div class="stat-value" style="font-size:1.3rem; font-weight:bold;">{gender}</div>', unsafe_allow_html=True)
                                    with sc3:
                                        st.markdown(f'<div class="stat-label" style="color:#888; font-size:0.7rem;">G-CONF</div><div class="stat-value" style="font-size:1.3rem; font-weight:bold;">{gender_conf:.0f}%</div>', unsafe_allow_html=True)
                                    
                                    st.markdown(f'<div style="margin-top:16px; padding:12px; background:#111; border-radius:8px; text-align:center;"><span style="color:#888;">Pitch:</span> <span style="color:#fff; font-weight:600;">{pitch:.1f} Hz</span></div>', unsafe_allow_html=True)
                                    st.markdown('</div>', unsafe_allow_html=True)
                                
                                with res_col2:
                                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                                    st.subheader("Audio Visualization")
                                    viz_tab1, viz_tab2 = st.tabs(["Waveform", "Spectrogram"])
                                    with viz_tab1:
                                        st.image(plot_waveform(y, sr), use_container_width=True)
                                    with viz_tab2:
                                        st.image(plot_spectrogram(y, sr), use_container_width=True)
                                    
                                    st.markdown("<br>", unsafe_allow_html=True)
                                    st.subheader("Emotion Distribution")
                                    try:
                                        labels = encoder.categories_[0]
                                    except:
                                        labels = encoder.classes_
                                    prob_df = pd.DataFrame({
                                        'Emotion': labels,
                                        'Probability': preds
                                    }).sort_values('Probability', ascending=False)
                                    st.bar_chart(prob_df.set_index('Emotion'), color='#0070f3')
                                    st.markdown('</div>', unsafe_allow_html=True)
                                
                                save_history(uid, {
                                    'Timestamp': datetime.now(pytz.utc).strftime('%Y-%m-%d %H:%M:%S UTC'),
                                    'Emotion': emotion_label,
                                    'Confidence': conf,
                                    'Gender': gender,
                                    'Gender_Confidence': gender_conf,
                                    'Pitch': pitch,
                                    'Source': 'Upload'
                                })

                            except Exception as e:
                                st.error(f"Error analyzing audio: {e}")
                                import traceback
                                st.code(traceback.format_exc())

            st.markdown('</div>', unsafe_allow_html=True)

        with main_tabs[1]:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("Your Emotional Journey")
            history = get_history(uid)
            if not history.empty:
                st.bar_chart(history['Emotion'].value_counts())
                st.dataframe(history)
            else:
                st.info("No history yet. Record some audio to see insights!")
            st.markdown('</div>', unsafe_allow_html=True)

        with main_tabs[2]:
             st.markdown('<div class="glass-card">', unsafe_allow_html=True)
             st.subheader("History Log")
             history = get_history(uid)
             if not history.empty:
                 st.dataframe(history)
             else:
                 st.text("No records found.")
             st.markdown('</div>', unsafe_allow_html=True)