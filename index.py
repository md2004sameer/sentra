from flask import Flask, request, jsonify, render_template_string
from detoxify import Detoxify
import easyocr
import os
import time
from io import BytesIO
import base64
from PIL import Image
import numpy as np
import speech_recognition as sr
from pydub import AudioSegment
import tempfile

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# ---------------- Toxicity Analyzer ----------------
# It's better to load the model once when the application starts
detox_model = Detoxify('original')

# Real Detoxify categories used in both backend and frontend
DETOXIFY_CATEGORIES = {
    "toxic": "General Toxicity",
    "severe_toxic": "Severe Toxicity",
    "obscene": "Obscene Language",
    "threat": "Threats",
    "insult": "Insults",
    "identity_hate": "Identity Hate"
}

# Threshold for considering a category as "detected"
TOXICITY_THRESHOLD = 0.5

def convert_float32_to_float(obj):
    """Recursively convert float32 to float for JSON serialization"""
    if isinstance(obj, dict):
        return {key: convert_float32_to_float(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_float32_to_float(element) for element in obj]
    elif isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, float):
        return obj
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def predict_detox(text: str) -> dict:
    """Get actual Detoxify predictions"""
    results = detox_model.predict(text)
    return convert_float32_to_float(results)

def analyze_toxicity(text: str) -> dict:
    """Analyze text and return only detected categories with meaningful scores"""
    raw_scores = predict_detox(text)

    detected_categories = {}
    overall_score = 0
    detected_count = 0

    for category, score in raw_scores.items():
        if score >= TOXICITY_THRESHOLD:
            detected_categories[category] = score
            overall_score += score
            detected_count += 1

    # Calculate weighted overall score
    if detected_count > 0:
        overall_score = (overall_score / detected_count) * 100
    else:
        # If no category is above the threshold, the overall score is based on the max score but scaled down
        overall_score = max(raw_scores.values()) * 50

    return {
        "detected_categories": detected_categories,
        "overall_score": min(overall_score, 100),  # Cap at 100%
        "all_scores": raw_scores,
        "is_toxic": len(detected_categories) > 0
    }

# ---------------- Image Text Extractor ----------------
reader = easyocr.Reader(['en'], gpu=False)

def extract_text_from_image(image_data) -> str:
    """Extracts text from image data (bytes)"""
    try:
        image = Image.open(BytesIO(image_data))
        # Convert to numpy array for easyocr
        img_np = np.array(image)
        results = reader.readtext(img_np, detail=0)
        return "\n".join(results) if results else "No text detected in image"
    except Exception as e:
        return f"Error extracting text: {str(e)}"

# ---------------- Audio Speech-to-Text ----------------
def allowed_audio_file(filename):
    """Check if the audio file format is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'wav', 'mp3', 'm4a', 'flac', 'aac'}

def speech_to_text(audio_data, original_filename):
    """Convert speech audio to text using Google Speech Recognition"""
    recognizer = sr.Recognizer()
    try:
        # Use pydub to open audio file from memory
        audio_segment = AudioSegment.from_file(BytesIO(audio_data))
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_wav:
            audio_segment.export(temp_wav.name, format="wav")
            
            with sr.AudioFile(temp_wav.name) as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = recognizer.record(source)
                
                try:
                    return recognizer.recognize_google(audio)
                except sr.UnknownValueError:
                    return "Could not understand audio"
                except sr.RequestError as e:
                    return f"Speech recognition error: {str(e)}"

    except Exception as e:
        return f"Audio processing error: {str(e)}"

# ---------------- Modern Frontend ----------------
frontend_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🚀 SENTRA - AI Content Moderator</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #6366f1;
            --primary-dark: #4f46e5;
            --secondary: #8b5cf6;
            --accent: #06b6d4;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --dark: #1f2937;
            --darker: #111827;
            --light: #f8fafc;
            --gray: #6b7280;
            --gray-light: #9ca3af;
            --gradient: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #06b6d4 100%);
            --gradient-dark: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #0891b2 100%);
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Inter', sans-serif;
        }
        
        body {
            background: var(--darker);
            color: var(--light);
            min-height: 100vh;
            padding: 20px;
            background-image: 
                radial-gradient(circle at 10% 20%, rgba(99, 102, 241, 0.1) 0%, transparent 20%),
                radial-gradient(circle at 90% 80%, rgba(139, 92, 246, 0.1) 0%, transparent 20%);
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .header {
            text-align: center;
            margin-bottom: 50px;
            padding: 40px 20px;
            background: var(--gradient);
            border-radius: 24px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
            position: relative;
            overflow: hidden;
        }
        
        .header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.05'%3E%3Ccircle cx='30' cy='30' r='2'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
        }
        
        .header h1 {
            font-size: 4rem;
            font-weight: 700;
            margin-bottom: 10px;
            background: linear-gradient(45deg, #fff, #e0e7ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            position: relative;
        }
        
        .header p {
            font-size: 1.3rem;
            opacity: 0.9;
            font-weight: 300;
        }
        
        .app-card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 24px;
            box-shadow: 0 25px 50px rgba(0, 0, 0, 0.4);
            overflow: hidden;
            margin-bottom: 40px;
        }
        
        .tab-container {
            display: flex;
            background: rgba(255, 255, 255, 0.05);
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .tab {
            padding: 20px 30px;
            cursor: pointer;
            border: none;
            background: none;
            font-size: 1.1rem;
            font-weight: 500;
            color: var(--gray-light);
            transition: all 0.3s ease;
            flex: 1;
            text-align: center;
            position: relative;
            overflow: hidden;
        }
        
        .tab::before {
            content: '';
            position: absolute;
            bottom: 0;
            left: 50%;
            width: 0;
            height: 3px;
            background: var(--accent);
            transition: all 0.3s ease;
            transform: translateX(-50%);
        }
        
        .tab.active {
            color: white;
            background: rgba(255, 255, 255, 0.05);
        }
        
        .tab.active::before {
            width: 80%;
        }
        
        .tab:hover {
            color: white;
            background: rgba(255, 255, 255, 0.02);
        }
        
        .tab-content {
            display: none;
            padding: 40px;
        }
        
        .tab-content.active {
            display: block;
            animation: slideUp 0.5s ease;
        }
        
        @keyframes slideUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .input-group {
            margin-bottom: 30px;
        }
        
        .input-group label {
            display: block;
            margin-bottom: 12px;
            font-weight: 600;
            color: white;
            font-size: 1.1rem;
        }
        
        .textarea {
            width: 100%;
            min-height: 150px;
            padding: 20px;
            border: 2px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            resize: vertical;
            font-size: 1rem;
            transition: all 0.3s ease;
            background: rgba(255, 255, 255, 0.05);
            color: white;
        }
        
        .textarea:focus {
            outline: none;
            border-color: var(--accent);
            box-shadow: 0 0 0 3px rgba(6, 182, 212, 0.2);
        }
        
        .textarea::placeholder {
            color: var(--gray-light);
        }
        
        .file-upload {
            position: relative;
            border: 2px dashed rgba(255, 255, 255, 0.2);
            border-radius: 16px;
            padding: 50px 30px;
            text-align: center;
            transition: all 0.3s ease;
            cursor: pointer;
            background: rgba(255, 255, 255, 0.02);
        }
        
        .file-upload:hover {
            border-color: var(--accent);
            background: rgba(255, 255, 255, 0.05);
        }
        
        .file-upload.dragover {
            border-color: var(--success);
            background: rgba(16, 185, 129, 0.1);
        }
        
        .file-input {
            position: absolute;
            width: 100%;
            height: 100%;
            top: 0;
            left: 0;
            opacity: 0;
            cursor: pointer;
        }
        
        .upload-icon {
            font-size: 4rem;
            color: var(--accent);
            margin-bottom: 15px;
            filter: drop-shadow(0 4px 8px rgba(6, 182, 212, 0.3));
        }
        
        .btn {
            padding: 16px 35px;
            border: none;
            border-radius: 12px;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            position: relative;
            overflow: hidden;
        }
        
        .btn::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
            transition: left 0.5s;
        }
        
        .btn:hover::before {
            left: 100%;
        }
        
        .btn-primary {
            background: var(--gradient);
            color: white;
            box-shadow: 0 8px 25px rgba(99, 102, 241, 0.4);
        }
        
        .btn-primary:hover {
            background: var(--gradient-dark);
            transform: translateY(-3px);
            box-shadow: 0 12px 35px rgba(99, 102, 241, 0.6);
        }
        
        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none !important;
            box-shadow: none !important;
        }
        
        .results-container {
            margin-top: 40px;
            display: none;
        }
        
        .results-container.show {
            display: block;
            animation: slideUp 0.5s ease;
        }
        
        .toxicity-meter {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 20px;
            padding: 30px;
            margin-bottom: 30px;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .overall-score {
            font-size: 4rem;
            font-weight: 700;
            margin-bottom: 10px;
            background: var(--gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .score-label {
            font-size: 1.3rem;
            color: var(--gray-light);
            margin-bottom: 20px;
        }
        
        .verdict-badge {
            display: inline-block;
            padding: 12px 25px;
            border-radius: 25px;
            font-weight: 600;
            font-size: 1.1rem;
            margin-top: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        
        .verdict-safe {
            background: var(--success);
            color: white;
        }
        
        .verdict-toxic {
            background: var(--danger);
            color: white;
        }
        
        .meter-bar {
            height: 16px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            overflow: hidden;
            margin: 25px 0;
        }
        
        .meter-fill {
            height: 100%;
            border-radius: 8px;
            transition: width 0.8s ease;
            background: var(--gradient);
        }
        
        .category-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }
        
        .category-card {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 25px;
            text-align: center;
            transition: all 0.3s ease;
            backdrop-filter: blur(10px);
        }
        
        .category-card.detected {
            border-color: var(--danger);
            background: rgba(239, 68, 68, 0.1);
            transform: scale(1.02);
            box-shadow: 0 8px 25px rgba(239, 68, 68, 0.3);
        }
        
        .category-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 30px rgba(0,0,0,0.3);
        }
        
        .category-name {
            font-weight: 600;
            margin-bottom: 15px;
            color: white;
            font-size: 1.2rem;
        }
        
        .category-score {
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 10px;
        }
        
        .score-high { color: var(--danger); }
        .score-low { color: var(--success); }
        
        .detection-status {
            font-size: 0.9rem;
            font-weight: 600;
            padding: 6px 15px;
            border-radius: 15px;
            display: inline-block;
        }
        
        .status-detected {
            background: var(--danger);
            color: white;
        }
        
        .status-clean {
            background: var(--success);
            color: white;
        }
        
        .extracted-text {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 20px;
            margin-top: 20px;
            white-space: pre-wrap;
            max-height: 200px;
            overflow-y: auto;
            color: white;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 40px;
        }
        
        .spinner {
            width: 50px;
            height: 50px;
            border: 4px solid rgba(255, 255, 255, 0.1);
            border-top: 4px solid var(--accent);
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .alert {
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 25px;
            display: none;
            border-left: 4px solid;
        }
        
        .alert-error {
            background: rgba(239, 68, 68, 0.1);
            border-color: var(--danger);
            color: #fca5a5;
        }
        
        .alert-success {
            background: rgba(16, 185, 129, 0.1);
            border-color: var(--success);
            color: #6ee7b7;
        }
        
        .image-preview, .audio-preview {
            max-width: 100%;
            max-height: 300px;
            border-radius: 12px;
            margin-top: 15px;
            display: none;
            border: 2px solid rgba(255, 255, 255, 0.1);
        }
        
        .audio-visualizer {
            width: 100%;
            height: 80px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            margin-top: 15px;
            display: none;
            position: relative;
            overflow: hidden;
        }
        
        .audio-wave {
            position: absolute;
            bottom: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, var(--accent), var(--secondary));
            opacity: 0.3;
            animation: wave 2s infinite linear;
        }
        
        @keyframes wave {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }
        
        .history-panel {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 16px;
            padding: 25px;
            margin-top: 30px;
        }
        
        .history-item {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 15px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .history-item:hover {
            border-color: var(--accent);
            transform: translateX(5px);
        }
        
        .no-results {
            text-align: center;
            padding: 60px 40px;
            color: var(--gray-light);
        }
        
        .feature-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 25px;
            margin-top: 40px;
        }
        
        .feature-card {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 16px;
            padding: 30px;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: all 0.3s ease;
        }
        
        .feature-card:hover {
            transform: translateY(-5px);
            border-color: var(--accent);
        }
        
        .feature-icon {
            font-size: 3rem;
            margin-bottom: 15px;
        }
        
        @media (max-width: 768px) {
            .header h1 { font-size: 2.5rem; }
            .tab { padding: 15px 20px; font-size: 1rem; }
            .category-grid { grid-template-columns: 1fr; }
            .overall-score { font-size: 3rem; }
            .feature-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 SENTRA AI</h1>
            <p>Advanced Multi-Modal Content Moderation Platform</p>
        </div>
        
        <div class="feature-grid">
            <div class="feature-card">
                <div class="feature-icon">📝</div>
                <h3>Text Analysis</h3>
                <p>Analyze text content for toxicity, hate speech, and inappropriate language</p>
            </div>
            <div class="feature-card">
                <div class="feature-icon">🖼️</div>
                <h3>Image OCR</h3>
                <p>Extract and analyze text from images with advanced OCR technology</p>
            </div>
            <div class="feature-card">
                <div class="feature-icon">🎤</div>
                <h3>Audio Analysis</h3>
                <p>Convert speech to text and analyze audio content for toxicity</p>
            </div>
        </div>
        
        <div class="app-card">
            <div class="tab-container">
                <button class="tab active" onclick="switchTab('text')">📝 Text</button>
                <button class="tab" onclick="switchTab('image')">🖼️ Image</button>
                <button class="tab" onclick="switchTab('audio')">🎤 Audio</button>
                <button class="tab" onclick="switchTab('history')">📊 History</button>
            </div>
            
            <!-- Text Analysis Tab -->
            <div id="text-tab" class="tab-content active">
                <div class="input-group">
                    <label for="text-input">Enter text to analyze for toxicity:</label>
                    <textarea 
                        id="text-input" 
                        class="textarea" 
                        placeholder="Type or paste your text here..."
                    ></textarea>
                </div>
                <button id="analyze-text-btn" class="btn btn-primary" onclick="analyzeText()">
                    🔍 Analyze Text
                </button>
                
                <div id="text-results" class="results-container">
                    <!-- Results will be populated here -->
                </div>
            </div>
            
            <!-- Image Analysis Tab -->
            <div id="image-tab" class="tab-content">
                <div class="input-group">
                    <label>Upload an image to extract and analyze text:</label>
                    <div class="file-upload" id="drop-zone-image">
                        <input type="file" id="image-input" class="file-input" accept="image/*">
                        <div class="upload-icon">🖼️</div>
                        <p>Drag & drop an image or click to browse</p>
                        <p style="font-size: 0.9rem; color: var(--gray-light); margin-top: 5px;">
                            Supports JPG, PNG, GIF (Max 16MB)
                        </p>
                    </div>
                    <img id="image-preview" class="image-preview" alt="Image preview">
                </div>
                <button id="analyze-image-btn" class="btn btn-primary" onclick="analyzeImage()" disabled>
                    🔍 Analyze Image
                </button>
                
                <div id="image-results" class="results-container">
                    <!-- Results will be populated here -->
                </div>
            </div>
            
            <!-- Audio Analysis Tab -->
            <div id="audio-tab" class="tab-content">
                <div class="input-group">
                    <label>Upload an audio file to convert speech to text:</label>
                    <div class="file-upload" id="drop-zone-audio">
                        <input type="file" id="audio-input" class="file-input" accept="audio/*">
                        <div class="upload-icon">🎤</div>
                        <p>Drag & drop an audio file or click to browse</p>
                        <p style="font-size: 0.9rem; color: var(--gray-light); margin-top: 5px;">
                            Supports WAV, MP3, M4A, FLAC (Max 16MB)
                        </p>
                    </div>
                    <div id="audio-visualizer" class="audio-visualizer">
                        <div class="audio-wave"></div>
                    </div>
                    <audio id="audio-preview" class="audio-preview" controls style="display: none;"></audio>
                </div>
                <button id="analyze-audio-btn" class="btn btn-primary" onclick="analyzeAudio()" disabled>
                    🔊 Analyze Audio
                </button>
                
                <div id="audio-results" class="results-container">
                    <!-- Results will be populated here -->
                </div>
            </div>
            
            <!-- History Tab -->
            <div id="history-tab" class="tab-content">
                <div class="history-panel">
                    <h3>Recent Analyses</h3>
                    <div id="history-list">
                        <!-- History items will be populated here -->
                    </div>
                    <button class="btn" onclick="clearHistory()" style="margin-top: 15px; background: rgba(239, 68, 68, 0.2); color: #fca5a5;">
                        🗑️ Clear History
                    </button>
                </div>
            </div>
        </div>
        
        <div id="loading" class="loading">
            <div class="spinner"></div>
            <p>Analyzing content... This may take a few seconds.</p>
        </div>
        
        <div id="alert" class="alert"></div>
    </div>

    <script>
        let currentTab = 'text';
        let currentImage = null;
        let currentAudio = null;
        let analysisHistory = JSON.parse(localStorage.getItem('sentra_history')) || [];
        
        // Initialize history display
        updateHistoryDisplay();
        
        function switchTab(tabName) {
            // Hide all tabs
            document.querySelectorAll('.tab-content').forEach(tab => {
                tab.classList.remove('active');
            });
            document.querySelectorAll('.tab').forEach(tab => {
                tab.classList.remove('active');
            });
            
            // Show selected tab
            document.getElementById(tabName + '-tab').classList.add('active');
            document.querySelector(`.tab[onclick="switchTab('${tabName}')"]`).classList.add('active');
            currentTab = tabName;
        }
        
        function showAlert(message, type) {
            const alert = document.getElementById('alert');
            alert.textContent = message;
            alert.className = `alert alert-${type}`;
            alert.style.display = 'block';
            
            setTimeout(() => {
                alert.style.display = 'none';
            }, 5000);
        }
        
        function setLoading(loading) {
            document.getElementById('loading').style.display = loading ? 'block' : 'none';
        }
        
        function analyzeText() {
            const text = document.getElementById('text-input').value.trim();
            if (!text) {
                showAlert('Please enter some text to analyze.', 'error');
                return;
            }
            
            setLoading(true);
            
            fetch('/analyze/text', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: text })
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                setLoading(false);
                if (data.error) {
                    showAlert(data.error, 'error');
                    return;
                }
                
                // Add to history
                addToHistory({
                    type: 'text',
                    content: text,
                    results: data,
                    timestamp: new Date().toISOString()
                });
                
                displayTextResults(data);
            })
            .catch(error => {
                setLoading(false);
                showAlert('Error analyzing text: ' + error.message, 'error');
                console.error('Analysis error:', error);
            });
        }
        
        function analyzeImage() {
            if (!currentImage) {
                showAlert('Please select an image first.', 'error');
                return;
            }
            
            setLoading(true);
            
            const formData = new FormData();
            formData.append('image', currentImage);
            
            fetch('/analyze/image', {
                method: 'POST',
                body: formData
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                setLoading(false);
                if (data.error) {
                    showAlert(data.error, 'error');
                    return;
                }
                
                // Add to history
                addToHistory({
                    type: 'image',
                    content: 'Image analysis',
                    results: data,
                    timestamp: new Date().toISOString()
                });
                
                displayImageResults(data);
            })
            .catch(error => {
                setLoading(false);
                showAlert('Error analyzing image: ' + error.message, 'error');
                console.error('Analysis error:', error);
            });
        }
        
        function analyzeAudio() {
            if (!currentAudio) {
                showAlert('Please select an audio file first.', 'error');
                return;
            }
            
            setLoading(true);
            
            const formData = new FormData();
            formData.append('audio', currentAudio);
            
            fetch('/analyze/audio', {
                method: 'POST',
                body: formData
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                setLoading(false);
                if (data.error) {
                    showAlert(data.error, 'error');
                    return;
                }
                
                // Add to history
                addToHistory({
                    type: 'audio',
                    content: 'Audio analysis',
                    results: data,
                    timestamp: new Date().toISOString()
                });
                
                displayAudioResults(data);
            })
            .catch(error => {
                setLoading(false);
                showAlert('Error analyzing audio: ' + error.message, 'error');
                console.error('Analysis error:', error);
            });
        }
        
        function addToHistory(analysis) {
            analysisHistory.unshift(analysis);
            // Keep only last 10 analyses
            if (analysisHistory.length > 10) {
                analysisHistory = analysisHistory.slice(0, 10);
            }
            localStorage.setItem('sentra_history', JSON.stringify(analysisHistory));
            updateHistoryDisplay();
        }
        
        function updateHistoryDisplay() {
            const historyList = document.getElementById('history-list');
            if (analysisHistory.length === 0) {
                historyList.innerHTML = '<p style="text-align: center; color: var(--gray-light); padding: 40px;">No analysis history yet.</p>';
                return;
            }
            
            historyList.innerHTML = analysisHistory.map((item, index) => {
                const overallScore = item.results.overall_score || 0;
                const isToxic = item.results.is_toxic || false;
                const typeIcon = item.type === 'text' ? '📝' : item.type === 'image' ? '🖼️' : '🎤';
                const contentPreview = (item.content || '').substring(0, 100) + ((item.content || '').length > 100 ? '...' : '');

                // ***FIXED ERROR HERE***
                // The stray '<div...>' tag before the return statement was removed.
                return `
                <div class="history-item" onclick="loadHistoryItem(${index})">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong>${typeIcon} ${item.type.charAt(0).toUpperCase() + item.type.slice(1)} Analysis</strong>
                            <div style="font-size: 0.9rem; color: var(--gray-light); margin-top: 5px;">
                                ${contentPreview}
                            </div>
                        </div>
                        <div style="text-align: right;">
                            <div class="overall-score" style="font-size: 1.5rem;">${overallScore.toFixed(1)}%</div>
                            <span class="detection-status ${isToxic ? 'status-detected' : 'status-clean'}">
                                ${isToxic ? '🚨 Toxic' : '✅ Clean'}
                            </span>
                        </div>
                    </div>
                    <div style="font-size: 0.8rem; color: var(--gray-light); margin-top: 10px;">
                        ${new Date(item.timestamp).toLocaleString()}
                    </div>
                </div>
                `;
            }).join('');
        }
        
        function loadHistoryItem(index) {
            const item = analysisHistory[index];
            if (!item) return;
            
            // Switch to the appropriate tab
            switchTab(item.type);
            
            // Populate the input based on type
            if (item.type === 'text') {
                document.getElementById('text-input').value = item.content;
                displayTextResults(item.results);
            } else if (item.type === 'image') {
                // For images, we can't reload the image file, but we can show the results
                const resultsDiv = document.getElementById('image-results');
                resultsDiv.innerHTML = createImageResultsHTML(item.results);
                resultsDiv.classList.add('show');
            } else if (item.type === 'audio') {
                // For audio, similarly show the results
                const resultsDiv = document.getElementById('audio-results');
                resultsDiv.innerHTML = createAudioResultsHTML(item.results);
                resultsDiv.classList.add('show');
            }
            
            showAlert('History item loaded successfully!', 'success');
        }
        
        function clearHistory() {
            if (confirm('Are you sure you want to clear all analysis history?')) {
                analysisHistory = [];
                localStorage.removeItem('sentra_history');
                updateHistoryDisplay();
                showAlert('History cleared successfully!', 'success');
            }
        }
        
        function displayTextResults(data) {
            const resultsDiv = document.getElementById('text-results');
            resultsDiv.innerHTML = createResultsHTML(data);
            resultsDiv.classList.add('show');
        }
        
        function displayImageResults(data) {
            const resultsDiv = document.getElementById('image-results');
            resultsDiv.innerHTML = createImageResultsHTML(data);
            resultsDiv.classList.add('show');
        }
        
        function displayAudioResults(data) {
            const resultsDiv = document.getElementById('audio-results');
            resultsDiv.innerHTML = createAudioResultsHTML(data);
            resultsDiv.classList.add('show');
        }
        
        function createResultsHTML(data) {
            const overallScore = data.overall_score || 0;
            const isToxic = data.is_toxic || false;
            const detectedCategories = data.detected_categories || {};
            const allScores = data.all_scores || {};
            
            return `
            <div class="toxicity-meter">
                <div class="overall-score">${overallScore.toFixed(1)}%</div>
                <div class="score-label">Overall Toxicity Score</div>
                <div class="meter-bar">
                    <div class="meter-fill" style="width: ${overallScore}%"></div>
                </div>
                <div class="verdict-badge ${isToxic ? 'verdict-toxic' : 'verdict-safe'}">
                    ${isToxic ? '🚨 TOXIC CONTENT DETECTED' : '✅ CONTENT IS CLEAN'}
                </div>
            </div>
            
            <div class="category-grid">
                ${Object.keys(DETOXIFY_CATEGORIES).map(category => {
                    const score = allScores[category] || 0;
                    const isDetected = detectedCategories[category] !== undefined;
                    const displayName = DETOXIFY_CATEGORIES[category];
                    
                    return `
                    <div class="category-card ${isDetected ? 'detected' : ''}">
                        <div class="category-name">${displayName}</div>
                        <div class="category-score ${score > 0.7 ? 'score-high' : 'score-low'}">
                            ${(score * 100).toFixed(1)}%
                        </div>
                        <div class="detection-status ${isDetected ? 'status-detected' : 'status-clean'}">
                            ${isDetected ? '🚨 Detected' : '✅ Clean'}
                        </div>
                    </div>
                    `;
                }).join('')}
            </div>
            
            ${data.extracted_text ? `
            <div style="margin-top: 30px;">
                <h3>📝 Extracted Text</h3>
                <div class="extracted-text">${data.extracted_text}</div>
            </div>
            ` : ''}
            `;
        }
        
        function createImageResultsHTML(data) {
            return createResultsHTML(data);
        }
        
        function createAudioResultsHTML(data) {
            return createResultsHTML(data);
        }
        
        // File upload handling
        function setupFileUpload() {
            // Image upload
            const imageInput = document.getElementById('image-input');
            const dropZoneImage = document.getElementById('drop-zone-image');
            const imagePreview = document.getElementById('image-preview');
            const analyzeImageBtn = document.getElementById('analyze-image-btn');
            
            // Audio upload
            const audioInput = document.getElementById('audio-input');
            const dropZoneAudio = document.getElementById('drop-zone-audio');
            const audioPreview = document.getElementById('audio-preview');
            const audioVisualizer = document.getElementById('audio-visualizer');
            const analyzeAudioBtn = document.getElementById('analyze-audio-btn');
            
            function preventDefaults(e) {
                e.preventDefault();
                e.stopPropagation();
            }
            
            // Drag and drop handlers for image
            ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                dropZoneImage.addEventListener(eventName, preventDefaults, false);
            });
            
            ['dragenter', 'dragover'].forEach(eventName => {
                dropZoneImage.addEventListener(eventName, () => dropZoneImage.classList.add('dragover'), false);
            });
            
            ['dragleave', 'drop'].forEach(eventName => {
                dropZoneImage.addEventListener(eventName, () => dropZoneImage.classList.remove('dragover'), false);
            });
            
            dropZoneImage.addEventListener('drop', (e) => {
                if (e.dataTransfer.files.length > 0) handleImageFile(e.dataTransfer.files[0]);
            }, false);
            
            imageInput.addEventListener('change', (e) => {
                if (e.target.files.length > 0) handleImageFile(e.target.files[0]);
            });
            
            function handleImageFile(file) {
                if (!file.type.match('image.*')) {
                    showAlert('Please select a valid image file.', 'error');
                    return;
                }
                currentImage = file;
                analyzeImageBtn.disabled = false;
                const reader = new FileReader();
                reader.onload = (e) => {
                    imagePreview.src = e.target.result;
                    imagePreview.style.display = 'block';
                };
                reader.readAsDataURL(file);
                showAlert('Image ready for analysis!', 'success');
            }
            
            // Drag and drop handlers for audio
            ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                dropZoneAudio.addEventListener(eventName, preventDefaults, false);
            });
            
            ['dragenter', 'dragover'].forEach(eventName => {
                dropZoneAudio.addEventListener(eventName, () => dropZoneAudio.classList.add('dragover'), false);
            });
            
            ['dragleave', 'drop'].forEach(eventName => {
                dropZoneAudio.addEventListener(eventName, () => dropZoneAudio.classList.remove('dragover'), false);
            });
            
            dropZoneAudio.addEventListener('drop', (e) => {
                if (e.dataTransfer.files.length > 0) handleAudioFile(e.dataTransfer.files[0]);
            }, false);
            
            audioInput.addEventListener('change', (e) => {
                if (e.target.files.length > 0) handleAudioFile(e.target.files[0]);
            });
            
            function handleAudioFile(file) {
                if (!file.type.match('audio.*')) {
                    showAlert('Please select a valid audio file.', 'error');
                    return;
                }
                currentAudio = file;
                analyzeAudioBtn.disabled = false;
                const reader = new FileReader();
                reader.onload = (e) => {
                    audioPreview.src = e.target.result;
                    audioPreview.style.display = 'block';
                    audioVisualizer.style.display = 'block';
                };
                reader.readAsDataURL(file);
                showAlert('Audio ready for analysis!', 'success');
            }
        }
        
        // Detoxify categories for frontend
        const DETOXIFY_CATEGORIES = {
            "toxic": "General Toxicity",
            "severe_toxic": "Severe Toxicity", 
            "obscene": "Obscene Language",
            "threat": "Threats",
            "insult": "Insults",
            "identity_hate": "Identity Hate"
        };
        
        // Initialize the application
        document.addEventListener('DOMContentLoaded', function() {
            setupFileUpload();
            showAlert('🚀 SENTRA AI is ready! Choose a tab to start analyzing content.', 'success');
        });
    </script>
</body>
</html>
"""

# ---------------- Flask Routes ----------------
@app.route('/')
def home():
    return render_template_string(frontend_html)

@app.route('/analyze/text', methods=['POST'])
def analyze_text_route():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text'].strip()
        if not text:
            return jsonify({'error': 'Text cannot be empty'}), 400
        
        toxicity_results = analyze_toxicity(text)
        return jsonify(toxicity_results)
        
    except Exception as e:
        return jsonify({'error': f'Text analysis failed: {str(e)}'}), 500

@app.route('/analyze/image', methods=['POST'])
def analyze_image_route():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        extracted_text = extract_text_from_image(image_file.read())
        
        if extracted_text.startswith('Error') or extracted_text == 'No text detected in image':
            return jsonify({
                'extracted_text': extracted_text,
                'overall_score': 0,
                'is_toxic': False,
                'detected_categories': {},
                'all_scores': {category: 0.0 for category in DETOXIFY_CATEGORIES}
            })
        
        toxicity_results = analyze_toxicity(extracted_text)
        toxicity_results['extracted_text'] = extracted_text
        return jsonify(toxicity_results)
        
    except Exception as e:
        return jsonify({'error': f'Image analysis failed: {str(e)}'}), 500

@app.route('/analyze/audio', methods=['POST'])
def analyze_audio_route():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'No audio selected'}), 400
        
        if not allowed_audio_file(audio_file.filename):
            return jsonify({'error': 'Invalid audio format. Supported: WAV, MP3, M4A, FLAC, AAC'}), 400
        
        extracted_text = speech_to_text(audio_file.read(), audio_file.filename)
        
        if extracted_text.startswith('Error') or extracted_text == 'Could not understand audio':
            return jsonify({
                'extracted_text': extracted_text,
                'overall_score': 0,
                'is_toxic': False,
                'detected_categories': {},
                'all_scores': {category: 0.0 for category in DETOXIFY_CATEGORIES}
            })
        
        toxicity_results = analyze_toxicity(extracted_text)
        toxicity_results['extracted_text'] = extracted_text
        return jsonify(toxicity_results)
        
    except Exception as e:
        return jsonify({'error': f'Audio analysis failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
