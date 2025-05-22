from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import speech_recognition as sr
from gtts import gTTS
import os
import torch
import logging
from datetime import datetime
import google.generativeai as genai
from dotenv import load_dotenv
# Load environment variables
load_dotenv() 

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/audio'

print(os.path.abspath('.env'))  
print("API Key:", os.getenv('GOOGLE_API_KEY')) 

# Konfigurasi Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')  # Atau langsung masukkan API key di sini
print(f"api key:{GEMINI_API_KEY}")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')
r = sr.Recognizer()


# Atau menggunakan DeepSeek API (Alternatif)
# Daftar di https://platform.deepseek.com/signup
# DEEPSEEK_API_KEY = "your_api_key_here"

def generate_response(prompt):
    """Menggunakan Google Gemini API"""
    try:
        response = model.generate_content(
            f"Berikan jawaban dalam Bahasa sunda dan informatif: {prompt}"
            
        )
        return response.text
    except Exception as e:
        return f"Maaf terjadi error: {str(e)}"

# Alternatif menggunakan DeepSeek API
"""
def generate_response(prompt):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }
    
    data = {
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "model": "deepseek-chat",
        "temperature": 0.7
    }
    
    response = requests.post(
        "https://api.deepseek.com/v1/chat/completions",
        headers=headers,
        json=data
    )
    
    return response.json()['choices'][0]['message']['content']
"""

@app.route('/')
def home():
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['message']
    print(f"[USER INPUT]: {user_input}")  # Tampilkan di terminal
    
    response = generate_response(user_input)
    print(f"[BOT RESPONSE]: {response}")  # Tampilkan di terminal
    
    return jsonify({'response': response})

@app.route('/voice', methods=['POST'])
def voice():
    try:
        # Membuat folder upload jika tidak ada
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        input_filename = f"input_{timestamp}.wav"
        output_filename = f"response_{timestamp}.mp3"
        
        # Save audio file
        audio_file = request.files['audio']
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
        audio_file.save(input_path)
        
        # Speech-to-text
        with sr.AudioFile(input_path) as source:
            audio = r.record(source)
            text = r.recognize_google(audio, language='id-ID')
            logging.info(f"Recognized text: {text}")
        
        # Generate response
        response_text = generate_response(text)
        
        # Text-to-speech
        tts = gTTS(text=response_text, lang='id')
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        tts.save(output_path)
        
        return jsonify({
            'text': response_text,
            'audio': f"/static/audio/{output_filename}"
        })
        
    except sr.UnknownValueError:
        return jsonify({'error': 'Tidak dapat mengenali suara'}), 400
    except sr.RequestError as e:
        return jsonify({'error': f'Error service speech recognition: {str(e)}'}), 500
    except Exception as e:
        logging.error(f"Error in voice processing: {str(e)}")
        return jsonify({'error': 'Terjadi kesalahan internal'}), 500

if __name__ == '__main__':
    app.run(debug=True)