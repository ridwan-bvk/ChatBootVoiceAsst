from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import speech_recognition as sr
from gtts import gTTS
import os
import torch
import logging
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/audio'

# Load LLM model (Ganti dengan model yang diinginkan) [1]
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# IndoGPT (Ganti di bagian inisialisasi model)
# tokenizer = AutoTokenizer.from_pretrained("Wikidepia/IndoGPT-LTS")
# model = AutoModelForCausalLM.from_pretrained("Wikidepia/IndoGPT-LTS")

# Atau
# tokenizer = AutoTokenizer.from_pretrained("Indonesia-ai/gpt2-medium-indonesian-522M")
# model = AutoModelForCausalLM.from_pretrained("Indonesia-ai/gpt2-medium-indonesian-522M")
        

# Inisialisasi speech recognizer
r = sr.Recognizer()

[1]
def generate_response(input_text):
    """Generate response menggunakan LLM"""
    try:
        # Encode input dengan return_attention_mask=True
        inputs = tokenizer(
            input_text + tokenizer.eos_token, 
            return_tensors='pt',
            return_attention_mask=True  # ← Solusi utama
        )
        
        # Ambil input_ids dan attention_mask
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        # Tambahkan attention_mask ke generate()
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,  # ← Tambahkan ini
            max_length=1000,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_k=100,
            top_p=0.7,
            temperature=0.8
        )
        
        # IndoGPT (Ganti di bagian inisialisasi model)
        # tokenizer = AutoTokenizer.from_pretrained("Wikidepia/IndoGPT-LTS")
        # model = AutoModelForCausalLM.from_pretrained("Wikidepia/IndoGPT-LTS")

        # # Atau
        # tokenizer = AutoTokenizer.from_pretrained("Indonesia-ai/gpt2-medium-indonesian-522M")
        # model = AutoModelForCausalLM.from_pretrained("Indonesia-ai/gpt2-medium-indonesian-522M")
        
        response = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
        return response
    except Exception as e:
        return "Maaf, terjadi kesalahan dalam memproses permintaan Anda."


# def generate_response(input_text):
#     system_prompt = "Anda adalah asisten yang membantu pengguna dalam bahasa Indonesia. Berikan jawaban yang ramah dan informatif."
#     full_text = system_prompt + "\n\nUser: " + input_text + "\nAsisten:"
    
#     inputs = tokenizer(
#         input_text + tokenizer.eos_token, 
#         return_tensors='pt',
#         max_length=512,
#         truncation=True
#     )
    
#     # Parameter yang lebih optimal untuk percakapan
#     output = model.generate(
#         inputs.input_ids,
#         attention_mask=inputs.attention_mask,
#         max_new_tokens=100,
#         repetition_penalty=1.2,  # Mengurangi pengulangan
#         temperature=0.7,         # Lebih deterministik (0.1-1.0)
#         top_k=50,                # Filter vocabulary
#         top_p=0.9,               # Nucleus sampling
#         do_sample=True,
#         pad_token_id=tokenizer.eos_token_id
#     )
    
#     response = tokenizer.decode(output[0], skip_special_tokens=True)
#     return response.split(tokenizer.eos_token)[-1].strip()  # Ambil bagian terakhir

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