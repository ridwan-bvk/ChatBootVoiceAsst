import google.generativeai as genai
import os
# Ganti dengan API key Anda
# API_KEY = "YOUR_API_KEY_HERE"  
# Konfigurasi Gemini API
API_KEY = "AIzaSyDiVUOkbmEWuAB3EasY6Ms8nT_OPCpMVbU"
print(f'api key:{API_KEY}')


try:
    # Konfigurasi
    genai.configure(api_key=API_KEY)
    
    

    # Inisialisasi model
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    # list model lain
    # for model in genai.list_models():
    #     print(f"model - {model.name}")

    # Test query
    response = model.generate_content("Apa ibu kota Indonesia?")
    
    print("API Key Valid!")
    print("Response:", response.text)
    
    

except Exception as e:
    print("Error:", str(e))