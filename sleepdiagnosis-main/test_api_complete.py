import requests
import json

# Test the API with the test audio file
url = "http://localhost:8000/analyze"
files = {'audio': open('test_recording.wav', 'rb')}

try:
    print("Testing API endpoint...")
    response = requests.post(url, files=files)
    print("Status Code:", response.status_code)
    
    if response.status_code == 200:
        result = response.json()
        print("✅ API Response successful!")
        print("Label:", result.get('label'))
        print("Probability:", result.get('probability'))
        print("Has spectrogram:", 'spectrogram' in result)
        if 'spectrogram' in result:
            spec = result['spectrogram']
            print("Spectrogram keys:", list(spec.keys()))
            print("Has frequency analysis:", 'frequency_analysis' in spec)
            print("Has image:", 'image_base64' in spec)
    else:
        print("❌ API Error:", response.text)
        
except Exception as e:
    print("❌ Connection Error:", e)
finally:
    files['audio'].close()