import requests
import io
import os

def test_error_scenarios():
    """Test various error scenarios to help debug the Internal Server Error"""
    base_url = "http://localhost:8000"
    
    print("🔍 Testing Error Scenarios")
    print("=" * 50)
    
    # Test 1: Health check
    print("\n1. Testing backend health...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        print(f"✅ Health check: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return
    
    # Test 2: Empty file
    print("\n2. Testing empty file...")
    try:
        empty_file = io.BytesIO(b"")
        files = {'audio': ('empty.wav', empty_file, 'audio/wav')}
        response = requests.post(f"{base_url}/analyze", files=files, timeout=30)
        print(f"Empty file response: {response.status_code}")
        if response.status_code != 200:
            print(f"Error details: {response.text}")
    except Exception as e:
        print(f"Empty file test error: {e}")
    
    # Test 3: Invalid file format
    print("\n3. Testing invalid file format...")
    try:
        text_content = b"This is not an audio file"
        text_file = io.BytesIO(text_content)
        files = {'audio': ('fake.wav', text_file, 'audio/wav')}
        response = requests.post(f"{base_url}/analyze", files=files, timeout=30)
        print(f"Invalid format response: {response.status_code}")
        if response.status_code != 200:
            print(f"Error details: {response.text}")
    except Exception as e:
        print(f"Invalid format test error: {e}")
    
    # Test 4: Missing dependencies
    print("\n4. Testing dependencies...")
    try:
        import librosa
        import numpy as np
        import matplotlib
        import joblib
        print("✅ All required dependencies are available")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
    
    # Test 5: File permissions
    print("\n5. Testing file permissions...")
    try:
        test_dir = "tmp_uploads"
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
            print(f"✅ Created {test_dir} directory")
        
        test_file = os.path.join(test_dir, "test_permissions.txt")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        print("✅ File operations working correctly")
    except Exception as e:
        print(f"❌ File permission error: {e}")
    
    # Test 6: Very small audio file (but valid)
    print("\n6. Testing very small audio file...")
    try:
        # Create minimal WAV header + some data
        wav_header = b'RIFF\x24\x08\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x22\x56\x00\x00\x44\xac\x00\x00\x02\x00\x10\x00data\x00\x08\x00\x00'
        # Add some random audio data (2048 samples)
        import struct
        audio_data = b''.join(struct.pack('<h', i % 1000) for i in range(1024))
        mini_wav = wav_header + audio_data
        
        small_file = io.BytesIO(mini_wav)
        files = {'audio': ('small.wav', small_file, 'audio/wav')}
        response = requests.post(f"{base_url}/analyze", files=files, timeout=30)
        print(f"Small file response: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Small file analysis successful: {result.get('label')}")
        else:
            print(f"Error details: {response.text}")
    except Exception as e:
        print(f"Small file test error: {e}")
    
    print("\n" + "=" * 50)
    print("Error scenario testing complete!")

if __name__ == "__main__":
    test_error_scenarios()