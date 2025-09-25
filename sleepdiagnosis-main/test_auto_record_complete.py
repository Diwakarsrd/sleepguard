import requests
import numpy as np
import wave
import os

def create_test_audio():
    """Create a test WAV file for testing"""
    print("Creating test audio file...")
    
    # Generate a simple sine wave
    sample_rate = 22050
    duration = 5  # 5 seconds
    frequency = 440  # A4 note
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    # Add some noise to simulate snoring
    audio_data = np.sin(2 * np.pi * frequency * t) + 0.3 * np.random.normal(0, 1, len(t))
    audio_data = (audio_data * 32767).astype(np.int16)
    
    filename = "test_recording.wav"
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 2 bytes per sample
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())
    
    print(f"Created {filename} ({duration}s, {sample_rate}Hz)")
    return filename

def test_backend_analysis():
    """Test the backend analysis endpoint"""
    print("\n=== Testing Backend Analysis ===")
    
    # Test health endpoint
    try:
        health_response = requests.get("http://localhost:8000/health", timeout=5)
        print(f"Health check: {health_response.status_code}")
        if health_response.status_code == 200:
            print(f"Backend status: {health_response.json()}")
        else:
            print("❌ Backend health check failed")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to backend: {e}")
        return False
    
    # Create test audio
    audio_file = create_test_audio()
    
    try:
        # Test analysis endpoint
        print(f"Testing analysis with {audio_file}...")
        with open(audio_file, 'rb') as f:
            files = {'audio': (audio_file, f, 'audio/wav')}
            response = requests.post("http://localhost:8000/analyze", files=files, timeout=30)
        
        print(f"Analysis response: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Analysis successful!")
            print(f"Label: {result.get('label', 'unknown')}")
            print(f"Probability: {result.get('probability', 0):.3f}")
            print(f"Confidence: {result.get('confidence_score', 0)}%")
            
            # Check spectrogram data
            spectrogram = result.get('spectrogram', {})
            if spectrogram.get('image_base64'):
                print(f"✅ Spectrogram generated (base64 length: {len(spectrogram['image_base64'])})")
            else:
                print("⚠️ No spectrogram image generated")
                
            if spectrogram.get('frequency_analysis'):
                print("✅ Frequency analysis available")
                freq_analysis = spectrogram['frequency_analysis']
                print(f"  - Low freq energy: {freq_analysis.get('low_freq_energy', 'N/A')}")
                print(f"  - Mid freq energy: {freq_analysis.get('mid_freq_energy', 'N/A')}")
                print(f"  - High freq energy: {freq_analysis.get('high_freq_energy', 'N/A')}")
                print(f"  - Dominant frequency: {freq_analysis.get('dominant_frequency', 'N/A')} Hz")
            else:
                print("⚠️ No frequency analysis available")
                
            print("\n✅ Auto record analysis functionality should work!")
            return True
        else:
            print(f"❌ Analysis failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Analysis request failed: {e}")
        return False
    finally:
        # Clean up test file
        if os.path.exists(audio_file):
            os.remove(audio_file)
            print(f"Cleaned up {audio_file}")

if __name__ == "__main__":
    print("🎙️ Testing Auto Record Analysis Functionality")
    print("=" * 50)
    
    success = test_backend_analysis()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed! Auto record analysis should work properly.")
        print("\nTo use auto recording:")
        print("1. Open index.html in a browser")
        print("2. Go to the 'Auto Record' tab")
        print("3. Click 'Start Auto Recording'")
        print("4. Allow microphone access")
        print("5. Let it record for a few seconds")
        print("6. Click 'Stop Recording'")
        print("7. Click 'Analyze Recorded Audio'")
    else:
        print("❌ Tests failed. Please check the backend and try again.")