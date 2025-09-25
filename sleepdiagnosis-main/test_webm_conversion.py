import requests
import io

def test_webm_conversion():
    """Test WebM conversion with the backend"""
    print("🧪 Testing WebM Conversion")
    print("=" * 40)
    
    # Test backend health first
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Backend is healthy")
        else:
            print(f"❌ Backend health check failed: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Cannot connect to backend: {e}")
        return
    
    # Create a minimal WebM file (header only for testing)
    webm_header = b'\x1a\x45\xdf\xa3'  # WebM signature
    webm_data = webm_header + b'\x00' * 200  # Minimal WebM data
    
    print(f"📊 Testing with {len(webm_data)} bytes of WebM data...")
    
    try:
        # Send WebM data to backend
        file_data = io.BytesIO(webm_data)
        files = {'audio': ('recording.webm', file_data, 'audio/webm')}
        
        response = requests.post("http://localhost:8000/analyze", files=files, timeout=30)
        
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("🎉 WebM conversion successful!")
            print(f"Label: {result.get('label')}")
            print(f"Confidence: {result.get('confidence_score')}%")
        else:
            error_text = response.text
            print(f"❌ WebM conversion failed: {response.status_code}")
            try:
                import json
                error_json = json.loads(error_text)
                error_detail = error_json.get('detail', 'No detail')
                print(f"Error details: {error_detail}")
                
                # Check for specific conversion issues
                if 'pydub' in error_detail.lower():
                    print("💡 Suggestion: pydub conversion failed")
                elif 'ffmpeg' in error_detail.lower():
                    print("💡 Suggestion: FFmpeg conversion failed - may need FFmpeg installation")
                elif 'webm format detected' in error_detail.lower():
                    print("💡 Suggestion: WebM detected but conversion failed")
                    
            except:
                print(f"Raw error: {error_text[:200]}...")
                
    except Exception as e:
        print(f"❌ Request failed: {e}")
    
    print("\n" + "=" * 40)
    print("WebM conversion test complete!")

def test_wav_format():
    """Test that WAV format still works correctly"""
    print("\n🧪 Testing WAV Format (Control Test)")
    print("=" * 40)
    
    # Create a minimal but valid WAV file
    import struct
    
    sample_rate = 22050
    duration = 1
    samples = sample_rate * duration
    
    # Generate simple audio data
    audio_data = []
    for i in range(samples):
        sample = int(16383 * (i % 100) / 100)  # Simple sawtooth
        audio_data.append(struct.pack('<h', sample))
    
    # Create WAV header
    header = struct.pack('<4sI4s4sIHHIIHH4sI',
        b'RIFF', 36 + len(audio_data) * 2, b'WAVE',
        b'fmt ', 16, 1, 1, sample_rate, sample_rate * 2, 2, 16,
        b'data', len(audio_data) * 2
    )
    
    wav_data = header + b''.join(audio_data)
    
    try:
        file_data = io.BytesIO(wav_data)
        files = {'audio': ('test.wav', file_data, 'audio/wav')}
        
        response = requests.post("http://localhost:8000/analyze", files=files, timeout=30)
        
        print(f"WAV Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ WAV format working correctly!")
            print(f"Label: {result.get('label')}")
            print(f"Confidence: {result.get('confidence_score')}%")
            
            # Check spectrogram
            if result.get('spectrogram', {}).get('image_base64'):
                print("✅ Spectrogram generated successfully")
            else:
                print("⚠️ No spectrogram generated")
        else:
            print(f"❌ WAV format failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ WAV test failed: {e}")

if __name__ == "__main__":
    print("🔧 Auto-Record WebM Conversion Test")
    print("=" * 50)
    
    # Test WebM conversion capability
    test_webm_conversion()
    
    # Test WAV as control
    test_wav_format()
    
    print("\n" + "=" * 50)
    print("Testing complete!")
    print("\nRecommendations:")
    print("• If WebM works: Auto-record should work in all browsers")
    print("• If WebM fails: Use Firefox or file upload as workaround")
    print("• If WAV works: Backend is functioning correctly")