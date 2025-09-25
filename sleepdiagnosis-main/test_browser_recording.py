import requests
import io
import json

def test_browser_recording_scenarios():
    """Test various browser recording scenarios that might cause issues"""
    print("🎙️ Testing Browser Recording Scenarios")
    print("=" * 60)
    
    # Test 1: Health check
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
    
    # Test scenarios that might cause issues
    test_scenarios = [
        {
            "name": "Empty Blob",
            "data": b"",
            "filename": "empty_recording.wav",
            "content_type": "audio/wav"
        },
        {
            "name": "Very Small Audio (< 1KB)",
            "data": b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x22\x56\x00\x00\x44\xac\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00",
            "filename": "tiny_recording.wav",
            "content_type": "audio/wav"
        },
        {
            "name": "WebM Format (Browser Default)",
            "data": b"\x1a\x45\xdf\xa3" + b"\x00" * 100,  # WebM signature + dummy data
            "filename": "recording.webm",
            "content_type": "audio/webm"
        },
        {
            "name": "Unknown Content Type",
            "data": create_minimal_wav(),
            "filename": "recording.wav",
            "content_type": "application/octet-stream"
        },
        {
            "name": "No Filename",
            "data": create_minimal_wav(),
            "filename": "",
            "content_type": "audio/wav"
        }
    ]
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n{i}. Testing: {scenario['name']}")
        print("-" * 40)
        
        try:
            # Create file-like object
            file_data = io.BytesIO(scenario['data'])
            files = {
                'audio': (
                    scenario['filename'], 
                    file_data, 
                    scenario['content_type']
                )
            }
            
            print(f"   Data size: {len(scenario['data'])} bytes")
            print(f"   Filename: '{scenario['filename']}'")
            print(f"   Content-Type: {scenario['content_type']}")
            
            # Send request
            response = requests.post(
                "http://localhost:8000/analyze", 
                files=files, 
                timeout=30
            )
            
            print(f"   Response: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ SUCCESS: {result.get('label')} ({result.get('confidence_score')}%)")
                
                if result.get('spectrogram', {}).get('image_base64'):
                    print(f"   📊 Spectrogram: Generated")
                else:
                    print(f"   📊 Spectrogram: Failed - {result.get('spectrogram', {}).get('error', 'Unknown error')}")
                    
            else:
                try:
                    error_detail = response.json().get('detail', 'No detail')
                except:
                    error_detail = response.text[:200]
                print(f"   ❌ FAILED: {error_detail}")
                
        except Exception as e:
            print(f"   💥 EXCEPTION: {e}")
    
    print(f"\n{'=' * 60}")
    print("Browser recording scenario testing complete!")
    print("\nRecommendations based on results:")
    print("- If WebM fails: Browser needs format conversion")
    print("- If empty fails: Need better validation")
    print("- If small files fail: Need minimum size handling")

def create_minimal_wav():
    """Create a minimal but valid WAV file"""
    import struct
    
    # Create a very short WAV file (1 second, mono, 22050Hz)
    sample_rate = 22050
    duration = 1
    samples = sample_rate * duration
    
    # Generate simple tone
    audio_data = []
    for i in range(samples):
        sample = int(16383 * (i % 100) / 100)  # Simple sawtooth wave
        audio_data.append(struct.pack('<h', sample))
    
    # WAV header
    header = struct.pack('<4sI4s4sIHHIIHH4sI',
        b'RIFF', 36 + len(audio_data) * 2, b'WAVE',
        b'fmt ', 16, 1, 1, sample_rate, sample_rate * 2, 2, 16,
        b'data', len(audio_data) * 2
    )
    
    return header + b''.join(audio_data)

if __name__ == "__main__":
    test_browser_recording_scenarios()