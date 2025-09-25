import requests
import io
import struct

def test_auto_recorded_audio_format():
    """Test with audio format similar to browser-recorded audio"""
    print("🎙️ Testing Auto-Recorded Audio Format")
    print("=" * 50)
    
    # Test 1: Check backend health
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
    
    # Test 2: Create audio blob similar to MediaRecorder output
    print("\n📊 Creating browser-like audio blob...")
    
    # Create a simple audio blob similar to what MediaRecorder would produce
    # This simulates the blob created by browser's MediaRecorder API
    sample_rate = 48000  # Common browser recording rate
    duration = 3  # 3 seconds
    samples = int(sample_rate * duration)
    
    # Generate sine wave audio data (similar to what might be recorded)
    import math
    audio_data = []
    for i in range(samples):
        # Create a complex waveform that might resemble snoring
        t = i / sample_rate
        # Mix of frequencies to simulate snoring sounds
        sample = (
            0.3 * math.sin(2 * math.pi * 80 * t) +   # Low frequency base
            0.2 * math.sin(2 * math.pi * 200 * t) +  # Mid frequency
            0.1 * math.sin(2 * math.pi * 400 * t)    # Higher harmonic
        )
        # Add some noise
        sample += 0.05 * (2 * (i % 1000) / 1000 - 1)
        # Convert to 16-bit integer
        sample_int = int(sample * 32767)
        sample_int = max(-32768, min(32767, sample_int))
        audio_data.append(struct.pack('<h', sample_int))
    
    # Create WAV header for the audio data
    def create_wav_header(sample_rate, num_samples, num_channels=1, bits_per_sample=16):
        byte_rate = sample_rate * num_channels * bits_per_sample // 8
        block_align = num_channels * bits_per_sample // 8
        data_size = num_samples * num_channels * bits_per_sample // 8
        file_size = 36 + data_size
        
        header = struct.pack('<4sI4s4sIHHIIHH4sI',
            b'RIFF', file_size, b'WAVE',
            b'fmt ', 16, 1, num_channels, sample_rate, byte_rate, block_align, bits_per_sample,
            b'data', data_size
        )
        return header
    
    wav_header = create_wav_header(sample_rate, samples)
    wav_data = wav_header + b''.join(audio_data)
    
    print(f"✅ Created WAV audio: {len(wav_data)} bytes, {duration}s at {sample_rate}Hz")
    
    # Test 3: Send to backend (similar to auto-recorded audio)
    print("\n🔍 Testing analysis...")
    try:
        # Create file-like object similar to browser blob
        audio_blob = io.BytesIO(wav_data)
        files = {'audio': ('recorded_audio.wav', audio_blob, 'audio/wav')}
        
        response = requests.post("http://localhost:8000/analyze", files=files, timeout=60)
        
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Auto-recorded audio analysis successful!")
            print(f"Label: {result.get('label')}")
            print(f"Confidence: {result.get('confidence_score')}%")
            
            # Check spectrogram
            if result.get('spectrogram', {}).get('image_base64'):
                print(f"✅ Spectrogram generated: {len(result['spectrogram']['image_base64'])} chars")
            else:
                print("⚠️ No spectrogram generated")
                if result.get('spectrogram', {}).get('error'):
                    print(f"Spectrogram error: {result['spectrogram']['error']}")
                    
        else:
            error_text = response.text
            print(f"❌ Analysis failed: {response.status_code}")
            print(f"Error details: {error_text}")
            
            # Try to extract detailed error message
            try:
                import json
                error_json = json.loads(error_text)
                print(f"Detailed error: {error_json.get('detail', 'No detail available')}")
            except:
                print(f"Raw error response: {error_text}")
                
    except Exception as e:
        print(f"❌ Request failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("Auto-recorded audio format test complete!")

if __name__ == "__main__":
    test_auto_recorded_audio_format()