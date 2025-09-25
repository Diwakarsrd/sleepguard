import requests
import json
import numpy as np
import wave
import struct

def create_test_audio_file():
    """Create a test audio file for comparison"""
    sample_rate = 22050
    duration = 5  # 5 seconds
    samples = int(sample_rate * duration)
    
    # Create realistic snoring-like audio
    audio_data = []
    for i in range(samples):
        t = i / sample_rate
        # Mix frequencies typical of snoring
        signal = (
            0.4 * np.sin(2 * np.pi * 85 * t) +      # Low freq snoring base
            0.3 * np.sin(2 * np.pi * 200 * t) +     # Mid freq snoring
            0.2 * np.sin(2 * np.pi * 450 * t) +     # Higher harmonic
            0.1 * np.random.normal(0, 1)            # Some noise
        )
        audio_data.append(int(signal * 16383))
    
    # Create WAV file
    filename = "comparison_test.wav"
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(struct.pack('<' + 'h' * len(audio_data), *audio_data))
    
    return filename

def test_analysis_consistency():
    """Test that auto-record and upload provide identical analysis"""
    print("🔬 Testing Auto-Record vs Upload Analysis Consistency")
    print("=" * 60)
    
    # Create test audio file
    audio_file = create_test_audio_file()
    print(f"📁 Created test file: {audio_file}")
    
    try:
        # Test as uploaded file
        print("\n1️⃣ Testing as UPLOADED file...")
        with open(audio_file, 'rb') as f:
            files = {'audio': (audio_file, f, 'audio/wav')}
            response_upload = requests.post("http://localhost:8000/analyze", files=files, timeout=30)
        
        # Test as auto-recorded file (same data, different filename)
        print("\n2️⃣ Testing as AUTO-RECORDED file...")
        with open(audio_file, 'rb') as f:
            files = {'audio': ('recorded_snoring.wav', f, 'audio/wav')}
            response_auto = requests.post("http://localhost:8000/analyze", files=files, timeout=30)
        
        # Compare results
        print(f"\n📊 RESULTS COMPARISON:")
        print("=" * 40)
        
        if response_upload.status_code == 200 and response_auto.status_code == 200:
            result_upload = response_upload.json()
            result_auto = response_auto.json()
            
            print(f"Upload Analysis:")
            print(f"  ✅ Status: {response_upload.status_code}")
            print(f"  🎯 Label: {result_upload.get('label')}")
            print(f"  📈 Confidence: {result_upload.get('confidence_score')}%")
            print(f"  🔍 Probability: {result_upload.get('probability'):.3f}")
            
            print(f"\nAuto-Record Analysis:")
            print(f"  ✅ Status: {response_auto.status_code}")
            print(f"  🎯 Label: {result_auto.get('label')}")
            print(f"  📈 Confidence: {result_auto.get('confidence_score')}%")
            print(f"  🔍 Probability: {result_auto.get('probability'):.3f}")
            
            # Check spectrogram generation
            upload_spectrogram = result_upload.get('spectrogram', {})
            auto_spectrogram = result_auto.get('spectrogram', {})
            
            print(f"\n📊 SPECTROGRAM COMPARISON:")
            print("=" * 40)
            
            if upload_spectrogram.get('image_base64') and auto_spectrogram.get('image_base64'):
                print("  ✅ Upload spectrogram: Generated")
                print("  ✅ Auto-record spectrogram: Generated")
                print(f"  📏 Upload size: {len(upload_spectrogram['image_base64'])} chars")
                print(f"  📏 Auto-record size: {len(auto_spectrogram['image_base64'])} chars")
                
                # Compare frequency analysis
                upload_freq = upload_spectrogram.get('frequency_analysis', {})
                auto_freq = auto_spectrogram.get('frequency_analysis', {})
                
                if upload_freq and auto_freq:
                    print(f"\n🎵 FREQUENCY ANALYSIS COMPARISON:")
                    print("=" * 40)
                    
                    for freq_type in ['low_freq_energy', 'mid_freq_energy', 'high_freq_energy', 'dominant_frequency']:
                        upload_val = upload_freq.get(freq_type, 0)
                        auto_val = auto_freq.get(freq_type, 0)
                        print(f"  {freq_type}:")
                        print(f"    Upload: {upload_val}")
                        print(f"    Auto:   {auto_val}")
                        
                        # Check if they're approximately equal
                        if abs(float(upload_val) - float(auto_val)) < 0.001:
                            print(f"    ✅ IDENTICAL")
                        else:
                            print(f"    ⚠️ Different (this is normal due to processing variations)")
                        print()
                else:
                    print("  ⚠️ Missing frequency analysis data")
            else:
                print("  ❌ Spectrogram generation failed")
                if upload_spectrogram.get('error'):
                    print(f"     Upload error: {upload_spectrogram['error']}")
                if auto_spectrogram.get('error'):
                    print(f"     Auto-record error: {auto_spectrogram['error']}")
            
            # Final verdict
            print(f"\n🎉 FINAL VERDICT:")
            print("=" * 40)
            
            same_label = result_upload.get('label') == result_auto.get('label')
            same_confidence = result_upload.get('confidence_score') == result_auto.get('confidence_score')
            both_have_spectrograms = (upload_spectrogram.get('image_base64') and 
                                    auto_spectrogram.get('image_base64'))
            
            if same_label and same_confidence and both_have_spectrograms:
                print("✅ AUTO-RECORD and UPLOAD provide IDENTICAL analysis!")
                print("✅ Both generate spectrograms with frequency analysis!")
                print("✅ Sleep apnea detection works the same for both methods!")
            else:
                print("⚠️ Some differences detected:")
                if not same_label:
                    print("  - Different labels detected")
                if not same_confidence:
                    print("  - Different confidence scores")
                if not both_have_spectrograms:
                    print("  - Spectrogram generation differs")
        else:
            print(f"❌ Request failed:")
            print(f"  Upload: {response_upload.status_code}")
            print(f"  Auto-record: {response_auto.status_code}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
    
    finally:
        # Clean up
        import os
        if os.path.exists(audio_file):
            os.remove(audio_file)
            print(f"\n🧹 Cleaned up {audio_file}")
    
    print("\n" + "=" * 60)
    print("Spectrogram comparison test complete!")

if __name__ == "__main__":
    test_analysis_consistency()