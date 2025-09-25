import numpy as np
import librosa
import librosa.display
import joblib
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import io
import base64

def extract_features(file_path: str, sr: int = 22050):
    try:
        print(f"Loading audio file: {file_path}")
        if not os.path.exists(file_path):
            raise ValueError(f"Audio file not found: {file_path}")
            
        file_size = os.path.getsize(file_path)
        print(f"Audio file size: {file_size} bytes")
        
        if file_size == 0:
            raise ValueError("Audio file is empty")
            
        # Load audio with error handling and format detection
        # Prioritize MP3 and WAV for optimal spectrogram generation
        try:
            # Check if it's MP3 format for optimal processing
            if file_path.lower().endswith('.mp3'):
                print("MP3 format detected - using optimized loading for spectrogram generation")
                # Use librosa with specific parameters optimized for MP3 and spectrogram analysis
                y, sr = librosa.load(file_path, sr=22050, mono=True, duration=300.0, offset=0.0)
                print(f"MP3 audio loaded with optimized settings: duration={len(y)/sr:.2f}s, sample_rate={sr}Hz")
            else:
                # Standard loading for other formats
                y, sr = librosa.load(file_path, sr=sr, mono=True, duration=300.0)
                print(f"Audio loaded with librosa: duration={len(y)/sr:.2f}s, sample_rate={sr}Hz")
        except Exception as librosa_error:
            print(f"Librosa loading failed: {librosa_error}")
            
            # Try alternative loading methods for browser recordings
            try:
                import soundfile as sf
                y, sr = sf.read(file_path)
                if len(y.shape) > 1:  # Convert stereo to mono
                    y = y.mean(axis=1)
                # Resample if needed
                if sr != 22050:
                    y = librosa.resample(y, orig_sr=sr, target_sr=22050)
                    sr = 22050
                print(f"Audio loaded with soundfile: duration={len(y)/sr:.2f}s, sample_rate={sr}Hz")
            except Exception as sf_error:
                print(f"Soundfile loading failed: {sf_error}")
                
                # Last resort: try to convert using ffmpeg if available
                try:
                    # Try loading as raw audio
                    with open(file_path, 'rb') as f:
                        data = f.read()
                    print(f"File contains {len(data)} bytes of data")
                    
                    # Check if it's a WebM file (common for browser recordings)
                    if data.startswith(b'\x1a\x45\xdf\xa3'):  # WebM signature
                        print("WebM format detected, attempting conversion...")
                        try:
                            # Method 1: Try using pydub for WebM conversion
                            from pydub import AudioSegment
                            from pydub.utils import which
                            
                            # Check if ffmpeg is available
                            if not which("ffmpeg"):
                                # Try common Windows ffmpeg locations
                                import subprocess
                                ffmpeg_paths = [
                                    "ffmpeg",
                                    "C:\\ffmpeg\\bin\\ffmpeg.exe",
                                    "C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe",
                                    os.path.join(os.getcwd(), "ffmpeg.exe")
                                ]
                                
                                ffmpeg_found = False
                                for path in ffmpeg_paths:
                                    try:
                                        subprocess.run([path, "-version"], capture_output=True, check=True)
                                        AudioSegment.converter = path
                                        ffmpeg_found = True
                                        print(f"Found ffmpeg at: {path}")
                                        break
                                    except (subprocess.CalledProcessError, FileNotFoundError):
                                        continue
                                
                                if not ffmpeg_found:
                                    raise ValueError("FFmpeg not found. WebM conversion requires FFmpeg to be installed.")
                            
                            # Load WebM file with pydub
                            print("Loading WebM file with pydub...")
                            audio = AudioSegment.from_file(file_path, format="webm")
                            
                            # Convert to numpy array
                            samples = audio.get_array_of_samples()
                            if audio.channels == 2:
                                samples = samples[0::2]  # Take only left channel for stereo
                            
                            # Convert to float32 and normalize
                            y = np.array(samples, dtype=np.float32) / 32768.0
                            sr = audio.frame_rate
                            
                            # Resample to target sample rate if needed
                            if sr != 22050:
                                print(f"Resampling from {sr}Hz to 22050Hz...")
                                y = librosa.resample(y, orig_sr=sr, target_sr=22050)
                                sr = 22050
                                
                            print(f"WebM conversion successful: duration={len(y)/sr:.2f}s, sample_rate={sr}Hz")
                            
                        except ImportError:
                            raise ValueError("WebM format detected but pydub is not installed. Please install pydub and ffmpeg for WebM support.")
                        except Exception as webm_error:
                            print(f"WebM conversion error: {webm_error}")
                            # Try alternative method with ffmpeg-python
                            try:
                                import ffmpeg
                                import tempfile
                                
                                print("Trying ffmpeg-python for WebM conversion...")
                                
                                # Create temporary WAV file
                                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                                    temp_wav_path = temp_wav.name
                                
                                # Convert WebM to WAV using ffmpeg
                                (
                                    ffmpeg
                                    .input(file_path)
                                    .output(temp_wav_path, acodec='pcm_s16le', ac=1, ar=22050)
                                    .overwrite_output()
                                    .run(capture_stdout=True, capture_stderr=True)
                                )
                                
                                # Load the converted WAV file
                                y, sr = librosa.load(temp_wav_path, sr=22050, mono=True)
                                
                                # Clean up temporary file
                                os.unlink(temp_wav_path)
                                
                                print(f"FFmpeg conversion successful: duration={len(y)/sr:.2f}s, sample_rate={sr}Hz")
                                
                            except Exception as ffmpeg_error:
                                raise ValueError(f"WebM conversion failed with both pydub and ffmpeg: {webm_error}, {ffmpeg_error}")
                    else:
                        # If we get here, the file format is unsupported
                        raise ValueError(f"Unsupported audio format. Original errors: librosa='{librosa_error}', soundfile='{sf_error}'")
                    
                except Exception as final_error:
                    raise ValueError(f"Cannot load audio file. File may be corrupted or in unsupported format. Details: {final_error}")
        
        if y.size == 0:
            raise ValueError("Empty audio data after loading")
            
        if len(y) < 1024:
            raise ValueError("Audio too short for analysis (minimum 1024 samples required)")

        # Extract audio features
        rms = librosa.feature.rms(y=y).mean()
        zcr = librosa.feature.zero_crossing_rate(y).mean()
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_means = mfcc.mean(axis=1)
        
        print(f"Features extracted: RMS={rms:.4f}, ZCR={zcr:.4f}, SC={spectral_centroid:.1f}")

        vector = np.hstack([rms, zcr, spectral_centroid, spectral_bandwidth,
                            spectral_rolloff, mfcc_means]).astype(float)

        # Generate spectrogram data with error handling
        try:
            spectrogram_data = generate_spectrogram(y, sr)
            print(f"Spectrogram generation: {'successful' if spectrogram_data.get('image_base64') else 'failed'}")
        except Exception as e:
            print(f"Warning: Spectrogram generation failed: {e}")
            spectrogram_data = {
                "image_base64": None,
                "frequency_analysis": {},
                "time_duration": len(y)/sr,
                "error": str(e)
            }

        summary = {
            "rms": float(rms),
            "zcr": float(zcr),
            "spectral_centroid": float(spectral_centroid),
            "spectral_bandwidth": float(spectral_bandwidth),
            "spectral_rolloff": float(spectral_rolloff),
            "mfcc_1": float(mfcc_means[0]),
        }
        return {"vector": vector, "summary": summary, "spectrogram": spectrogram_data}
        
    except Exception as e:
        print(f"Error during audio processing: {e}")
        raise ValueError(f"Audio processing failed: {str(e)}")

def generate_spectrogram(y, sr):
    """Generate spectrogram data for visualization - optimized for MP3 format"""
    try:
        print(f"Generating enhanced spectrogram for audio: length={len(y)}, sr={sr}")  # Debug log
        
        # Ensure we have enough audio data
        if len(y) < 1024:
            print("Audio too short for spectrogram analysis")
            return {
                "image_base64": None,
                "frequency_analysis": {},
                "time_duration": 0.0,
                "error": "Audio too short for analysis"
            }
        
        # Enhanced STFT parameters optimized for MP3 audio and sleep apnea analysis
        # Increased frequency resolution for better MP3 spectrogram quality
        n_fft = 4096  # Increased from 2048 for better frequency resolution with MP3
        hop_length = 1024  # Adjusted for MP3 format
        
        # Create spectrogram with enhanced parameters
        D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window='hann')
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        
        # Get time and frequency axes
        times = librosa.times_like(S_db, sr=sr, hop_length=hop_length)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        
        print(f"Enhanced spectrogram shape: {S_db.shape}, Frequency range: {freqs[0]:.1f}-{freqs[-1]:.1f} Hz")
        
        # Generate enhanced base64 encoded plot optimized for MP3 analysis
        plt.figure(figsize=(16, 10))  # Larger figure for better MP3 spectrogram detail
        
        # Main spectrogram plot with enhanced parameters
        plt.subplot(2, 1, 1)
        # Use enhanced frequency range for MP3 sleep apnea analysis
        librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz',
                                fmax=1000, cmap='plasma', shading='gouraud')  # Enhanced for MP3
        plt.colorbar(format='%+2.0f dB')
        plt.title('🎵 Enhanced MP3 Audio Spectrogram - Sleep Apnea Analysis', fontsize=16, fontweight='bold')
        plt.ylabel('Frequency (Hz)', fontsize=14)
        
        # Enhanced average frequency spectrum plot
        plt.subplot(2, 1, 2)
        avg_spectrum = np.mean(S_db, axis=1)
        relevant_freqs = freqs[freqs <= 1000]  # Focus on sleep apnea relevant frequency range
        relevant_spectrum = avg_spectrum[:len(relevant_freqs)]
        
        # Enhanced visualization with frequency band highlighting
        plt.plot(relevant_freqs, relevant_spectrum, linewidth=2.5, color='darkblue', label='Average Spectrum')
        plt.fill_between(relevant_freqs, relevant_spectrum, alpha=0.4, color='lightblue')
        
        # Highlight important frequency bands for sleep apnea
        plt.axvspan(0, 100, alpha=0.2, color='green', label='Low Freq (0-100Hz)')
        plt.axvspan(100, 500, alpha=0.2, color='yellow', label='Mid Freq (100-500Hz)')
        plt.axvspan(500, 1000, alpha=0.2, color='red', label='High Freq (500Hz+)')
        
        plt.title('🔍 Enhanced Frequency Analysis - Sleep Apnea Detection', fontsize=16, fontweight='bold')
        plt.xlabel('Frequency (Hz)', fontsize=14)
        plt.ylabel('Magnitude (dB)', fontsize=14)
        plt.xlim(0, 1000)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert plot to high-quality base64 string
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=200, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')  # Higher DPI for MP3 analysis
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        plt.close()
        
        print(f"Enhanced MP3-optimized spectrogram generated, base64 length: {len(img_base64)}")
        
        # Enhanced frequency features extraction optimized for MP3 and sleep apnea analysis
        if len(relevant_spectrum) > 0:
            dominant_freq_idx = np.argmax(relevant_spectrum)
            dominant_freq = relevant_freqs[dominant_freq_idx] if dominant_freq_idx < len(relevant_freqs) else 0
            
            # Calculate energy in frequency bands (optimized for MP3 sleep apnea analysis)
            low_mask = relevant_freqs <= 100    # Deep breathing, body movements
            mid_mask = (relevant_freqs > 100) & (relevant_freqs <= 500)  # Primary snoring range
            high_mask = relevant_freqs > 500    # Airway turbulence, apnea events
            
            # Enhanced frequency analysis for MP3 format
            freq_range_energy = {
                "low_freq_energy": float(np.mean(relevant_spectrum[low_mask])) if np.any(low_mask) else 0.0,
                "mid_freq_energy": float(np.mean(relevant_spectrum[mid_mask])) if np.any(mid_mask) else 0.0,
                "high_freq_energy": float(np.mean(relevant_spectrum[high_mask])) if np.any(high_mask) else 0.0,
                "dominant_frequency": float(dominant_freq),
                "spectral_peak": float(np.max(relevant_spectrum)),
                "frequency_spread": float(np.std(relevant_spectrum)),  # Added for MP3 analysis
                "spectral_centroid_enhanced": float(np.sum(relevant_freqs * relevant_spectrum) / np.sum(relevant_spectrum)) if np.sum(relevant_spectrum) > 0 else 0.0
            }
            
            # Add MP3-specific sleep apnea indicators
            total_energy = np.sum(relevant_spectrum)
            if total_energy > 0:
                freq_range_energy.update({
                    "low_freq_ratio": float(np.sum(relevant_spectrum[low_mask]) / total_energy) if np.any(low_mask) else 0.0,
                    "mid_freq_ratio": float(np.sum(relevant_spectrum[mid_mask]) / total_energy) if np.any(mid_mask) else 0.0,
                    "high_freq_ratio": float(np.sum(relevant_spectrum[high_mask]) / total_energy) if np.any(high_mask) else 0.0
                })
        else:
            freq_range_energy = {
                "low_freq_energy": 0.0, "mid_freq_energy": 0.0, "high_freq_energy": 0.0,
                "dominant_frequency": 0.0, "spectral_peak": 0.0, "frequency_spread": 0.0,
                "spectral_centroid_enhanced": 0.0, "low_freq_ratio": 0.0, "mid_freq_ratio": 0.0, "high_freq_ratio": 0.0
            }
        
        result = {
            "image_base64": img_base64,
            "frequency_analysis": freq_range_energy,
            "time_duration": float(times[-1]) if len(times) > 0 else 0.0,
            "mp3_optimized": True  # Flag to indicate enhanced MP3 processing
        }
        
        print(f"Enhanced MP3 spectrogram analysis complete. Dominant frequency: {freq_range_energy['dominant_frequency']:.1f} Hz")
        print(f"Frequency distribution - Low: {freq_range_energy.get('low_freq_ratio', 0)*100:.1f}%, Mid: {freq_range_energy.get('mid_freq_ratio', 0)*100:.1f}%, High: {freq_range_energy.get('high_freq_ratio', 0)*100:.1f}%")
        return result
        
        # Extract key frequency features with better handling
        if len(relevant_spectrum) > 0:
            dominant_freq_idx = np.argmax(relevant_spectrum)
            dominant_freq = relevant_freqs[dominant_freq_idx] if dominant_freq_idx < len(relevant_freqs) else 0
            
            # Calculate energy in frequency bands
            low_mask = relevant_freqs <= 100
            mid_mask = (relevant_freqs > 100) & (relevant_freqs <= 500)
            high_mask = relevant_freqs > 500
            
            freq_range_energy = {
                "low_freq_energy": float(np.mean(relevant_spectrum[low_mask])) if np.any(low_mask) else 0.0,
                "mid_freq_energy": float(np.mean(relevant_spectrum[mid_mask])) if np.any(mid_mask) else 0.0,
                "high_freq_energy": float(np.mean(relevant_spectrum[high_mask])) if np.any(high_mask) else 0.0,
                "dominant_frequency": float(dominant_freq),
                "spectral_peak": float(np.max(relevant_spectrum))
            }
        else:
            freq_range_energy = {
                "low_freq_energy": 0.0,
                "mid_freq_energy": 0.0,
                "high_freq_energy": 0.0,
                "dominant_frequency": 0.0,
                "spectral_peak": 0.0
            }
        
        result = {
            "image_base64": img_base64,
            "frequency_analysis": freq_range_energy,
            "time_duration": float(times[-1]) if len(times) > 0 else 0.0
        }
        
        print(f"Spectrogram generation successful. Dominant frequency: {freq_range_energy['dominant_frequency']:.1f} Hz")
        return result
        
    except Exception as e:
        print(f"Error generating spectrogram: {e}")
        import traceback
        traceback.print_exc()
        return {
            "image_base64": None,
            "frequency_analysis": {},
            "time_duration": 0.0,
            "error": str(e)
        }

def load_model(path: str):
    return joblib.load(path) if os.path.exists(path) else None

def predict_from_file(file_path: str, model):
    feat = extract_features(file_path)
    x = feat["vector"].reshape(1, -1)

    if model is not None:
        try:
            probs = model.predict_proba(x)[0]
            prob = float(probs[1])
            label = "likely_apnea" if prob >= 0.5 else "unlikely"
            return {"probability": prob, "label": label,
                    "features": feat["summary"],
                    "spectrogram": feat["spectrogram"],
                    "note": "Model-based prediction"}
        except Exception:
            pred = int(model.predict(x)[0])
            prob = float(pred)
            label = "likely_apnea" if pred == 1 else "unlikely"
            return {"probability": prob, "label": label,
                    "features": feat["summary"],
                    "spectrogram": feat["spectrogram"],
                    "note": "Prediction (no probability)"}
    else:
        s = feat["summary"]
        score = min(1.0, s["rms"] * 10.0) + max(0.0, (0.2 - s["zcr"]) * 5.0)
        prob = max(0.0, min(1.0, score / 2.0))
        label = "likely_apnea" if prob >= 0.5 else "unlikely"
        return {"probability": prob, "label": label,
                "features": feat["summary"],
                "spectrogram": feat["spectrogram"],
                "note": "Heuristic fallback"}
