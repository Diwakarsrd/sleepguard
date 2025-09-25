import gradio as gr
import numpy as np
import librosa
import joblib
import matplotlib.pyplot as plt

# Load trained model
model = joblib.load("sleep_apnea_model.pkl")

def predict(file):
    # Extract features
    y, sr = librosa.load(file, sr=None)
    energy = np.array([sum(abs(y[i:i+2048]**2)) for i in range(0, len(y), 512)])
    energy = (energy - np.min(energy)) / (np.max(energy) - np.min(energy) + 1e-6)

    mean_energy = np.mean(energy)
    std_energy = np.std(energy)
    pause_count = np.sum(energy < 0.02)

    features = [mean_energy, std_energy, pause_count]

    # Prediction
    pred = model.predict([features])[0]
    result = "⚠️ Apnea Detected" if pred == 1 else "✅ Normal Breathing"

    # Plot waveform
    fig, ax = plt.subplots()
    ax.plot(y)
    ax.set_title("Audio Waveform")
    ax.set_xlabel("Time (samples)")
    ax.set_ylabel("Amplitude")

    return result, fig

# Gradio UI
demo = gr.Interface(
    fn=predict,
    inputs=gr.Audio(sources=["upload"], type="filepath"),  # only file upload on HF
    outputs=[gr.Textbox(label="Prediction"), gr.Plot(label="Breathing Waveform")],
    title="😴 Sleep Apnea Detection",
    description="Upload your sleep breathing audio (WAV). The model predicts if apnea is detected and shows waveform."
)

if __name__ == "__main__":
    demo.launch()
