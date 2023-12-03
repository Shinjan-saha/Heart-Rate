import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


audio_directory = './training-a/'


audio_files = [f for f in os.listdir(audio_directory) if f.endswith('.wav')]


for audio_file in audio_files:
   
    audio_path = os.path.join(audio_directory, audio_file)

   
    y, sr = librosa.load(audio_path)

    # Waveform
    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(y, sr=sr)
    plt.title(f'Waveform - {audio_file}')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()

    # Spectrogram
    plt.figure(figsize=(14, 5))
    D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.title(f'Spectrogram - {audio_file}')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar(format='%+2.0f dB')
    plt.show()
