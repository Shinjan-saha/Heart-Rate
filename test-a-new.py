import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

audio_directory = './training-a/'
output_directory = './output_plots/'  

os.makedirs(output_directory, exist_ok=True)

audio_files = [f for f in os.listdir(audio_directory) if f.endswith('.wav')]

#  PDF file to store all plots
pdf_filename = os.path.join(output_directory, 'all_plots.pdf')
pdf_pages = []

for audio_file in audio_files:
    audio_path = os.path.join(audio_directory, audio_file)
    y, sr = librosa.load(audio_path)

    # Waveform
    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(y, sr=sr)
    plt.title(f'Waveform - {audio_file}')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    waveform_save_path = os.path.join(output_directory, f'waveform_{audio_file.replace(".wav", ".png")}')
    plt.savefig(waveform_save_path)
    plt.close()

    # Spectrogram
    plt.figure(figsize=(14, 5))
    D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.title(f'Spectrogram - {audio_file}')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar(format='%+2.0f dB')
    spectrogram_save_path = os.path.join(output_directory, f'spectrogram_{audio_file.replace(".wav", ".png")}')
    plt.savefig(spectrogram_save_path)
    plt.close()

    
    pdf_pages.extend([waveform_save_path, spectrogram_save_path])


try:
    import img2pdf

    with open(pdf_filename, "wb") as pdf_file:
        pdf_file.write(img2pdf.convert(pdf_pages))
    print(f'All plots saved in {pdf_filename}')
except ImportError:
    print("img2pdf module is not installed. You can install it with 'pip install img2pdf'")
