import sounddevice as sd
import numpy as np
import keyboard
from pydub import AudioSegment


# Parameters for audio recording
fs = 44100  # Sampling frequency
duration = 30  # Maximum duration to record (in seconds)
recording = False
audio_data = []


def record_voice(device_index=0):
    global recording, audio_data
    print("Unesite R da biste započeli snimanje zvuka i X da biste završili")

    while True:
        if keyboard.is_pressed('R') and not recording:
            print("Snimanje počelo...")
            recording = True
            audio_data = []  # Clear previous data
            sd.stop()  # Stop any ongoing playback
            sd.default.device = device_index
            sd.default.channels = 2
            sd.default.samplerate = fs
            sd.default.dtype = np.int16
            sd_stream = sd.InputStream(callback=callback)
            sd_stream.start()

        elif keyboard.is_pressed('x') and recording:
            print("Snimanje zaustavljeno.")
            recording = False
            sd_stream.stop()
            break

    return np.concatenate(audio_data)


def callback(indata, frames, time, status):
    if status:
        print(status, flush=True)

    audio_data.append(indata.copy())


def save_recording(captured_audio):
    # Convert the NumPy array to an AudioSegment
    audio_segment = AudioSegment(
        captured_audio.tobytes(),
        frame_rate=fs,
        sample_width=captured_audio.dtype.itemsize,
        channels=2
    )

    # Save the AudioSegment as an MP3 file
    audio_segment.export('./data/captured_audio.wav', format='wav')

    print("Audio saved as captured_audio.wav")