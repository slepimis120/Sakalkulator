import csv
import os

import cv2
import joblib
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
import soundfile as sf
from pydub import AudioSegment
from pydub.silence import split_on_silence


class_mapping = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9", 10: "-",
                     11: "n", 12: "+", 13: "/", 14: "*"}


def delete_files_in_folder(folder_path):
    try:
        files = os.listdir(folder_path)

        for file in files:
            file_path = os.path.join(folder_path, file)
            if file_path == os.path.join(folder_path, "example.txt"):
                continue
            if os.path.isfile(file_path):
                os.remove(file_path)
    except Exception as e:
        print(f"An error occurred: {e}")


def load_csv(csv_path="data/audio/testing_data/res.csv"):
    data = {}
    csv_file = open(csv_path, 'r', encoding='utf-8')
    csv_reader = csv.reader(csv_file)

    next(csv_reader)  # preskoci zaglavlje
    for row in csv_reader:
        data[row[0]] = row[1]

    return data


def extract_features(temp_dir, f):
    audio, sample_rate = librosa.load(os.path.join(temp_dir, f))
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfcc = np.mean(mfcc.T, axis=0)
    return mfcc


def predict_segments(model, temp_dir):
    # one audio is separated into segments divided by silence
    audio_files = [f for f in os.listdir(temp_dir) if f != "example.txt"]
    if not audio_files:
        return []
    features = []
    scaler = StandardScaler()
    for f in audio_files:
        mfcc = extract_features(temp_dir, f)
        features.append(mfcc)
    features = np.array(features)
    features = scaler.fit_transform(features)
    predictions = model.predict(features)
    return predictions


def process_audio(dir_path, f):
    model = joblib.load('data/model/svm_model.h5')

    audio = AudioSegment.from_file(dir_path + f)

    intervals = split_on_silence(audio,
                              # must be silent for at least half a second
                              min_silence_len=500,
                              # consider it silent if quieter than -16 dBFS
                              silence_thresh=-50)

    temp_dir = "data/temp/"
    for i, segment in enumerate(intervals):
        segment.export(f"{temp_dir}{f[0:-4]}_{i + 1}.mp3", format="mp3")
        segment.export(f"data/idk/{f[0:-4]}_{i + 1}.mp3", format="mp3")
    predictions = predict_segments(model, temp_dir)
    print(predictions)
    delete_files_in_folder(temp_dir)


def test_audio(directory_path="data/audio/testing_data/"):
    csv_file = "res.csv"
    csv_results = load_csv(directory_path + csv_file)
    results = {}
    for f in os.listdir(directory_path):
        if f == csv_file:
            continue
        print("\n" + f + ":")
        segments = process_audio(directory_path, f)
    #     result = ""
    #     for gesture in gestures:
    #         result += class_mapping[gesture[0]]
    #     result = process_result(result)
    #     results[f] = result
    #     print(result)
    # accuracy = calculate_accuracy(results, csv_results)
    # print("\nAccuracy: " + str(accuracy))


if __name__ == "__main__":
    test_audio("data/audio/testing_data/")

