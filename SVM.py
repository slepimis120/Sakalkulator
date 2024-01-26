import os
import librosa
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics

import numpy as np


def read_data():
    path = 'data/audio/training_data'
    audio_files = [f for f in os.listdir(path) if f.endswith('.mp3')]
    features = []
    labels = []

    for f in audio_files:
        audio, sample_rate = librosa.load(os.path.join(path, f))
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfcc = np.mean(mfcc.T, axis=0)
        features.append(mfcc)
        labels.append(f.split('_')[0])

    features = np.array(features)
    labels = np.array(labels)

    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


def train_svm(X_train, y_train):
    model = SVC(kernel='linear', C=1.0)
    model.fit(X_train, y_train)
    return model


def predict_audio():
    X_train, X_test, y_train, y_test = read_data()

    print(f"Number of training samples: {len(X_train)}")
    print(f"Number of testing samples: {len(X_test)}")

    model = train_svm(X_train, y_train)

    y_pred_train = model.predict(X_train)
    print("Train Accuracy:", metrics.accuracy_score(y_train, y_pred_train))

    y_pred_test = model.predict(X_test)
    print("Test Accuracy:", metrics.accuracy_score(y_test, y_pred_test))