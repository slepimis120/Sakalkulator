import os

import joblib
import librosa
import contextlib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def print_metrics(x_val, y_val, best_model):
    # Make predictions on the validation set
    y_val_pred = best_model.predict(x_val)

    # Calculate precision, recall, and F1 score
    precision = precision_score(y_val, y_val_pred, average='weighted', zero_division=1)
    recall = recall_score(y_val, y_val_pred, average='weighted', zero_division=1)
    f1 = f1_score(y_val, y_val_pred, average='weighted')

    # Calculate accuracy
    accuracy = accuracy_score(y_val, y_val_pred)

    # Print the accuracy, precision, recall, and F1 score
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')


def read_data(path):
    audio_files = [f for f in os.listdir(path) if f.endswith('.mp3')]
    features = []
    labels = []

    for f in audio_files:
        with contextlib.redirect_stderr(None):
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


def get_audio_model(path):
    X_train, X_test, y_train, y_test = read_data(path)

    model = train_svm(X_train, y_train)
    joblib.dump(model, './data/model/svm_model.h5')

    print_metrics(X_test, y_test, model)
