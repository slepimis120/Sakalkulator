import csv
import os

import cv2
import tensorflow as tf
import numpy as np
from keras.preprocessing.image import img_to_array

from helper.common_functions import load_csv, calculate_accuracy

class_mapping = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9", 10: "-",
                     11: "n", 12: "+", 13: "/", 14: "*"}


def preprocess_frame(frame):
    frame = cv2.resize(frame, (224, 224))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=30)
    frame_array = img_to_array(frame) / 255.0
    frame_array = np.expand_dims(frame_array, axis=0)
    return frame_array


def process_video(video_path, frame_interval_seconds):
    model = tf.keras.models.load_model('data/model/cnn_model.h5')
    last_gesture = None
    all_gestures = []
    frame_num = 0

    cap = cv2.VideoCapture(video_path)
    cap.set(1, frame_num)

    # Get frames per second (fps) of the video
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(frame_rate * frame_interval_seconds)

    # frame by frame video analysis
    while True:
        frame_num += 1
        grabbed, frame = cap.read()

        if not grabbed:
            break
        if frame_rate <= 0:
            print("Warning: Unable to determine video frame rate. Defaulting to 30 frames per second.")
            frame_rate = 30
        if frame_num % frame_interval != 0:
            continue

        frame = preprocess_frame(frame)
        pred_classes = model.predict(frame, verbose=0)
        max_prob = np.max(pred_classes, axis=1)
        if max_prob > 0.9:
            predicted_gesture = np.argmax(pred_classes, axis=1)
            if predicted_gesture != last_gesture:
                last_gesture = predicted_gesture
                all_gestures.append(last_gesture)
    cap.release()
    return all_gestures


def process_result(input_string):
    i = 0
    prev_char = ''
    result_string = ""

    if not input_string or input_string == '':
        return ""

    while i < len(input_string):
        char = input_string[i]
        i += 1
        # number after number is not allowed. There must be an operation or "n" between numbers
        if char.isdigit() and prev_char.isdigit():
            prev_char = char
            continue
        # after operation/n must go a number
        if not char.isdigit() and not prev_char.isdigit():
            prev_char = char
            continue
        # first thing in a result must be a number
        if result_string == "" and not char.isdigit():
            prev_char = char
            continue
        if char != "n":
            result_string += char
        prev_char = char
    return result_string


def test_video(directory_path="data/video/testing_data/"):
    csv_file = "res.csv"
    csv_results = load_csv(directory_path + csv_file)
    results = {}

    for f in os.listdir(directory_path):
        if f == csv_file:
            continue
        print("\n" + f + ":")

        gestures = process_video(directory_path + f, 0.5)
        result = ""
        for gesture in gestures:
            result += class_mapping[gesture[0]]
        result = process_result(result)
        results[f] = result
        print(result)

    accuracy = calculate_accuracy(results, csv_results)
    print("\nAccuracy: " + str(accuracy))


if __name__ == "__main__":
    test_video("data/video/testing_data/")
