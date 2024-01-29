import os

import cv2
import tensorflow as tf
import numpy as np
from keras.preprocessing.image import img_to_array

class_mapping = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9", 10: "-",
                     11: "no", 12: "+", 13: "/", 14: "*"}


def preprocess_frame(frame):
    frame = cv2.resize(frame, (224, 224))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=30)
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    # Apply the sharpening kernel
    frame = cv2.filter2D(frame, -1, kernel)
    frame_array = img_to_array(frame) / 255.0
    frame_array = np.expand_dims(frame_array, axis=0)
    return frame_array


def analyse_video(video_path):
    model = tf.keras.models.load_model('data/model/cnn_model.h5')
    last_gesture = None
    all_gestures = []
    frame_num = 0

    cap = cv2.VideoCapture(video_path)
    cap.set(1, frame_num)

    # Get frames per second (fps) of the video
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # frame_interval = int(fps)

    # frame by frame video analysis
    while True:
        frame_num += 1
        grabbed, frame = cap.read()
        if not grabbed:
            break
        # if frame_num % frame_interval != 0:
        #     continue
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


if __name__ == "__main__":
    directory_path = "data/video/testing_data"
    for f in os.listdir(directory_path):
        print("\n" + f + ":")
        gestures = analyse_video(directory_path + "/" + f)
        result = ""
        for gesture in gestures:
            result += class_mapping[gesture[0]]
        print(result)
