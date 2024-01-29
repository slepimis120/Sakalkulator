import os
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.regularizers import l2
from sklearn.metrics import precision_score, recall_score, f1_score


def prepare_data(path):
    image_files = [f for f in os.listdir(path) if f.endswith('.jpg')]
    images = []
    labels = []
    class_mapping = {}

    for image in image_files:
        image_path = os.path.join(path, image)
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        images.append(img_array)
        label = image.split('_')[0]
        if label not in class_mapping:
            class_mapping[label] = len(class_mapping)
        labels.append(class_mapping[label])


    return images, labels, class_mapping


def split_dataset(images, labels, class_mapping):
    x = np.array(images)
    y = to_categorical(labels, num_classes=len(class_mapping))

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
    return x_train, x_val, y_train, y_val


def create_model():
    tf.random.set_seed(42)

    model = models.Sequential()

    model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), kernel_regularizer=l2(0.003), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # multi-dim to one-dim (before dense layers)
    model.add(layers.Flatten())

    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.6))

    model.add(layers.Dense(15, activation='softmax'))

    return model


def train_model(path):
    images, labels, class_mapping = prepare_data(path)
    x_train, x_val, y_train, y_val = split_dataset(images, labels, class_mapping)
    model = create_model()

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # this will return the best validation accuracy, regardless what was in the last epoch
    early_stopping = EarlyStopping(monitor='val_accuracy', mode='max', restore_best_weights=True, patience=400)

    # train the model
    history = model.fit(
        x_train, y_train,
        epochs=400,
        batch_size=64,
        validation_data=(x_val, y_val),
        callbacks=[early_stopping],
        verbose=1
    )
    model.save('./data/model/cnn_model.h5')
    print_evaluation(x_val, y_val, history)
    return model


def print_evaluation(x_val, y_val, history):
    best_model = tf.keras.models.load_model('data/model/cnn_model.h5')

    y_val_pred = best_model.predict(x_val)
    y_val_pred_classes = np.argmax(y_val_pred, axis=1)
    y_val_true_classes = np.argmax(y_val, axis=1)

    precision = precision_score(y_val_true_classes, y_val_pred_classes, average='weighted')
    recall = recall_score(y_val_true_classes, y_val_pred_classes, average='weighted')
    f1 = f1_score(y_val_true_classes, y_val_pred_classes, average='weighted')

    print(f'Best Validation Accuracy: {np.max(history.history["val_accuracy"]):.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')


def get_model(path):
    return train_model(path)


if __name__ == '__main__':
    path = './data/video/training_data'
    get_model(path)