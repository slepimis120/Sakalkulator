![Sakalkulator](https://i.imgur.com/CABPQWh.png)

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

# Šakalkulator
Šakalkulator project for "Soft Computing"

## Project goal
"Šakalkulator", developed as our 4th year college project, is a program used for class "Soft Computing" in 2024. 

The goal of this project is to enable a simple calculator that can perform basic arithmetic operations (+, -, *, /), with any numbers, using hand gestures and audio recordings.

All of the data used for the program was made by the 3 students who created the project.

## Installing / Getting started

To install this program, you're required to do two things.

1. CD into the folder where requirements.txt can be found and do:
   
```shell
pip install -r requirements.txt
```

2. Run the script itself

```shell
python main.py
```

## Database

Database consists of 2 parts: audio and video data. Audio data is stored in .mp3 format, while video data in .mp4 format.

Both audio and video data consist of 2 separated folders: training and testing, 80% and 20% respectively.

## Training models

Mainly two types of models are trained: CNN for image classification and SVM for audio classification. 

**CNN**, used for image classification, is trained on a dataset of hand gesture images. Each gesture represents a number or an arithmetic operation. The training of the gestural part of the application was performed by a fully connected neural network, which consists of 5 convolutional, Flatten, fully connected Dense and output Dense layers.

**SVM**, used for audio classification, is trained with linear kernel, and MFCC was used for feature extraction.

Testing the performance of the calculator on videos was done by taking frame by frame and making a prediction for each one.

Testing the operation of the calculator on audio recordings was done by dividing each one into segments where the division was made according to the absence of sound and for each segment a prediction was made after that. The exact results are placed in a csv file.

## Evaluation

Both models are evaluated using various metrics such as accuracy, precision, recall, and F1 score.

## Licence 

PlaylistGenie is available under the GNU GPLv3 license.
