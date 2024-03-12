import tensorflow
data_src="/Users/koksziszdave/Downloads/lipread_test"

import os
import cv2
import numpy as np


def togreyscale(aframes):

    gf = np.zeros((aframes.shape[0], aframes.shape[1], aframes.shape[2],1))
    for i in range(aframes.shape[0]):
        gf[i] = cv2.cvtColor(aframes[i], cv2.COLOR_BGR2GRAY).reshape(aframes.shape[1], aframes.shape[2], 1)

    return gf


def load_data(dataset_path):


    X_train , Y_train , X_valid,Y_valid, X_test, Y_test = [], [], [], [], [], []
    labels = []
    i=0
    for label_dir in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label_dir)
        labels.append(label_dir)
        i=i+1
        if not os.path.isdir(label_path):
            continue
        for data_type in ['train', 'val', 'test']:  # Assuming all data in one directory
            data_type_path = os.path.join(label_path, data_type)

            for filename in os.listdir(data_type_path):
                if filename.endswith('.mp4'):
                    video_path = os.path.join(data_type_path, filename)

                    # Use opencv to read video frames efficiently
                    cap = cv2.VideoCapture(video_path)
                    frames = []
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frames.append(frame)
                    cap.release()
                    if data_type == 'train':
                        aframes = np.array(frames)
                        split = int((aframes.shape[0]-11)/2)
                        if (aframes.shape[0] % 2 == 1):
                            aframes = aframes[split:(aframes.shape[0] - split)]
                        else:
                            aframes = aframes[split:(aframes.shape[0] - split - 1)]
                        gframes = togreyscale(aframes)
                        X_train.append(gframes)
                        Y_train.append(i-2)
                    elif data_type == 'val':
                        aframes = np.array(frames)
                        split = int((aframes.shape[0] - 11) / 2)
                        if (aframes.shape[0] % 2 == 1):
                            aframes = aframes[split:(aframes.shape[0] - split)]
                        else:
                            aframes = aframes[split:(aframes.shape[0] - split - 1)]
                        gframes = togreyscale(aframes)
                        X_valid.append(gframes)
                        Y_valid.append(i-2)
                    elif data_type == 'test':
                        aframes = np.array(frames)
                        split = int((aframes.shape[0] - 11) / 2)
                        if (aframes.shape[0] % 2 == 1):
                            aframes = aframes[split:(aframes.shape[0] - split)]
                        else:
                            aframes = aframes[split:(aframes.shape[0] - split - 1)]

                        gframes = togreyscale(aframes)
                        X_test.append(gframes)
                        Y_test.append(i-2)

    Y_train = np.array(Y_train).reshape(-1, 1)
    Y_valid = np.array(Y_valid).reshape(-1, 1)
    Y_test = np.array(Y_test).reshape(-1, 1)

    nb_classes = len(np.unique(Y_train))
    Y_train = tensorflow.keras.utils.to_categorical(Y_train, nb_classes)
    Y_valid = tensorflow.keras.utils.to_categorical(Y_valid, nb_classes)
    Y_test = tensorflow.keras.utils.to_categorical(Y_test, nb_classes)


    return X_train , Y_train , X_valid,Y_valid, X_test, Y_test



X_train , Y_train , X_valid,Y_valid, X_test, Y_test = load_data(data_src)

print(X_valid[0].shape, Y_valid[0])