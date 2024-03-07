data_src="/Users/koksziszdave/Downloads/lipread_test"

import os
import cv2
import numpy as np

def load_data(dataset_path):
    """
    Loads video frames from a dataset directory and prepares them for prediction of directory name.

    Args:
        dataset_path (str): Path to the dataset directory containing labeled dictionaries.
        img_height (int, optional): Target image height for resizing (if desired). Defaults to None.
        img_width (int, optional): Target image width for resizing (if desired). Defaults to None.

    Returns:
        tuple: A tuple containing the following elements:
            - X (list): List of NumPy arrays representing video frames for prediction.
            - Y (list): List of true directory names (labels) corresponding to the frames in X.
    """

    X_train , Y_train , X_valid,Y_valid, X_test, Y_test = [], [], [], [], [], []

    for label_dir in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label_dir)
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
                        X_train.append(np.array(frames))
                        Y_train.append(label_dir)
                    elif data_type == 'val':
                        X_valid.append(np.array(frames))
                        Y_valid.append(label_dir)
                    elif data_type == 'test':
                        X_test.append(np.array(frames))
                        Y_test.append(label_dir)


    return X_train , Y_train , X_valid,Y_valid, X_test, Y_test



X_train , Y_train , X_valid,Y_valid, X_test, Y_test = load_data(data_src)

print(X_valid[0].shape, Y_valid[0])