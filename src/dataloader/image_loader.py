import os
import cv2
import imghdr
import numpy as np

from config.constant import image_ext
def load_and_preprocess_data(data_dir):
    """
    Loads images from the specified directory, preprocesses them, and creates labels.

    Args:
        data_dir (str): Path to the directory containing subdirectories of image classes.

    Returns:
        tuple: A tuple containing:
            - X (numpy.ndarray): Array of preprocessed image data.
            - y (numpy.ndarray): Array of corresponding labels.
            - class_names (list): List of class names (derived from subdirectory names).
    """
    X = []
    y = []
    class_names = os.listdir(data_dir)
    
    filtered_names = [item for item in class_names if item != '.DS_Store']
    

    for label, class_name in enumerate(filtered_names):
        class_dir = os.path.join(data_dir, class_name)
        for image_file in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_file)
            try:
                img = cv2.imread(image_path)
                if img is not None and imghdr.what(image_path)  in image_ext:
                    img = cv2.resize(img, (128, 128))
                    img = img / 255.0
                    X.append(img)
                    y.append(label)
                else:
                    print(f"Warning: Could not read image: {image_path}")
            except Exception as e:
                print(f"Error processing image: {image_path} - {e}")

    X = np.array(X)
    y = np.array(y)
    return X, y, filtered_names


