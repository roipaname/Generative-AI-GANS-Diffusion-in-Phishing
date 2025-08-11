import os
import tensorflow as tf
import cv2
import imghdr
import numpy as np
data_dir="data/phishingimage"
image_ext=['jpeg','png','jpg','bmg']
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
                if img is not None:
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

X,y,class_names=load_and_preprocess_data(data_dir)
print(y)
"""
for img_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir,img_class)):
        image_path = os.path.join(data_dir, img_class, image)
        try:
            img=cv2.imread(image_path)
            ext=imghdr.what(image_path)
            if ext not in image_ext:
                print("removing image :",image_path)
                os.remove(image_path)
        except Exception as e:
            print("issue with :",image_path)
            os.remove(image_path)
        
import numpy as np
from matplotlib import pyplot as plt

data = tf.keras.utils.image_dataset_from_directory(data_dir)
print(data)

data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()
fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])
plt.tight_layout()
plt.show()

data = data.map(lambda x,y: (x/255, y))
train_size = int(len(data)*.7)
val_size = int(len(data)*.2)
test_size = int(len(data)*.1)"""