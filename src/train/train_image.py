from src.dataloader.image_loader import load_and_preprocess_data
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.utils import class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
data_dir="data/phishingimage"
X,y,class_label=load_and_preprocess_data(data_dir)

y=to_categorical(y)
print(y)
X_train,X_temp,y_train,y_temp=train_test_split(X,y,random_state=42,test_size=0.3,stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

def create_cnn_model(input_shape, num_classes):
    """
    Defines the convolutional neural network (CNN) model architecture.

    Args:
        input_shape (tuple): Shape of the input images (height, width, channels).
        num_classes (int): Number of output classes.

    Returns:
        tf.keras.models.Sequential: The compiled CNN model.
    """
    model = models.Sequential()

    # Convolutional block 1
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))

    # Convolutional block 2
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Convolutional block 3
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Convolutional block 4
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Convolutional block 5
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Convolutional block 6
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))

    # Convolutional block 7
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))

    # Convolutional block 8
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))

    # Convolutional block 9
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))

    # Convolutional block 10
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))

    # Flatten the output from the convolutional layers
    model.add(layers.Flatten())
    # Fully connected layer 1
    model.add(layers.Dense(512, activation='relu'))
    # Output layer
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model
input_shape = X_train.shape[1:]
num_classes=len(class_label)
model=create_cnn_model(input_shape,num_classes)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)
datagen.fit(X_train)

# Calculate class weights to handle imbalanced datasets
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(np.argmax(y_train, axis=1)), y=np.argmax(y_train, axis=1))
# Convert class weights to a dictionary for use during training
class_weight_dict = dict(enumerate(class_weights))

history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    epochs=40,
                    validation_data=(X_val, y_val),
                    class_weight=class_weight_dict)

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# 5. Results Visualization
# Plot training and validation accuracy over epochs
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

y_pred = model.predict(X_test)
# Convert the predicted probabilities to class labels
y_pred_classes = np.argmax(y_pred, axis=1)
# Get the true class labels from the one-hot encoded vectors
y_true_classes = np.argmax(y_test, axis=1)

# Calculate Precision, Recall, and F1-Score on the test set
precision = precision_score(y_true_classes, y_pred_classes, average='weighted')
recall = recall_score(y_true_classes, y_pred_classes, average='weighted')
f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')

print(f"Test Precision: {precision}")
print(f"Test Recall: {recall}")
print(f"Test F1-Score: {f1}")