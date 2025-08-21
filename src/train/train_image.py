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

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from src.models.image.cnn_classifier import create_cnn_model
from config.constant import output_dir
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score

data_dir="data/phishingimage"
X,y,class_label=load_and_preprocess_data(data_dir)

y=to_categorical(y)
print(y)
X_train,X_temp,y_train,y_temp=train_test_split(X,y,random_state=42,test_size=0.3,stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)


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

callbacks = [
    ModelCheckpoint('best_cnn_model.keras', save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
]


history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=100,
    validation_data=(X_val, y_val),
    class_weight=class_weight_dict,
    callbacks=callbacks
)

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
import os

os.makedirs(output_dir, exist_ok=True)

# Save the entire model (architecture + weights + training config)
model_path = os.path.join(output_dir, 'cnn_model.h5')
model.save(model_path)

print(f"Model saved to {model_path}")
