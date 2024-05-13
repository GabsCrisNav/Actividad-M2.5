# Import necessary libraries and modules from TensorFlow and other packages
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

# Load the CIFAR-10 dataset and split into training and testing datasets
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Determine the number of unique labels/classes in the training data
num_labels = len(np.unique(y_train))

# Convert labels to one-hot encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Get the size of the images
image_size = x_train.shape[1]

# Normalize pixel values to be between 0 and 1
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Define the first neural network model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=20, batch_size=128)

# Evaluate the model on test set
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print("Test accuracy:", test_accuracy)

# Define a more complex model with additional Convolutional, MaxPooling, and Dropout layers
model_2 = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Flatten(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Compile the model
model_2.compile(optimizer=Adam(),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# Train the model
history = model_2.fit(x_train, y_train, epochs=20, batch_size=128)

# Evaluate the model on test set
test_loss, test_accuracy = model_2.evaluate(x_test, y_test)
print("Test accuracy:", test_accuracy)

# Continue training with a longer duration and validation data
history = model_2.fit(x_train, y_train, epochs=75, validation_data=(x_test, y_test), batch_size=128)

# Display training parameters and keys of the history object
print(history.params)
print(history.history.keys())

# Evaluate the model on the test set again
test_loss, test_accuracy = model_2.evaluate(x_test, y_test)
print("Test accuracy:", test_accuracy)

# Plot the loss and accuracy for both training and validation sets
_, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 5))
axs[0].plot(history.history['loss'], marker='.', linewidth=1)
axs[0].plot(history.history['val_loss'], marker='.', linewidth=1)
axs[0].set_ylabel("Loss")
axs[1].plot(history.history['accuracy'], marker='.', linewidth=1)
axs[1].plot(history.history['val_accuracy'], marker='.', linewidth=1)
axs[1].set_ylabel("Accuracy")
axs[1].set_xlabel("Epoch")
axs[0].legend(["train", "validation"], loc="upper right")
plt.show()

# Repeat the above steps for another similar model setup with different hyperparameters
model_3 = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Flatten(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Compile and train with different batch size and number of epochs
model_3.compile(optimizer=Adam(),
                loss='categorical_crossentropy',
                metrics=['accuracy'])
history = model_3.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test), batch_size=64)

# Evaluate and plot results as before
test_loss, test_accuracy = model_3.evaluate(x_test, y_test)
print("Test accuracy:", test_accuracy)

print(history.params)
print(history.history.keys())

_, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 5))
axs[0].plot(history.history['loss'], marker='.', linewidth=1)
axs[0].plot(history.history['val_loss'], marker='.', linewidth=1)
axs[0].set_ylabel("Loss")
axs[1].plot(history.history['accuracy'], marker='.', linewidth=1)
axs[1].plot(history.history['val_accuracy'], marker='.', linewidth=1)
axs[1].set_ylabel("Accuracy")
axs[1].set_xlabel("Epoch")
axs[0].legend(["train", "validation"], loc="upper right")
plt.show()
