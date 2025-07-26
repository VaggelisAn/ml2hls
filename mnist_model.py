import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Rescaling
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# 1. Load and preprocess MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Resize from 28x28 to 14x14 to reduce input size
x_train = tf.image.resize(x_train[..., tf.newaxis], [14, 14]).numpy()
x_test = tf.image.resize(x_test[..., tf.newaxis], [14, 14]).numpy()

# Normalize
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# One-hot labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 2. Define lightweight CNN
model = Sequential([
    Conv2D(8, kernel_size=3, activation='relu', input_shape=(14, 14, 1)),  # fewer filters
    MaxPooling2D(pool_size=2),
    Flatten(),
    Dense(16, activation='relu'),  # reduced dense layer
    Dense(10, activation='softmax')
])

# 3. Compile and train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, batch_size=16, validation_data=(x_test, y_test))

# 4. Save the model
model.save("model.h5")
print("✅ Lightweight model saved as 'model.h5'")