import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import time

# Check for GPU
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
print("Default GPU Device:", tf.test.gpu_device_name())

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize

# Add channel dimension for CNN
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# Build model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)
])

# Compile
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train and time it
print("\n🔧 Starting training...")
start = time.time()
model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test))
end = time.time()

print(f"\n🧪 Test complete in {end - start:.2f} seconds")