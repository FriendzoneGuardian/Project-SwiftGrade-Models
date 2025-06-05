# model_training.py

import os
import tensorflow as tf
from prepare_dataset import build_dataset, train_csv, valid_csv, test_csv, train_img_dir, valid_img_dir, test_img_dir


# Recreate datasets (no shuffle in validation/test)
train_ds = build_dataset(train_csv, train_img_dir)
valid_ds = build_dataset(valid_csv, valid_img_dir, shuffle=False)
test_ds  = build_dataset(test_csv, test_img_dir, shuffle=False)

# Verification step: print batch shapes and first label
for images, labels in train_ds.take(1):
    print("🖼 Image batch shape:", images.shape)
    print("🏷 Label batch shape:", labels.shape)
    print("First label:", labels[0].numpy())

# Define simple CNN model for multi-label classification (4 outputs)
def create_model(input_shape=(224, 224, 3), num_classes=4):
    inputs = tf.keras.Input(shape=input_shape)
    
    # Layer 1: Convolution + MaxPooling at 32 neurons
    x = tf.keras.layers.Conv2D(32, 3, activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)

    # Layer 1: Convolution + MaxPooling at to 64 neurons
    x = tf.keras.layers.Conv2D(64, 3, activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    
    x = tf.keras.layers.Conv2D(128, 3, activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    
    x = tf.keras.layers.Conv2D(256, 3, activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256)(x)
    # x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    # Multi-label sigmoid output for 4 classes
    outputs = tf.keras.layers.Dense(num_classes, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


model = create_model()
model.summary()

# Compile model with binary crossentropy for multi-label classification
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train model
EPOCHS = 20
history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS
)

# Evaluate on test dataset
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

def save_and_convert_to_tflite(model, save_dir="ModelBackEnd/SwiftGrade_Datasets", model_name="swiftgrade_model"):
    os.makedirs(save_dir, exist_ok=True)
    keras_model_path = os.path.join(save_dir, model_name + ".h5")
    tflite_model_path = os.path.join(save_dir, model_name + ".tflite")

    # Save Keras model
    model.save(keras_model_path)
    print(f"✅ Keras model saved to {keras_model_path}")

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save TFLite model
    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)
    print(f"✅ TFLite model exported to {tflite_model_path}")

# --- Ask user ---
answer = input("Do you want to save and export the trained model to .tflite? (Y/N): ").strip().lower()
if answer == 'y':
    save_and_convert_to_tflite(model)
else:
    print("Model save/export skipped.")
