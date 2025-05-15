import os
import pandas as pd
import tensorflow as tf

# Set image size and number of classes
IMG_SIZE = (224, 224)
NUM_CLASSES = 4  # crossed, default, filled, invalid

# Set base directory of the dataset inside the cloned repository
BASE_DIR = "ModelBackEnd/SwiftGrade_Datasets"

# Function to load image paths and label vectors from CSV
def load_csv_labels(csv_path, img_folder):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()  # Normalize headers

    expected_labels = ['crossed', 'default', 'filled', 'invalid']
    for label in expected_labels:
        if label not in df.columns:
            df[label] = 0.0

    df['filepath'] = df['filename'].apply(lambda fname: os.path.join(img_folder, fname))
    labels = df[expected_labels].astype('float32').values
    return df['filepath'].tolist(), labels

# Function to load and preprocess each image
def preprocess_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

# Build tf.data.Dataset from image paths and labels
def build_dataset(csv_file, image_dir, batch_size=32, shuffle=True):
    image_paths, labels = load_csv_labels(csv_file, image_dir)
    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    ds = ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(image_paths))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# Paths to CSVs and image folders
train_csv = os.path.join(BASE_DIR, "train", "_classes.csv")
valid_csv = os.path.join(BASE_DIR, "valid", "_classes.csv")
test_csv  = os.path.join(BASE_DIR, "test", "_classes.csv")

train_img_dir = os.path.join(BASE_DIR, "train")
valid_img_dir = os.path.join(BASE_DIR, "valid")
test_img_dir  = os.path.join(BASE_DIR, "test")

# Create the TensorFlow datasets
train_ds = build_dataset(train_csv, train_img_dir)
valid_ds = build_dataset(valid_csv, valid_img_dir, shuffle=False)
test_ds  = build_dataset(test_csv, test_img_dir, shuffle=False)

print("✅ Dataset pipelines are ready.")
