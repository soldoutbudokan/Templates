import os
import json
import shutil
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Set up Kaggle credentials
kaggle_dir = os.path.expanduser('~/.kaggle')
os.makedirs(kaggle_dir, exist_ok=True)

credentials = {
    "username": "soldoutbudokan",
    "key": "d9ee0aab5179b72286b3360eab0a69e0"
}
with open(os.path.join(kaggle_dir, 'kaggle.json'), 'w') as f:
    json.dump(credentials, f)

os.chmod(os.path.join(kaggle_dir, 'kaggle.json'), 0o600)

import kaggle
kaggle.api.authenticate()

# Download dataset with correct format
print("Downloading dataset...")
owner_slug = "ikarus777"
dataset_slug = "best-artworks-of-all-time"
download_path = './kaggle_dataset'
os.makedirs(download_path, exist_ok=True)

kaggle.api.dataset_download_files(
    dataset=f"{owner_slug}/{dataset_slug}",
    path=download_path,
    unzip=True
)

# Update training directory path
train_dir = os.path.join(download_path, 'resized')

# ------------------------------------------------------
# 1. Read artist data from CSV to get counts per artist
# ------------------------------------------------------
artists_csv = "kaggle_dataset/artists.csv"  # relative path
df_artists = pd.read_csv(artists_csv)

# We assume the CSV has columns: ['name', 'paintings']
# Construct a dict of {artist_name: count_of_paintings}
artist_counts = dict(zip(df_artists['name'], df_artists['paintings']))

# ------------------------------------------------------
# 2. Prepare directory for images
#    Images are under "kaggle_dataset/resized/resized"
# ------------------------------------------------------
data_dir = "kaggle_dataset/images/images"  # adapt if needed

# ------------------------------------------------------
# 3. Build class_weight for all classes
#    We'll do this AFTER the generators are created,
#    because we need to map class_index -> class_name
# ------------------------------------------------------
batch_size = 16
img_height = 224
img_width = 224

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2  # 20% of data for validation
)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    subset='training',
    class_mode='categorical'
)

validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    subset='validation',
    class_mode='categorical'
)

# Check discovered classes
print("Class indices:", train_generator.class_indices)
num_classes = train_generator.num_classes
print("Total classes:", num_classes)

# Construct class_weight based on inverse frequency (or another formula)
# sum(counts) / (num_classes * class_count)
total_images = sum(artist_counts.values())
class_weight = {}

for artist_name, class_idx in train_generator.class_indices.items():
    # Some folders might not match CSV exactly; ensure they match
    # If your folder name is "Albrecht_Dürer" but CSV says "Albrecht_Dürer",
    # fix them or do a matching step. We'll assume they match for now.
    if artist_name in artist_counts:
        count = artist_counts[artist_name]
        weight = total_images / (num_classes * count)
        class_weight[class_idx] = weight
    else:
        # If an artist folder doesn't appear in CSV, assign weight 1
        class_weight[class_idx] = 1.0

print("Class Weights:", class_weight)

# ------------------------------------------------------
# 4. Define Patches and PatchEncoder Layers (unchanged)
# ------------------------------------------------------
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size
        
    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )
        
    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

# ------------------------------------------------------
# 5. Create a Vision Transformer model
# ------------------------------------------------------
input_shape = (img_height, img_width, 3)
patch_size = 16
num_patches = (img_height // patch_size) * (img_width // patch_size)
projection_dim = 64
num_heads = 4
transformer_units = [projection_dim * 2, projection_dim]
transformer_layers = 4  # fewer layers for quicker training
mlp_head_units = [512, 256]

def create_vit_classifier():
    inputs = layers.Input(shape=input_shape)
    # Create patches
    patches = Patches(patch_size)(inputs)
    # Encode patches
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
    
    # Transformer blocks
    for _ in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        x2 = layers.Add()([attention_output, encoded_patches])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = layers.Dense(transformer_units[0], activation='gelu')(x3)
        x3 = layers.Dense(transformer_units[1], activation='gelu')(x3)
        encoded_patches = layers.Add()([x3, x2])

    # Final layers
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)

    for units in mlp_head_units:
        representation = layers.Dense(units, activation='gelu')(representation)
        representation = layers.Dropout(0.5)(representation)
    
    # Output
    logits = layers.Dense(num_classes)(representation)
    model = keras.Model(inputs=inputs, outputs=logits)
    return model

model = create_vit_classifier()
learning_rate = 1e-3

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate),
    loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.summary()

# ------------------------------------------------------
# 6. Train the model using class weights
# ------------------------------------------------------
epochs = 10
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=epochs,
    class_weight=class_weight
)

# ------------------------------------------------------
# 7. Evaluate the model on validation
# ------------------------------------------------------
val_loss, val_accuracy = model.evaluate(validation_generator)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")
