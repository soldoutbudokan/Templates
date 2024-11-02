import os
import zipfile
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import kaggle

# Define the Kaggle dataset
dataset = 'ikarus777/best-artworks-of-all-time'

# Set the download path
download_path = './kaggle_dataset'

# Create the directory if it doesn't exist
os.makedirs(download_path, exist_ok=True)

# Authenticate with Kaggle API
kaggle.api.authenticate()

# Download 'resized.zip' from the dataset
kaggle.api.dataset_download_file(
    dataset,
    file_name='resized.zip',
    path=download_path,
    force=True
)

# Path to the downloaded zip file
resized_zip_path = os.path.join(download_path, 'resized.zip')

# Extract the zip file
with zipfile.ZipFile(resized_zip_path, 'r') as zip_ref:
    zip_ref.extractall('./resized_images')

# Set directories
train_dir = './resized_images/resized'

# Image parameters
batch_size = 32
img_height = 224
img_width = 224

# Data generators for training and validation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2  # 20% of data for validation
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    subset='training',
    class_mode='categorical'
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    subset='validation',
    class_mode='categorical'
)

# Define the Patches layer
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

# Define the PatchEncoder layer
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

# Vision Transformer parameters
num_classes = train_generator.num_classes
input_shape = (img_height, img_width, 3)
patch_size = 16
num_patches = (img_height // patch_size) * (img_width // patch_size)
projection_dim = 64
num_heads = 4
transformer_units = [projection_dim * 2, projection_dim]
transformer_layers = 8
mlp_head_units = [2048, 1024]

# Build the Vision Transformer model
def create_vit_classifier():
    inputs = layers.Input(shape=input_shape)
    # Create patches
    patches = Patches(patch_size)(inputs)
    # Encode patches
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
    # Transformer blocks
    for _ in range(transformer_layers):
        # Layer normalization 1
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Multi-head attention layer
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP
        x3 = layers.Dense(transformer_units[0], activation='gelu')(x3)
        x3 = layers.Dense(transformer_units[1], activation='gelu')(x3)
        # Skip connection 2
        encoded_patches = layers.Add()([x3, x2])
    # Final layers
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # MLP head
    features = layers.Dense(mlp_head_units[0], activation='gelu')(representation)
    features = layers.Dropout(0.5)(features)
    features = layers.Dense(mlp_head_units[1], activation='gelu')(features)
    features = layers.Dropout(0.5)(features)
    # Output layer
    logits = layers.Dense(num_classes)(features)
    # Define the model
    model = keras.Model(inputs=inputs, outputs=logits)
    return model

# Compile the model
model = create_vit_classifier()
learning_rate = 1e-3
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate),
    loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Train the model
epochs = 10
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=epochs
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(validation_generator)
print(f"Test Loss: {test_loss}")
print(f"Validation Accuracy: {test_accuracy}")