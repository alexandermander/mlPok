import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Define paths
base_dir = 'C:/USERS/ALEXA/SANDERSTUFF/ML/FINDCONER/POKEMONS/Stellar_Crown'
batch_size = 32
img_height, img_width = 128, 128  # Adjust based on image size

# Get the list of class names from folders inside Stellar_Crown
class_names = [folder for folder in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, folder))]
num_classes = len(class_names)
print(f"Detected {num_classes} classes: {class_names}")

# Data augmentation for training
train_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Validation data without augmentation
validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Load training images (exclude "test" folders)
train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    classes=class_names,
    class_mode='categorical',
    subset=None,  # Do not split since "test" folders are for validation
    shuffle=True
)

# Load validation images from "test" subdirectories
validation_data_paths = [os.path.join(base_dir, cls, 'test') for cls in class_names]
validation_generator = validation_datagen.flow_from_directory(
    directory=base_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    classes={cls: f'{cls}/test' for cls in class_names},  # Point to 'test' subfolder for each class
    class_mode='categorical',
    shuffle=False
)

# Define the CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(img_height, img_width, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.5),
    
    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.5),
    
    tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
epochs = 10
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=epochs
)

# Save the model
model.save('pokemin.h5')

