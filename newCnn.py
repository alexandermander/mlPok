import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt
import numpy as np
print(tf.__version__)
# Define paths and parameters
base_dir = './Pokemons/Stellar_Crown/'
batch_size = 16
img_height, img_width = 256, 256

# Get class names from the directory structure
class_names = []
#[folder for folder in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, folder))]
for i,folder in enumerate(os.listdir(base_dir)):
    if os.path.isdir(os.path.join(base_dir, folder)):
        class_names.append(folder)

num_classes = len(class_names)
print(f"Detected {num_classes} classes: {class_names}")

# Enhanced data augmentation for training and validation split
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # Split 20% of data for validation
)

# Load training and validation data
train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    classes=class_names,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

validation_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    classes=class_names,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

class CustomCNN:
    def __init__(self, num_classes, img_height, img_width, learning_rate=0.001):
        self.num_classes = num_classes
        self.img_height = img_height
        self.img_width = img_width
        # Initialize kernels for convolutional layers
        self.kernel_1 = tf.Variable(tf.random.normal([3, 3, 3, 32]), trainable=True)
        self.kernel_2 = tf.Variable(tf.random.normal([3, 3, 32, 64]), trainable=True)
        self.kernel_3 = tf.Variable(tf.random.normal([3, 3, 64, 128]), trainable=True)

        # Initialize weights and biases for fully connected layers
        self.fc_weights = tf.Variable(tf.random.normal([512, num_classes]), trainable=True)
        self.fc_bias = tf.Variable(tf.zeros([num_classes]), trainable=True)

        # Define the optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def conv2d(self, image, kernel, stride=1, padding='same'):
        conv_output = tf.nn.conv2d(image, kernel, strides=[1, stride, stride, 1], padding=padding)
        return tf.nn.relu(conv_output)  # Using ReLU activation

    def max_pooling2d(self, image, pool_size=2, stride=2, padding='same'):
        return tf.nn.max_pool2d(image, ksize=[1, pool_size, pool_size, 1], strides=[1, stride, stride, 1], padding=padding)

    def flatten(self, input_tensor):
        return tf.reshape(input_tensor, [input_tensor.shape[0], -1])

    def dense(self, input_tensor, units, weights, bias):
        return tf.nn.relu(tf.matmul(input_tensor, weights) + bias)

    def forward(self, image):
        # Use persistent kernels in each layer
        x = self.conv2d(image, self.kernel_1)
        x = self.max_pooling2d(x)
        x = tf.nn.dropout(x, rate=0.5)

        x = self.conv2d(x, self.kernel_2)
        x = self.max_pooling2d(x)
        x = tf.nn.dropout(x, rate=0.5)

        x = self.conv2d(x, self.kernel_3)
        x = self.max_pooling2d(x)

        x = self.conv2d(x, self.kernel_4)
        x = self.max_pooling2d(x)

        # Flatten and dense layers
        x = self.flatten(x)
        x = self.dense(x, 512, self.fc_weights, self.fc_bias)
        x = tf.nn.dropout(x, rate=0.5)

        # Output layer
        return tf.nn.softmax(tf.matmul(x, self.fc_weights) + self.fc_bias)

    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            predictions = self.forward(images)
            loss_fn = tf.keras.losses.CategoricalCrossentropy()
            loss = loss_fn(labels, predictions)
        gradients = tape.gradient(loss, self.get_trainable_variables())
        self.optimizer.apply_gradients(zip(gradients, self.get_trainable_variables()))
        
    def get_trainable_variables(self):
        # Retrieve all trainable variables (e.g., weights and biases)
        return [self.kernel_1, self.kernel_2, self.kernel_3, self.kernel_4, self.fc_weights, self.fc_bias]

    def validate(self, validation_generator):
        total_loss = 0
        total_accuracy = 0
        num_batches = 0

        for images, labels in validation_generator:
            # Perform forward pass on validation data
            predictions = self.forward(images)
            
            # Calculate the loss
            loss_fn = tf.keras.losses.CategoricalCrossentropy()
            loss = loss_fn(labels, predictions)
            total_loss += loss.numpy()  # Convert to numpy for aggregation

            # Calculate accuracy
            predicted_labels = np.argmax(predictions, axis=1)
            true_labels = np.argmax(labels, axis=1)
            accuracy = np.mean(predicted_labels == true_labels)
            total_accuracy += accuracy

            num_batches += 1

        # Average the loss and accuracy over all validation batches
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches

        print(f"Validation Loss: {avg_loss:.4f}, Validation Accuracy: {avg_accuracy:.4f}")
        return avg_loss, avg_accuracy


model = CustomCNN(num_classes, img_height, img_width)

# 3. Training Loop
for epoch in range(15):
    for images, labels in train_generator:
        model.train_step(images, labels)

    print(f"Epoch {epoch+1}/{15}")
    model.validate(validation_generator)


