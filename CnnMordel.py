import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


print(tf.__version__)
base_dir = './Pokemons/Stellar_Crown/'
batch_size = 32
img_height, img_width = 250, 175


class_names = []
#[folder for folder in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, folder))]
for i,folder in enumerate(os.listdir(base_dir)):
    if i == 10:
        break
    if os.path.isdir(os.path.join(base_dir, folder)):
        class_names.append(folder)

num_classes = len(class_names)
print(f"Detected {num_classes} classes: {class_names}")


train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,# Normalize pixel values to be between 0 and 1
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # Split 20% of data for validation
)

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
        self.kernel_4 = tf.Variable(tf.random.normal([3, 3, 128, 256]), trainable=True)

        # Initialize weights and biases for fully connected layers
        self.fc_weights = tf.Variable(tf.random.normal([45056, num_classes]), trainable=True)
        self.fc_bias = tf.Variable(tf.zeros([num_classes]), trainable=True)

        # Define the optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def forward(self, image):
        conv1 = tf.nn.conv2d(image, self.kernel_1, strides=[1, 1, 1, 1], padding='SAME')
        conv1 = tf.nn.relu(conv1)
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        #dropout1 = tf.nn.dropout(pool1, 0.5)
        pool1 = tf.nn.dropout(pool1, 0.5)
        conv2 = tf.nn.conv2d(pool1, self.kernel_2, strides=[1, 1, 1, 1], padding='SAME')
        conv2 = tf.nn.relu(conv2)
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        pool2 = tf.nn.dropout(pool2, 0.5)
        conv3 = tf.nn.conv2d(pool2, self.kernel_3, strides=[1, 1, 1, 1], padding='SAME')
        conv3 = tf.nn.relu(conv3)
        pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv4 = tf.nn.conv2d(pool3, self.kernel_4, strides=[1, 1, 1, 1], padding='SAME')
        conv4 = tf.nn.relu(conv4)
        pool4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        flattened_output = tf.reshape(pool4, [tf.shape(pool4)[0], -1])
        flattened_output = tf.nn.dropout(flattened_output, 0.5)
        # Fully connected layer
        fc_output = tf.matmul(flattened_output, self.fc_weights) + self.fc_bias
        return fc_output

    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            predictions = self.forward(images)
            loss_fn = tf.keras.losses.CategoricalCrossentropy()
            loss = loss_fn(labels, predictions)
        gradients = tape.gradient(loss, self.get_trainable_variables())
        self.optimizer.apply_gradients(zip(gradients, self.get_trainable_variables()))


    def validate(self, validation_generator):
        print("Validating...")
        total_loss = 0
        total_accuracy = 0
        num_batches = 0

        for i, (images, labels) in enumerate(tqdm(validation_generator, desc="Validation", leave=False)):
            if i > validation_generator.__len__():
                break
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


    def get_trainable_variables(self):
        return [self.kernel_1, self.kernel_2, self.kernel_3, self.kernel_4, self.fc_weights, self.fc_bias]


model = CustomCNN(num_classes, img_height, img_width)

#train_generator.__len__()
print(train_generator.__len__())

epochs = 100
for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    epoch_loss = 0
    epoch_accuracy = 0
    num_batches = 0

    # Wrap train generator with tqdm for a progress bar
    for i, (images, labels) in enumerate(tqdm(train_generator, desc="Training", leave=False)):
        if i > train_generator.__len__():
            break
        model.train_step(images, labels)

    # Validation step at the end of each epoch
    val_loss, val_accuracy = model.validate(validation_generator)


