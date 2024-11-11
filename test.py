import tensorflow as tf
import numpy as np
import os

# Define a custom CNN model without tf.keras
class CustomCNN: 
    def __init__(self, input_shape, num_classes):
        self.num_classes = num_classes
        # Initialize weights and biases for layers
        self.weights = {
            'conv1': tf.Variable(tf.random.normal([3, 3, 3, 32]), name='w_conv1'),
            'conv2': tf.Variable(tf.random.normal([3, 3, 32, 64]), name='w_conv2'),
            'conv3': tf.Variable(tf.random.normal([3, 3, 64, 128]), name='w_conv3'),
            'fc1': tf.Variable(tf.random.normal([16*16*128, 512]), name='w_fc1'),
            'out': tf.Variable(tf.random.normal([512, num_classes]), name='w_out')
        }
        self.biases = {
            'conv1': tf.Variable(tf.random.normal([32]), name='b_conv1'),
            'conv2': tf.Variable(tf.random.normal([64]), name='b_conv2'),
            'conv3': tf.Variable(tf.random.normal([128]), name='b_conv3'),
            'fc1': tf.Variable(tf.random.normal([512]), name='b_fc1'),
            'out': tf.Variable(tf.random.normal([num_classes]), name='b_out')
        }


    def forward(self, x):
        # First Convolutional Layer
        x = tf.nn.conv2d(x, self.weights['conv1'], strides=[1, 1, 1, 1], padding='SAME')
        x = tf.nn.bias_add(x, self.biases['conv1'])
        x = tf.nn.relu(x)
        x = tf.nn.max_pool2d(x, ksize=2, strides=2, padding='SAME')
        
        # Second Convolutional Layer
        x = tf.nn.conv2d(x, self.weights['conv2'], strides=[1, 1, 1, 1], padding='SAME')
        x = tf.nn.bias_add(x, self.biases['conv2'])
        x = tf.nn.relu(x)
        x = tf.nn.max_pool2d(x, ksize=2, strides=2, padding='SAME')
        
        # Third Convolutional Layer
        x = tf.nn.conv2d(x, self.weights['conv3'], strides=[1, 1, 1, 1], padding='SAME')
        x = tf.nn.bias_add(x, self.biases['conv3'])
        x = tf.nn.relu(x)
        x = tf.nn.max_pool2d(x, ksize=2, strides=2, padding='SAME')
        
        # Flatten and Fully Connected Layer
        x = tf.reshape(x, [-1, self.weights['fc1'].shape[0]])
        x = tf.matmul(x, self.weights['fc1']) + self.biases['fc1']
        x = tf.nn.relu(x)
        
        # Output Layer
        logits = tf.matmul(x, self.weights['out']) + self.biases['out']
        return logits  # No softmax here; use it in the loss computation

# Define the Pokemon class to handle image and label loading
class Pokemon:
    def __init__(self, path, index):
        self.path = path
        self.name = path.split('/')[-1]
        self.label = index
        self.image_in_batch = []
        self.validation_images = self.get_images_for_validation()
        self.images = self.set_images()

    def get_images_for_validation(self):
        path = os.path.join(self.path, 'test')
        images = []
        for img in os.listdir(path):
            if img.endswith('.jpg'):
                img_path = os.path.join(path, img)
                image = tf.io.read_file(img_path)
                image = tf.image.decode_jpeg(image, channels=3)
                image = tf.image.resize(image, (256, 256))
                images.append(image)
        return images

    def set_images(self):
        images = []
        for img in os.listdir(self.path):
            if img.endswith('.jpg'):
                img_path = os.path.join(self.path, img)
                image = tf.io.read_file(img_path)
                image = tf.image.decode_jpeg(image, channels=3)
                image = tf.image.resize(image, (256, 256))
                images.append(image)
        return images

    def set_batch_images(self, batch_size=3):
        self.image_in_batch = [self.images.pop() for _ in range(batch_size) if self.images]

# Path to dataset
path = './Pokemons/Stellar_Crown/'

# Load Pokémon data and initialize model
pokemons = []
for i, pokemon_folder in enumerate(os.listdir(path)):
    if i == 50:
        break
    pokemons.append(Pokemon(os.path.join(path, pokemon_folder), i))

# Filter the loaded pokemons for only a subset if necessary
pokemons = pokemons[:3]

# Initialize CustomCNN model
cnn = CustomCNN(input_shape=(256, 256, 3), num_classes=len(pokemons))
optimizer = tf.optimizers.Adam(learning_rate=0.001)

# Define custom training step
def compute_loss(logits, labels):
    # Log the shapes to debug
    print(f"Logits shape: {logits.shape}")
    print(f"Labels shape: {labels.shape}")
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

@tf.function
def train_step(model, images, labels):
    with tf.GradientTape() as tape:
        predictions = model.forward(images)
        loss = compute_loss(predictions, labels)
    gradients = tape.gradient(loss, model.weights.values())
    optimizer.apply_gradients(zip(gradients, model.weights.values()))
    return loss

# Training loop
epochs = 10
batch_size = 3
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    epoch_loss = 0
    for pokemon in pokemons:
        pokemon.set_batch_images(batch_size)
        
        # Prepare images and labels batch
        images_batch = tf.stack(pokemon.image_in_batch)
        labels_batch = tf.one_hot([pokemon.label] * images_batch.shape[0], depth=len(pokemons))
        
        # Run training step
        loss = train_step(cnn, images_batch, labels_batch)
        epoch_loss += loss.numpy()
        pokemon.clear_batch_images()
        
    print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")

# Optional: Validation Loop (using validation_images in Pokemon instances)
def evaluate(model, pokemons):
    total_accuracy = 0
    for pokemon in pokemons:
        images_batch = tf.stack(pokemon.validation_images)
        labels_batch = tf.one_hot([pokemon.label] * len(pokemon.validation_images), depth=len(pokemons))
        
        predictions = model.forward(images_batch)
        correct_predictions = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels_batch, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        total_accuracy += accuracy.numpy()
        
    return total_accuracy / len(pokemons)

# Run evaluation
validation_accuracy = evaluate(cnn, pokemons)
print(f"Validation Accuracy: {validation_accuracy:.4f}")

