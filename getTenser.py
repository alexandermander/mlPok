import tensorflow as tf
from PIL import Image
import cv2
import numpy as np

# Load and resize the image
img = Image.open('./IMG_20241101_124547891.jpg')
img = img.resize((256, 256))
img = np.array(img)

# Convert the image to a TensorFlow tensor
color_image = tf.constant(img, dtype=tf.float32)
color_image = tf.reshape(color_image, [1, 256, 256, 3])  # (batch_size, height, width, channels)

# Function to display each image step
def display_step(image, step_title):
    image = np.clip(image, 0, 255).astype(np.uint8)
    cv2.imshow(step_title, image)
    print(f"Displaying {step_title} - Press 'q' to continue")
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

# Step 1: Initial Convolution - Edge Detection
edge_filter = tf.constant([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=tf.float32)
edge_filter = tf.reshape(edge_filter, [3, 3, 1, 1])

# Apply edge detection to each color channel
def apply_filter(image, filter, channel):
    return tf.nn.conv2d(image[:, :, :, channel:channel+1], filter, strides=[1, 1, 1, 1], padding='SAME')

# Edge detection results
edge_r = apply_filter(color_image, edge_filter, 0)
edge_g = apply_filter(color_image, edge_filter, 1)
edge_b = apply_filter(color_image, edge_filter, 2)

# Convert tensors to numpy for display
display_step(edge_r.numpy().squeeze(), "Edge Detection - Red Channel")
display_step(edge_g.numpy().squeeze(), "Edge Detection - Green Channel")
display_step(edge_b.numpy().squeeze(), "Edge Detection - Blue Channel")

# Step 2: Max Pooling after Edge Detection
pooled_r = tf.nn.max_pool2d(edge_r, ksize=2, strides=2, padding='SAME')
pooled_g = tf.nn.max_pool2d(edge_g, ksize=2, strides=2, padding='SAME')
pooled_b = tf.nn.max_pool2d(edge_b, ksize=2, strides=2, padding='SAME')

# Display pooled results
display_step(pooled_r.numpy().squeeze(), "Max Pooling - Red Channel")
display_step(pooled_g.numpy().squeeze(), "Max Pooling - Green Channel")
display_step(pooled_b.numpy().squeeze(), "Max Pooling - Blue Channel")

# Step 3: Shape Detection Convolution (e.g., Square Detection)
square_filter = tf.constant([
    [1, -1, 1],
    [-1, 4, -1],
    [1, -1, 1]
], dtype=tf.float32)
square_filter = tf.reshape(square_filter, [3, 3, 1, 1])

# Apply square detection
square_r = apply_filter(pooled_r, square_filter, 0)
square_g = apply_filter(pooled_g, square_filter, 0)
square_b = apply_filter(pooled_b, square_filter, 0)

# Display shape detection results
display_step(square_r.numpy().squeeze(), "Square Detection - Red Channel")
display_step(square_g.numpy().squeeze(), "Square Detection - Green Channel")
display_step(square_b.numpy().squeeze(), "Square Detection - Blue Channel")

# Step 4: Second Max Pooling
pooled_square_r = tf.nn.max_pool2d(square_r, ksize=2, strides=2, padding='SAME')
pooled_square_g = tf.nn.max_pool2d(square_g, ksize=2, strides=2, padding='SAME')
pooled_square_b = tf.nn.max_pool2d(square_b, ksize=2, strides=2, padding='SAME')

# Display second pooling results
display_step(pooled_square_r.numpy().squeeze(), "Second Max Pooling - Red Channel")
display_step(pooled_square_g.numpy().squeeze(), "Second Max Pooling - Green Channel")
display_step(pooled_square_b.numpy().squeeze(), "Second Max Pooling - Blue Channel")

# Step 5: Global Average Pooling
gap_square_r = tf.reduce_mean(pooled_square_r).numpy()
gap_square_g = tf.reduce_mean(pooled_square_g).numpy()
gap_square_b = tf.reduce_mean(pooled_square_b).numpy()
print(f"Global Average Pooling - Red Channel: {gap_square_r}")
print(f"Global Average Pooling - Green Channel: {gap_square_g}")
print(f"Global Average Pooling - Blue Channel: {gap_square_b}")

# Step 6: Flattening and Dense Layer
flattened_features = np.array([gap_square_r, gap_square_g, gap_square_b]).reshape(1, -1)
print("Flattened Features:", flattened_features)

# Dummy Fully Connected Layer and Softmax (for illustration purposes)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(3,)),
    tf.keras.layers.Dense(2, activation='softmax')
])

output = model(flattened_features)
print("Softmax Output:", output.numpy())

