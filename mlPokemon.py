#createa  clase called: Pokemon

#in this mordel we want to create a cnn mordel with 2 conv2d and 2 maxpooling: so it will looike like:
#input -> 6 x convetion -> 6 x maxpooling, thst was the first layer, now with the second layer we will have: 18 x convetion -> 18 x maxpooling
# and anfter thsi we can use a flatten layer and to ahave a fully connected layer with x neurons and the output layer with the number of classes pokemons this is in toal:

import os
import tensorflow as tf
import numpy as np
import cv2
print(tf.__version__)
#import the keras libary
#how to install? the keras libary this is from the tensorflow libary.: pip install keras

#fetures of the mordel 3x3
egde_filter = tf.constant([
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, 1]
], dtype=tf.float32)

vertical_filter = tf.constant([
    [-1, 0, 2],
    [-1, 0, 2],
    [-1, 0, 2]
], dtype=tf.float32)


horizontal_filter = tf.constant([
    [-1, -1, -1],
    [0, 0, 0],
    [2, 2, 2]
], dtype=tf.float32)


square_filter5x5 = tf.constant([
    [1, -1, 1, -1, 1],
    [-1, 4, -1, 4, -1],
    [1, -1, 1, -1, 1],
    [-1, 4, -1, 4, -1],
    [1, -1, 1, -1, 1]
], dtype=tf.float32)

# 5x5 circle pattern filter (complete your intended pattern as needed)
circle_filter5x5 = tf.constant([
    [1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1]
], dtype=tf.float32)

def display_step(image, step_title):
    image = np.clip(image, 0, 255).astype(np.uint8)
    cv2.imshow(step_title, image)
    print(f"Displaying {step_title} - Press 'q' to continue")
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

class Pokemon:
    def __init__(self, path ):
        self.path = path
        self.input_shape = (350,500,3)
        self.images = [img for img in os.listdir(path) if img.endswith('.jpg')]
        #self.training_images = []  # Use this to store processed images for training

# Define the CnnMordel class to build and control the CNN model
class CnnMordel:
    def __init__(self, num_classes, current_pokemon):
        self.pokemonClasse = current_pokemon
        self.input_shape =  (350,500,3)
        self.num_classes = num_classes
        # Define edge detection filter as a class attribute

    def startml(self):
        # Convert image data to tensors
        images_tensor = tf.convert_to_tensor(self.pokemonClasse.images[0])
        return images_tensor

    def firstConv2D(self, image, padding='SAME'):
        # Apply the edge filter to the first layer for all channels
        edge_filter_3d = tf.tile(egde_filter, [1, 1, 3])  # Apply to each channel
        return tf.nn.conv2d(image, edge_filter_3d, strides=[1, 1, 1, 1], padding=padding)

    def MaxPooling2D(self, image, ksize=2, strides=2, padding='SAME'):
        return tf.nn.max_pool2d(image, ksize=ksize, strides=strides, padding=padding)

if __name__ == "__main__":
    #in the 
    path = "./Pokemons/Stellar_Crown/"
    pokepaths = []
    for paths in os.listdir(path):
        print(paths)
        pokepaths.append(path + paths)

    pokemon = Pokemon(path)
    cnn = CnnMordel( len(pokepaths), pokepaths[0])

