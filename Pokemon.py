import os
import tensorflow as tf
import numpy as np
import cv2

class Pokemon:
    def __init__(self, path, index):
        #the name is teh last folder in the path
        self.path = path
        self.name = path.split('/')[-1]
        self.label = index
        self.image_in_batch = []
        self.validation_images = self.get_images_for_validation()
        self.images = self.set_images()

    def get_images_for_validation(self):
        path = os.path.join(self.path, 'test')
        images = []
        for img in os.listdir(self.path):
            if img.endswith('.jpg'):
                img_path = os.path.join(self.path, img)
                image = tf.io.read_file(img_path)
                image = tf.image.decode_jpeg(image, channels=3)
                image = tf.image.resize(image, (500, 350))
                image = np.expand_dims(image, axis=0).astype(np.float32)
                images.append(image)
        return images

    def set_images(self):
        images = []
        for img in os.listdir(self.path):
            if img.endswith('.jpg'):
                img_path = os.path.join(self.path, img)
                image = tf.io.read_file(img_path)
                image = tf.image.decode_jpeg(image, channels=3)
                image = tf.image.resize(image, (500, 350))
                image = np.expand_dims(image, axis=0).astype(np.float32)
                images.append(image)
        return images

    def set_batch_images(self, batch_size=3):
        for i in range(batch_size):
            if len(self.images) == 0:
                break
            image = self.images.pop()
            self.image_in_batch.append(image)

    def clear_batch_images(self):
        self.image_in_batch = []


if __name__ == '__main__':
    path = './Pokemons/Stellar_Crown/'
    pokemons = []
    for i, pokemon in enumerate(os.listdir(path)):
        pokemon_path = os.path.join(path, pokemon)
        pokemons.append(Pokemon(pokemon_path, i))

    pokemons = pokemons[:3]

    #add images to the pokemons batch_images
    for pokemon in pokemons:
        #print the size of the list of images
        print("the length of the images in the iimges list is: ", len(pokemon.images))
        pokemon.set_batch_images()
        print("the iamgesare set")
        print("the length of the images in the iimges list is: ", len(pokemon.images))

    for pokemon in pokemons:
        print("the length of the images in the batch is: ", len(pokemon.image_in_batch))

    for pokemon in pokemons:
        pokemon.clear_batch_images()
        print("the length of the images in the batch is: ", len(pokemon.image_in_batch))
        pokemon.set_batch_images()
        print("the iamgesare set")
        print("the length of the images in the iimges list is: ", len(pokemon.images))




