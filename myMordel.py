import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from Pokemon import Pokemon
#from Pokemon import Pokemon

#def display_step(image):
#    cv2.imshow("a pokeon", image)
#    print(f"Displaying  - Press 'q' to continue")
#    while True:
#        if cv2.waitKey(1) & 0xFF == ord('q'):
#            break
#    cv2.destroyAllWindows()

class CnnMordel:
    def __init__(self,input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = 0.001
        self.fc_weights = tf.Variable(tf.random.normal([4224, self.num_classes]))
        self.fc_bias = tf.Variable(tf.random.normal([self.num_classes]))
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)


    def firstConv2D(self, image, kernels, index, padding='SAME'):
        channels = image.shape[-1]
        conv_outputs = [
            tf.nn.conv2d(image[..., i:i+1], kernel, strides=[1, 1, 1, 1], padding=padding)
            for i, kernel in enumerate(kernels[:index]) for i in range(channels)
        ]
        return tf.concat(conv_outputs, axis=-1)

    def convImage(self, image, kernel ):
        #kerunel is a lsit of kernels
        #print("the shape of the image is: ", image.shape)
        #print("the shape of the kernel is: ", kernel.shape)
        convlution = tf.nn.conv2d(image, kernel, strides=[1, 1, 1, 1], padding='SAME')
        #display_step(np.squeeze(convlution[:,:,:,0:1]))
        #display_step(np.squeeze(convlution[:,:,:,1:2]))
        #display_step(np.squeeze(convlution[:,:,:,2:3]))
        return convlution

    def forward2(self, image, kernels):
    # use the random kernels: 32, 64, 128
        # Apply the first convolutional layerc 
        convImage = self.convImage(image, kernels[0])
        convImage = tf.nn.relu(convImage)
        convImage = tf.nn.max_pool2d(convImage, ksize=2, strides=2, padding='SAME')
        # Apply the second convolutional layer

        convImage = self.convImage(convImage, kernels[1])
        convImage = tf.nn.relu(convImage)
        convImage = tf.nn.max_pool2d(convImage, ksize=2, strides=2, padding='SAME')
        # Apply the third convolutional layer
        convImage = self.convImage(convImage, kernels[2])
        convImage = tf.nn.relu(convImage)
        convImage = tf.nn.max_pool2d(convImage, ksize=2, strides=2, padding='SAME')

        convImage = self.convImage(convImage, kernels[3])
        convImage = tf.nn.relu(convImage)
        convImage = tf.nn.max_pool2d(convImage, ksize=2, strides=2, padding='SAME')

        convImage = self.convImage(convImage, kernels[4])
        convImage = tf.nn.relu(convImage)
        convImage = tf.nn.max_pool2d(convImage, ksize=2, strides=2, padding='SAME')


        flattened_output = tf.reshape(convImage, [1, -1])
        #drop out
        flattened_output = tf.nn.dropout(flattened_output, 0.5)
        #print("the shape of the flattened_output is: ", flattened_output.shape)
        logits = tf.matmul(flattened_output, self.fc_weights) + self.fc_bias
        probabilities = tf.nn.softmax(logits)
        return probabilities

        



    def forward(self, image, kernels, show=False):
        # Apply the first convolutional layer
        conv1_output = self.firstConv2D(image, kernels, 3)
        #print("the shape of the conv1_output is: ", conv1_output[:,:,:,0:1].shape)

        conv1_output = tf.nn.relu(conv1_output)
        conv1_output = tf.nn.max_pool2d(conv1_output, ksize=2, strides=2, padding='SAME')
        #normalize the image

        # Apply the second convolutional layer
        conv2_output = self.firstConv2D(conv1_output, kernels, 3)
        conv2_output = tf.nn.relu(conv2_output)
        conv2_output = tf.nn.max_pool2d(conv2_output, ksize=2, strides=2, padding='SAME')

        # Apply the third convolutional layer
        conv3_output = self.firstConv2D(conv2_output, kernels, 3)
        conv3_output = tf.nn.relu(conv3_output)
        conv3_output = tf.nn.max_pool2d(conv3_output, ksize=2, strides=2, padding='SAME')

        # Flatten the output
        flattened_output = tf.reshape(conv3_output, [1, -1])
        #drop out
        flattened_output = tf.nn.dropout(flattened_output, 0.5)
        #print("the shape of the flattened_output is: ", flattened_output.shape)

        logits = tf.matmul(flattened_output, self.fc_weights) + self.fc_bias
        probabilities = tf.nn.softmax(logits)

        return probabilities

    def train_step(self, pokemon, one_kurnel):
        #poke is a list of dictionaries
        for image in pokemon.image_in_batch:
            print("the length of the image is: ", len(pokemon.image_in_batch))
            #get the shape of the image
            print("the shape of the image is: ", image.shape)
            with tf.GradientTape() as tape:
                # forward pass
                predictions = self.forward2(image, one_kurnel)
                # compute loss
                loss_fn = tf.keras.losses.CategoricalCrossentropy()
                loss = loss_fn(pokemon.label, predictions)
            # compute gradients
            gradients = tape.gradient(loss, [self.fc_weights, self.fc_bias])
            # update weights and bias
            self.optimizer.apply_gradients(zip(gradients, [self.fc_weights, self.fc_bias]))

    def add_lapel_to_pokemon(self, pokemons):
        pokemons_with_labels = []
        for i, pokemon in enumerate(pokemons):
            label = tf.one_hot(pokemon.label, self.num_classes)
            label = tf.expand_dims(label, axis=0)
            pokemon.label = label
            pokemons_with_labels.append(pokemon)

    def validate_model(self, pokemon, one_kurnel):
        # poke is a list of dictionaries
        total_accuracy = 0.0
        total_loss = 0.0
        total_images = 0

        #print("Testing model accuracy with the validation set")
        for image in pokemon.validation_images:
            # Forward pass
            predictions = self.forward2(image, one_kurnel)
            # Compute loss
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(pokemon.label, predictions))
            total_loss += loss.numpy()
            
            # Compute accuracy
            predicted_label = tf.argmax(predictions, axis=1)
            true_label = tf.argmax(pokemon.label, axis=1)
            accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted_label, true_label), tf.float32))
            total_accuracy += accuracy.numpy()
            total_images += 1

        # Calculate average loss and accuracy over the validation set
        avg_accuracy = total_accuracy / total_images
        avg_loss = total_loss / total_images
        print(f"Validation Accuracy: {avg_accuracy:.4f}")
        print(f"Validation Loss: {avg_loss:.4f}")
        return avg_accuracy, avg_loss

def load_first_tensor_from_json(json_path):
    # Load JSON data
    list_of_tensors = []
    with open(json_path, 'r') as json_file:
        tensors_as_lists = json.load(json_file)
    for tensor_as_list in tensors_as_lists:
        list_of_tensors.append(tf.convert_to_tensor(tensor_as_list, dtype=tf.float32))

    filterSize = [3,6,12,24,48] #3+6+12+24+48 = 93
    kernels = []
    for i in range(len(list_of_tensors)):
        kernel = tf.reshape(list_of_tensors[i], [1, 3, 3])
        kernels.append(kernel)
        #stack the kernel

    #konvert the list of tensors :
    kernels = tf.stack(kernels, axis=3)
    print("the shape of the kernel is: ", kernels.shape)
    #devide the kernel into the different filter sizes
    list_of_tensors = []
    list_of_tensors.append(tf.convert_to_tensor(kernels[:,:,:,0:3], dtype=tf.float32))
    list_of_tensors.append(tf.convert_to_tensor(kernels[:,:,:,3:6], dtype=tf.float32))
    list_of_tensors.append(tf.convert_to_tensor(kernels[:,:,:,9:12], dtype=tf.float32)) 
    list_of_tensors.append(tf.convert_to_tensor(kernels[:,:,:,12:24], dtype=tf.float32)) 
    list_of_tensors.append(tf.convert_to_tensor(kernels[:,:,:,24:48], dtype=tf.float32))
    return list_of_tensors

def get_random_kernels():
    #make a list that is divisible by 3
    filterSize = [3,6,12,24,48] #3+6+12+24+48 = 93
    kernels = []
    for i in range(len(filterSize)):
        print("the filter size is: ", i)
        if i == 0:
            #create 1x1 tenesor
            kernel_2d = tf.constant([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=tf.float32)
            kernel_4d = tf.stack([kernel_2d, kernel_2d, kernel_2d], axis=-1)  # 3 channels
            kernel_4d = tf.expand_dims(kernel_4d, axis=0)  # Batch size of 1
            print("the shape of the kernel is: ", kernel_4d.shape)
            kernels.append(kernel_4d)
        else:
            kernel = tf.Variable(tf.random.normal([1, 16, 16, filterSize[i]]))
            kernels.append(kernel)

    return kernels

def save_tensor_to_json(tensor):
    tensor_as_list = tensor.numpy().tolist()
    with open('tensor_output.json', 'a') as json_file:
        json.dump(tensor_as_list, json_file)
        json_file.write('\n')
        split = "----------------------------------------------------\n"
        json_file.write(split)
        json_file.close()


if __name__ == '__main__':
    path = './Pokemons/Stellar_Crown/'
    pokemons = []

    newTensers = load_first_tensor_from_json('./bedstTenrs.json')

    for i, pokemon in enumerate(os.listdir(path)):
        if i == 50:
            break
        pokemons.append(Pokemon(os.path.join(path, pokemon), i))
    pokemons = pokemons[:3]


    cnn = CnnMordel((500,350,3), len(pokemons))
    #show the first image
    print("the length of the labels is: ", len(pokemons))
    #kernels = load_first_tensor_from_json('./tensors_output.json')
    #kernels = get_random_kernels()

    #add the labels to the pokemons
    cnn.add_lapel_to_pokemon(pokemons)
    print("the lapel for the first pokemon is: ", pokemons[0].label)
    batch_size = 3 #the size of how many images to train on:

    #get the images for the pokemons
    for pokemon in pokemons:
        pokemon.set_batch_images(batch_size)

    lossList = []
    accuracyList = []
    for i in range(40):
        print(f"Training epoch {i+1}/40")
        for pokemon in pokemons:
            cnn.train_step(pokemon, newTensers)
        print("the training is done")
        # Validate the model
        for pokemon in pokemons:
            accuracy, loss = cnn.validate_model(pokemon, newTensers)
            lossList.append(loss)
            accuracyList.append(accuracy)
        print("the validation is done")
        for pokemon in pokemons:
            pokemon.clear_batch_images()
            pokemon.set_batch_images(batch_size)
                
    #calculate the accuracy and the loss
    plt.plot(lossList)
    plt.plot(accuracyList)
    plt.show()
    print("the training is done")
    print("the validation reis list is: ")
    mean_loss = np.mean(lossList)
    mean_accuracy = np.mean(accuracyList)
    print("the mean loss is: ", mean_loss)
    print("the mean accuracy is: ", mean_accuracy)


