import tensorflow as tf
from PIL import Image
import cv2
import numpy as np
import json
import random


def display_step(image, randomfielr=""):
    cv2.imshow("test", image)
    print(f"Displaying - Press 'q' to continue")
    isSaved = False
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return "skip"
        elif key == ord('s'):
            if isSaved:
                print("filter already saved")
                return "none"
            ##add the random filter to th button of an file
            file = open("randomfilter.txt", "a")
            file.write(str(randomfielr))
            file.write("\n")
            #sepreate the filters with a line
            file.write("----------------------------------------------------\n")
            file.close()
            print("filter saved")
            cv2.destroyAllWindows()
            isSaved = True
            return "none"


def create_random_filter():
    random_filter = tf.random.normal([3, 3, 1, 1])
    print(random_filter)
    return random_filter


def firstLayer(img, random_filter):
    # Convert the image to a tensor
    image = tf.convert_to_tensor(img, dtype=tf.float32)
    # Add batch dimension
    image = tf.expand_dims(image, axis=0)
    # Normalize the image
    image = image / 255.0
    # Apply random filter
    print(image.shape)
    #each cchannel is a filter
    print(image[:,:,:,0:1].shape)


    convBlue = tf.nn.conv2d(image[:,:,:,0:1], random_filter, strides=1, padding='SAME')
    convGreen = tf.nn.conv2d(image[:,:,:,1:2], random_filter, strides=1, padding='SAME')
    convRed = tf.nn.conv2d(image[:,:,:,2:3], random_filter, strides=1, padding='SAME')

    list_of_tensors = [convBlue, convGreen, convRed]
    #shuffling the list of tensors
    random.shuffle(list_of_tensors)

    display_step(np.squeeze(list_of_tensors[0].numpy()), random_filter)


    cv2.destroyAllWindows()



def testSomeFeature(image_path, random_filter):
    # Load the image using OpenCV and convert to RGB
    image = cv2.imread(image_path)
    image = cv2.resize(image, (350,500))  # Adjust to desired dimensions
    # Add batch dimension for model input
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Apply the first layer
    firstLayer(image, random_filter)


def load_first_tensor_from_json(json_path):
    # Load JSON data
    list_of_tensors = []
    with open(json_path, 'r') as json_file:
        tensors_as_lists = json.load(json_file)
    
    for tensor_as_list in tensors_as_lists:
        list_of_tensors.append(tf.convert_to_tensor(tensor_as_list, dtype=tf.float32))
    
    return list_of_tensors

# Usage example
if __name__ == "__main__":
    path = "./IMG_20241101_124547891.jpg"
    json_path = './tensors_output.json'
    for i in range(300):
        the_filer = create_random_filter()
        testSomeFeature(path, the_filer)

#    first_tensor = load_first_tensor_from_json(json_path)
#    print("First tensor loaded from JSON:")
#
#    for tensor in first_tensor:
#        testSomeFeature(path, tensor)
#
