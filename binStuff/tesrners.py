import tensorflow as tf
import re
import json

def load_tensors_from_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Split the content by the separator
    tensor_strings = content.split("----------------------------------------------------")
    
    tensors = []
    for tensor_str in tensor_strings:
        # Clean up and extract numbers
        numbers = re.findall(r"-?\d+\.\d+e[+-]?\d+|-?\d+\.\d+", tensor_str)
        if numbers:
            # Convert numbers to float and reshape to the correct tensor shape (e.g., (3, 3, 3, 1))
            values = list(map(float, numbers))
            tensor = tf.constant(values, dtype=tf.float32)
            tensor = tf.reshape(tensor, (3, 3, 1, 1))  # Adjust shape as needed
            tensors.append(tensor)
    
    return tensors

def save_tensors_to_json(tensors, output_path):
    # Convert tensors to lists to make them JSON serializable
    tensors_as_lists = [tensor.numpy().tolist() for tensor in tensors]
    
    # Save to JSON
    with open(output_path, 'w') as json_file:
        json.dump(tensors_as_lists, json_file, indent=4)

# Load and save tensors
file_path = './randomfilter.txt'
output_json_path = 'bedstTenrs.json'
tensors = load_tensors_from_file(file_path)
save_tensors_to_json(tensors, output_json_path)

print(f"Tensors have been saved to {output_json_path}")
