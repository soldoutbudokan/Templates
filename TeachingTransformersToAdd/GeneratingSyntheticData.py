import numpy as np
import os

def generate_and_split_data(train_filename, test_filename, min_num=0, max_num=999, train_ratio=0.8):
    # Generate all possible combinations of 'a' and 'b'
    a = np.arange(min_num, max_num + 1)
    b = np.arange(min_num, max_num + 1)
    A, B = np.meshgrid(a, b)
    a_flat = A.flatten()
    b_flat = B.flatten()
    c_flat = a_flat + b_flat
    data = np.stack((a_flat, b_flat, c_flat), axis=1)
    
    # Shuffle data
    np.random.shuffle(data)
    
    # Split data
    num_examples = data.shape[0]
    num_train_examples = int(num_examples * train_ratio)
    train_data = data[:num_train_examples]
    test_data = data[num_train_examples:]
    
    # Save data to binary files in .npy format
    np.save(train_filename, train_data)
    np.save(test_filename, test_data)
    return train_filename, test_filename

def load_data(filename, mmap_mode=None):
    return np.load(filename, mmap_mode=mmap_mode)

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Create a "SyntheticData" folder if it doesn't exist
synthetic_data_folder = os.path.join(current_dir, "SyntheticData")
if not os.path.exists(synthetic_data_folder):
    os.makedirs(synthetic_data_folder)

# Generate and split data
train_file = os.path.join(synthetic_data_folder, 'addition_train.npy')
test_file = os.path.join(synthetic_data_folder, 'addition_test.npy')
generate_and_split_data(train_file, test_file)

# Load data
train_data = load_data(train_file, mmap_mode='r')
test_data = load_data(test_file, mmap_mode='r')

print(f"Training set size: {len(train_data)}")
print(f"Test set size: {len(test_data)}")

print(f"\nData saved in: {os.path.abspath(synthetic_data_folder)}")

print("\nSample of training data:")
for i in range(5):
    a, b, sum_ = train_data[i]
    print(f"{a} + {b} = {sum_}")

print("\nSample of test data:")
for i in range(5):
    a, b, sum_ = test_data[i]
    print(f"{a} + {b} = {sum_}")

# Function to convert numbers to string representation (for tokenization)
def number_to_string(num):
    return str(num)

# Example of how to use this in a data loader
def data_generator(array, batch_size):
    num_examples = array.shape[0]
    while True:
        idx = np.random.choice(num_examples, batch_size, replace=False)
        batch = array[idx]
        inputs = [f"{number_to_string(a)} + {number_to_string(b)} =" for a, b in batch[:, :2]]
        targets = [number_to_string(sum_) for sum_ in batch[:, 2]]
        yield inputs, targets

# Example usage of data generator
batch_size = 32
gen = data_generator(train_data, batch_size)
inputs, targets = next(gen)
print(f"\nExample batch of {batch_size}:")
print("First input:", inputs[0])
print("First target:", targets[0])