import numpy as np
import os

def generate_addition_data(filename, num_examples, min_num=0, max_num=999):
    # Generate random numbers for 'a' and 'b'
    a = np.random.randint(min_num, max_num + 1, num_examples, dtype=np.int32)
    b = np.random.randint(min_num, max_num + 1, num_examples, dtype=np.int32)
    c = a + b  # Calculate sum
    data = np.stack((a, b, c), axis=1)
    
    # Save data to a binary file in .npy format
    np.save(filename, data)
    
    return filename

def load_data(filename, mmap_mode=None):
    return np.load(filename, mmap_mode=mmap_mode)

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Create a "SyntheticData" folder if it doesn't exist
synthetic_data_folder = os.path.join(current_dir, "SyntheticData")
if not os.path.exists(synthetic_data_folder):
    os.makedirs(synthetic_data_folder)

# Generate training data
num_train_examples = 800000  # 800,000 examples for training
train_file = os.path.join(synthetic_data_folder, 'addition_train.npy')
generate_addition_data(train_file, num_train_examples)

# Generate test data
num_test_examples = 200000  # 200,000 examples for testing
test_file = os.path.join(synthetic_data_folder, 'addition_test.npy')
generate_addition_data(test_file, num_test_examples, min_num=0, max_num=999)

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