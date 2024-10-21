import numpy as np
import os

def generate_addition_data(filename, num_examples, max_num=9999, min_num=0):
    data = np.random.randint(min_num, max_num + 1, (num_examples, 3), dtype=np.int32)
    data[:, 2] = data[:, 0] + data[:, 1]  # Calculate sum
    
    # Save data to a binary file
    data.tofile(filename)
    
    return filename

def create_memmap(filename, num_examples):
    return np.memmap(filename, dtype=np.int32, mode='r', shape=(num_examples, 3))

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Create a "Synthetic Data" folder if it doesn't exist
synthetic_data_folder = os.path.join(current_dir, "SyntheticData")
if not os.path.exists(synthetic_data_folder):
    os.makedirs(synthetic_data_folder)

# Generate training data
num_train_examples = 8000000  # 9 million examples for training
train_file = os.path.join(synthetic_data_folder, 'addition_train.bin')
generate_addition_data(train_file, num_train_examples)

# Generate test data with numbers outside the range used for training
num_test_examples = 2000000  # 1 million examples for testing
test_file = os.path.join(synthetic_data_folder, 'addition_test.bin')
generate_addition_data(test_file, num_test_examples, min_num=100000, max_num=199999)

# Create memory-mapped arrays
train_data = create_memmap(train_file, num_train_examples)
test_data = create_memmap(test_file, num_test_examples)

print(f"Training set size: {len(train_data)}")
print(f"Test set size: {len(test_data)}")
print(f"\nData saved in: {os.path.abspath(synthetic_data_folder)}")

print("\nSample of training data:")
for i in range(5):
    a, b, sum = train_data[i]
    print(f"{a} + {b} = {sum}")

print("\nSample of test data:")
for i in range(5):
    a, b, sum = test_data[i]
    print(f"{a} + {b} = {sum}")

# Function to convert numbers to string representation (for tokenization)
def number_to_string(num):
    return ' '.join(str(num))

# Example of how to use this in a data loader
def data_generator(mm_array, batch_size):
    num_examples = mm_array.shape[0]
    while True:
        idx = np.random.choice(num_examples, batch_size, replace=False)
        batch = mm_array[idx]
        inputs = [f"{number_to_string(a)} + {number_to_string(b)} =" for a, b in batch[:, :2]]
        targets = [number_to_string(sum) for sum in batch[:, 2]]
        yield inputs, targets

# Example usage of data generator
batch_size = 32
gen = data_generator(train_data, batch_size)
inputs, targets = next(gen)
print(f"\nExample batch of {batch_size}:")
print("First input:", inputs[0])
print("First target:", targets[0])
