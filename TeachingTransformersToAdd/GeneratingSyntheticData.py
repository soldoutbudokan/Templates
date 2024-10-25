import numpy as np
import os

def generate_addition_data(num_examples, min_num=0, max_num=9999):
    data = np.random.randint(min_num, max_num + 1, (num_examples, 2), dtype=np.int32)
    sums = data[:, 0] + data[:, 1]
    data = np.column_stack((data, sums))
    return data

def save_data(filename, data):
    # Save data to a binary file
    data.tofile(filename)

def create_memmap(filename, num_examples):
    return np.memmap(filename, dtype=np.int32, mode='r', shape=(num_examples, 3))

def get_input_strings(data):
    # Convert the first two columns (operands) to input strings
    input_strings = [f"{a}+{b}=" for a, b in data[:, :2]]
    return set(input_strings)

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Create a "SyntheticData" folder if it doesn't exist
synthetic_data_folder = os.path.join(current_dir, "SyntheticData")
if not os.path.exists(synthetic_data_folder):
    os.makedirs(synthetic_data_folder)

# Generate test data first
num_test_examples = 2000000  # 2 million examples for testing
test_file = os.path.join(synthetic_data_folder, 'addition_test.bin')
test_data = generate_addition_data(num_test_examples, min_num=0, max_num=9999)
save_data(test_file, test_data)

# Store test input strings to ensure no overlap with training data
test_input_strings = get_input_strings(test_data)

# Generate training data
num_train_examples_needed = 8000000  # 8 million examples for training
train_data_list = []
train_input_strings_set = set()
batch_size = 100000  # Generate data in batches to manage memory usage

print("Generating training data without overlap with test data...")
while len(train_data_list) < num_train_examples_needed:
    remaining_examples = num_train_examples_needed - len(train_data_list)
    current_batch_size = min(batch_size, remaining_examples)
    # Generate a batch of training data
    batch_data = generate_addition_data(current_batch_size, min_num=0, max_num=9999)
    batch_input_strings = [f"{a}+{b}=" for a, b in batch_data[:, :2]]
    # Filter out examples that are in the test set
    filtered_batch_data = []
    for i, input_str in enumerate(batch_input_strings):
        if input_str not in test_input_strings and input_str not in train_input_strings_set:
            filtered_batch_data.append(batch_data[i])
            train_input_strings_set.add(input_str)
    # Add filtered data to the training data list
    train_data_list.extend(filtered_batch_data)
    print(f"Collected {len(train_data_list)} / {num_train_examples_needed} training examples...")

# Convert the training data list to a numpy array
train_data = np.array(train_data_list, dtype=np.int32)

# Save training data to a file
train_file = os.path.join(synthetic_data_folder, 'addition_train.bin')
save_data(train_file, train_data)

# Create memory-mapped arrays
train_data_mm = create_memmap(train_file, len(train_data))
test_data_mm = create_memmap(test_file, num_test_examples)

print(f"\nTraining set size after removing duplicates: {len(train_data_mm)}")
print(f"Test set size: {len(test_data_mm)}")
print(f"\nData saved in: {os.path.abspath(synthetic_data_folder)}")

print("\nSample of training data:")
for i in range(5):
    a, b, sum = train_data_mm[i]
    print(f"{a} + {b} = {sum}")

print("\nSample of test data:")
for i in range(5):
    a, b, sum = test_data_mm[i]
    print(f"{a} + {b} = {sum}")

# Function to convert numbers to string representation (for tokenization)
def number_to_string(num):
    return str(num)

# Example of how to use this in a data loader
def data_generator(mm_array, batch_size):
    num_examples = mm_array.shape[0]
    while True:
        idx = np.random.choice(num_examples, batch_size, replace=False)
        batch = mm_array[idx]
        inputs = [f"{number_to_string(a)}+{number_to_string(b)}=" for a, b in batch[:, :2]]
        targets = [number_to_string(sum) for sum in batch[:, 2]]
        yield inputs, targets

# Example usage of data generator
batch_size = 32
gen = data_generator(train_data_mm, batch_size)
inputs, targets = next(gen)
print(f"\nExample batch of {batch_size}:")
print("First input:", inputs[0])
print("First target:", targets[0])