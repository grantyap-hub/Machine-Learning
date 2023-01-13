import numpy as np


# parse the string with numpy as a string array
frog_data = np.loadtxt("frogs.csv", delimiter=';', dtype=str)
num_samples = frog_data.shape[0]

# get rid of headers
frog_data = np.delete(frog_data, [0, 1], 0)

# get rid of first two columns (unused in calculations)
frog_data = np.delete(frog_data, [0, 1], 1)

# TODO: convert categories to one-hot encodings
# 5) TR
print(frog_data.shape)
TR_encodings = np.zeros((num_samples, 10), dtype=int)

print(TR_encodings)
frog_data = np.insert(frog_data, 3, 0, axis=1)

def replace_with_one_hot(data, row_index):
    encodings = np.zeros(data.shape[0])
