import numpy as np

path = '/home/go_team/Users/oqbrady/Downloads/parler_unique_clean.txt'
train_path = '/home/go_team/datasets/train.txt'
valid_path = '/home/go_team/datasets/validation.txt'
data = []
with open(path) as fp:
  for line in fp:
    data.append(str(line))
train_size = int(0.8 * len(data))
train_set = []
for i in range(train_size):
  train_set = np.random.choice(data, train_size, replace = False)

validation_set = [line for line in data if line not in train_set]

np.savetxt(train_path, train_set)
np.savetxt(valid_path, validation_set)