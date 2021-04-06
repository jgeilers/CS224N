import numpy as np
import random

path = '/home/go_team/Users/oqbrady/Downloads/parler_unique_clean.txt'
train_path = '/home/go_team/datasets/train_parler.txt'
valid_path = '/home/go_team/datasets/validation_parler.txt'

f1 = open(train_path, "a")
f2 = open(valid_path, "a")
fp = open(path)
i = 0
for line in fp:
  i += 1
  choice = random.choices([0,1], weights = (20, 80), k=1)
  if choice[0] == 0:
    f2.write(str(line))
  else:
    f1.write(str(line))
  if i % 10000 == 0:
    print(i)
