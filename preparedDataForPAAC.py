file_path = "./Encoding_result/paacFeatureEncoding.txt"

output_path_train = './Encoding_result/train_ptmForPAAC.txt'
output_path_test = './Encoding_result/test_ptmForPAAC.txt'

import random
from pprint import pprint as print
from pubscripts import save_file


file = open(file_path, 'r')
lines = file.readlines()
l = len(lines)

l_train = int(l * 0.8)
print(l_train)

l_test = l - l_train
print(l_test)

# train test split
random.shuffle(lines)
train_lines = lines[:l_train]
test_lines = lines[l_train:]


# save both file in the same folder
# save_file.save_file(train_lines, 'csv' , output_path_train)
# save_file.save_file(test_lines, 'csv'  , output_path_test)

#
#  file save in python

with open(output_path_train, 'w') as f:
    for line in train_lines:
        f.write(line)

with open(output_path_test, 'w') as f:
    for line in test_lines:
        f.write(line)