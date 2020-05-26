import pickle
import numpy as np

pkl_file = open('test_paths.pickle', 'rb')
data1 = pickle.load(pkl_file)
# print(len(data1))
# print(data1[1].shape)
print(data1)
# print(type(data1[0]))