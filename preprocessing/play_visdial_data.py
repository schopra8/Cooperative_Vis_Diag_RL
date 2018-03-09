import h5py
import numpy as np

'''
img_pos_train (82783,)
img_pos_val (40504,)

cap_train (82783, 40)
cap_length_train (82783,)

cap_val (40504, 40)
cap_length_val (40504,)

ques_train (82783, 10, 20)
ques_length_train (82783, 10)

ans_train (82783, 10, 20)
ans_length_train (82783, 10)

ques_val (40504, 10, 20)
ques_length_val (40504, 10)

ans_val (40504, 10, 20)
ans_length_val (40504, 10)

ans_index_train (82783, 10)
ans_index_val (40504, 10)
opt_length_train (252298,)
opt_length_val (104242,)
opt_list_train (252298, 20)
opt_list_val (104242, 20)
opt_train (82783, 10, 100)
opt_val (40504, 10, 100)
'''

if __name__ == '__main__':
  filename = 'visdial_data.h5'
  f = h5py.File('../data/{}'.format(filename), 'r')
  # for key in f.keys():
    # print key, f[key].shape

  print f['img_pos_train'].dtype
  print f['img_pos_train'][0:3]
  print np.array_equal(f['img_pos_train'], np.arange(82783))

  # print f['ques_length_train'].dtype
  # print np.max(f['ques_length_train'])

  # print f['ques_train'].dtype
  # print np.max(f['ques_train'])

  # print f['ans_index_train'].dtype
  # print np.max(f['ans_index_train'])






