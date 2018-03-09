import h5py

'''
images_train (82783, 4096)
images_val (40504, 4096)
'''

if __name__ == '__main__':
  filename = 'data_img.h5'
  f = h5py.File('../data/{}'.format(filename), 'r')
  print f.keys()
  for key in f.keys():
    print key, f[key].shape
