
import numpy as np


"""Parameters for synthetic data
"""
num_colors = 4
num_shapes = 4
num_fills = 4
save_type = '%d'
num_captions = 6

#Data generation
color = np.arange(num_colors)
shape= np.arange(num_shapes)

dataset_size= num_colors*num_fills*num_shapes
shape_color = np.transpose([np.tile(shape, len(color)),np.repeat(color,len(shape))])


data=[]
for i in range(num_fills):
	for j in range(len(shape_color)):
		data.append(np.append(shape_color[j],i))
captions = np.arange(num_captions)
data=np.asarray(data)
data = np.concatenate([np.repeat(data,num_captions, axis = 0),np.expand_dims(np.tile(captions.T,data.shape[0]),1)], axis = 1)

np.savetxt('synthetic_data.csv', data, fmt=save_type, delimiter=',', header="Shape, Color, Fill, Caption")
