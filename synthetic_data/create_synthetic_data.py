
import numpy as np

"""Parameters for synthetic data
"""
num_colors = 4
num_shapes = 4
num_fills = 4
save_type = '%d'

#Data generation
color = np.arange(num_colors)
shape= np.arange(num_shapes)

dataset_size= num_colors*num_fills*num_shapes
shape_color = np.transpose([np.tile(shape, len(color)),np.repeat(color,len(shape))])


data=[]
for i in range(num_fills):
	for j in range(len(shape_color)):
		data.append(np.append(shape_color[j],i))
np.savetxt('synthetic_data.csv', np.asarray(data), fmt=save_type, delimiter=',', header="Shape, Color, Fill")
