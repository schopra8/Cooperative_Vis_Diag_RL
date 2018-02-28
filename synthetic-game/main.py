import tensorflow as tf
import numpy as np
from model import Dialog_Bots
from config import config

def main():
	model=Dialog_Bots(config)
	model.train(batch_size=model.config.batch_size,
              num_iterations=model.config.num_iterations)
	# model.generate_graphs()
	# model.show_dialog(image, caption)

if __name__ == '__main__':
	main()
