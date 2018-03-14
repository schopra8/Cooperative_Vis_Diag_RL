import tensorflow as tf
import numpy as np
from model import Dialog_Bots
from config import config
import json
import os
def main():
	model = Dialog_Bots(config)
	# if os.path.isfile("./final_model.pkl"):
	# 	with open("./final_model.pkl") as f:
	# 		Qbot.Q_regression, Qbot.Q, Abot.Q = json.load(f)
	# 		model.Qbot.Q_regression, model.Qbot.Q, model.Abot.Q = np.asarray(Qbot.Q_regression),np.asarray(Qbot.Q),np.asarray(Qbot.A)
	model.train(batch_size=model.config.batch_size,
              num_iterations=model.config.num_iterations,
              max_dialog_rounds=model.config.max_dialog_rounds)
	model.generate_graphs()
	# save = raw_input("Save? [y/n]")
	# if save == "y":
	# 	model.save()
	batch_generator = model.get_minibatches(2)
	image,caption,label = batch_generator.next()
	model.show_dialog(image, caption, label)

if __name__ == '__main__':
	main()
